# run_eval.py - test_queries.csv 17건을 실제 에이전트에 돌려서 결과를 수집한다.
"""채점(정답 판정) 자체는 하지 않는다. 각 케이스의 실제 답변/호출 도구/HITL 여부를
JSON으로 저장해서, round1_report.md 작성 시 expected_traits/forbidden/expected_tools와
사람이 대조할 수 있게 하는 용도다.

실행: python run_eval.py (src/ 폴더에서 실행)
"""
import asyncio
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from agent import build_app, shutdown  # noqa: E402
from guards import get_text  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "test_queries.csv")
OUT_PATH = os.path.join(BASE, "eval_results.jsonl")


async def run():
    agent = await build_app()
    try:
        await _run_rows(agent)
    finally:
        await shutdown()


async def _run_rows(agent):
    # utf-8-sig: Excel로 열었다 저장하면 파일 앞에 BOM이 붙는데, 일반 utf-8로 읽으면
    # 첫 컬럼명이 "id"가 아니라 "﻿id"가 되어 KeyError가 난다. utf-8-sig는 BOM이
    # 있으면 제거하고, 없어도 그냥 정상 읽는다.
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    out_f = open(OUT_PATH, "w", encoding="utf-8")
    for row in rows:
        thread_id = f"eval_{row['id']}"
        config = {"configurable": {"thread_id": thread_id}}
        print(f"[{row['id']}] {row['input']}", flush=True)
        try:
            r = await agent.ainvoke({"messages": [HumanMessage(row["input"])]}, config=config)
            answer = get_text(r["messages"][-1])
            tool_calls = [
                tc["name"]
                for m in r["messages"]
                for tc in (getattr(m, "tool_calls", None) or [])
            ]
            state = await agent.aget_state(config)
            interrupted = bool(state.next)
            interrupt_value = None
            if interrupted and state.tasks and state.tasks[0].interrupts:
                interrupt_value = str(state.tasks[0].interrupts[0].value)
        except Exception as e:
            answer = f"ERROR: {e}"
            tool_calls = []
            interrupted = False
            interrupt_value = None

        record = {
            "id": row["id"],
            "category": row["category"],
            "input": row["input"],
            "actual_answer": answer,
            "actual_tools": tool_calls,
            "interrupted": interrupted,
            "interrupt_value": interrupt_value,
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"  -> tools={tool_calls} interrupted={interrupted}", flush=True)
        print(f"  -> answer[:150]={answer[:150]}", flush=True)

    out_f.close()
    print(f"\n결과 저장: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
