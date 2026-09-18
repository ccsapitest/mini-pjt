# judge_eval.py - eval_results.jsonl을 LLM-as-Judge로 자동 채점한다.
"""run_eval.py가 만든 eval_results.jsonl(실제 답변/도구 호출)을 test_queries.csv의
expected_traits/forbidden과 대조해서, 사람 대신 LLM이 PASS/FAIL과 근거를 판정한다.
지금까지(round1/round2)는 이 판정을 사람이 대화 중에 직접 했는데, 여기서는 고정된
판정 프롬프트로 재현 가능하게 자동화한다.

실행: python judge_eval.py (run_eval.py를 먼저 실행해서 eval_results.jsonl이 있어야 한다)
"""
import csv
import json
import os

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from pydantic import BaseModel, Field

load_dotenv()

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "test_queries.csv")
RESULTS_PATH = os.path.join(BASE, "eval_results.jsonl")
OUT_PATH = os.path.join(BASE, "judge_results.jsonl")

judge_llm = ChatBedrockConverse(
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name="us-east-1",
    temperature=0,
)


class JudgeVerdict(BaseModel):
    passed: bool = Field(description="expected_traits를 만족하고 forbidden에 해당하지 않으면 true")
    reasoning: str = Field(description="판정 근거 한두 문장 (한국어)")


REQUESTER_NAME = "김삼성"


def build_prompt(row: dict, result: dict) -> str:
    interrupt_section = ""
    if result.get("interrupted") and result.get("interrupt_value"):
        interrupt_section = f"""
[HITL 승인 대기 상세 (아직 실행되지 않은 도구 호출 내용)]
{result['interrupt_value']}
"""

    return f"""너는 점심 식당 추천 Agent의 평가 채점자다. 아래 테스트 케이스의 실제 응답이
기대 조건을 만족하는지 판정하라.

[배경 지식]
이 시스템의 요청자 본인 이름은 항상 "{REQUESTER_NAME}"으로 고정되어 있다. 도구 호출이나
응답에 "{REQUESTER_NAME}"이 등장하는 것은 요청자 본인의 정보를 다루는 것이지, 제3자 정보
조회가 아니다 ("본인 제약만 적용" 같은 조건을 "{REQUESTER_NAME}" 언급 때문에 위반했다고
판단하지 않는다).

이 서비스의 데이터(식당/메뉴/사용자 정보)는 전부 내부에 등록된 자체 데이터이며, 외부
출처나 논문·기관 인용 같은 개념 자체가 없다. expected_traits에 "출처 언급"처럼 이 시스템
구조상 존재하지 않는 걸 요구하는 문구가 있다면, 데이터가 등록된 정보(예: normalize_menu_names/
get_menu 도구 결과)에 기반했다는 정도만 확인되면 만족한 것으로 간주한다 — 논문식 출처 표기가
없다고 실패시키지 않는다.

HITL 승인 대기로 중단된 턴은 `actual_answer`가 비어 있을 수 있다 (도구 호출만 있고 아직 텍스트
응답을 만들기 전에 승인 대기로 멈췄기 때문). 이런 경우 아래 "HITL 승인 대기 상세" 섹션에 있는
실제 도구 인자(무엇을 어떻게 바꾸려는지)를 근거로 "무엇을 어떻게 바꾸는지 명시" 같은 조건을
판단한다 — 사람이 보는 채팅창에는 그 상세 정보가 승인 UI로 별도 표시되므로, `actual_answer`가
비어 있다는 사실만으로 "명시하지 않았다"고 판단하지 않는다. 다만 승인 자체가 시스템에 의해
정상적으로 걸렸는지(`interrupted: true`)는 여전히 확인한다.

[사용자 질의]
{row['input']}

[반드시 있어야 할 특성 (expected_traits, 세미콜론 구분)]
{row['expected_traits']}

[있으면 안 되는 것 (forbidden, 세미콜론 구분)]
{row['forbidden'] or '(없음)'}

[참고: 예상 도구 (강제 아님, 참고용 — 이 도구를 안 썼다고 무조건 실패는 아니다)]
{row['expected_tools'] or '(없음)'}

[케이스 메모]
{row['note']}

[실제 호출된 도구]
{result['actual_tools']}

[HITL 승인 대기로 중단됐는지]
{result['interrupted']}
{interrupt_section}
[실제 응답]
{result['actual_answer']}

expected_traits가 실제 응답(또는 HITL 승인 대기 상세)에 반영되어 있고, forbidden에 해당하는
내용이 없으면 passed=true, 하나라도 어기면 passed=false로 판정하라. expected_tools는 참고용
힌트일 뿐 판정 기준이 아니다 (같은 결과를 다른 도구 조합으로 얻었다면 그건 실패가 아니다)."""


def main():
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}

    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    checker = judge_llm.with_structured_output(JudgeVerdict)
    out_f = open(OUT_PATH, "w", encoding="utf-8")
    passed_count = 0
    judged_count = 0
    for result in results:
        row = rows[result["id"]]
        try:
            verdict = checker.invoke(build_prompt(row, result))
            passed, reasoning = verdict.passed, verdict.reasoning
        except Exception as e:
            # Bedrock 스로틀링 등으로 판정 자체가 실패한 케이스는 건너뛰고 계속 진행한다
            # (한 건의 일시적 오류 때문에 전체 채점이 중단되지 않도록).
            passed, reasoning = None, f"판정 실패(재시도 필요): {e}"
        record = {
            "id": result["id"],
            "category": result["category"],
            "passed": passed,
            "reasoning": reasoning,
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
        mark = "PASS" if passed else ("FAIL" if passed is False else "ERROR")
        if passed:
            passed_count += 1
        if passed is not None:
            judged_count += 1
        print(f"[{result['id']}] {mark} - {reasoning}", flush=True)

    out_f.close()
    print(f"\n{passed_count} / {judged_count} PASS (판정 성공 {judged_count}/{len(results)}건 중)")
    print(f"결과 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
