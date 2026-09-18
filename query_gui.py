# query_gui.py - 점심 추천 Agent에 자유롭게 질의해보는 단독 GUI 도구
"""테스트(evaluation/)와는 별개로, 사람이 직접 이것저것 물어보기 위한 도구다.
src/ 안의 다른 파일은 전혀 건드리지 않고, agent.build_app()만 가져다 쓴다.

실행: python query_gui.py (mini-pjt 폴더 어디서 실행해도 무방)

주의: tkinter 창(GUI)이 뜨므로, 화면이 있는 환경에서 실행해야 한다.
HITL(선호도 수정) 승인 대기 중에는 입력창에 "승인" 또는 "거절"을 입력하면 된다.
"""
import asyncio
import json
import os
import sys
import threading
import tkinter as tk
import uuid
from tkinter import scrolledtext

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from agent import build_app  # noqa: E402
from guards import get_text  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_json(filename):
    with open(os.path.join(DATA_DIR, filename), encoding="utf-8") as f:
        return json.load(f)


def _user_tooltip_text(user: dict) -> str:
    c = user["constraints"]
    lines = [
        user["name"],
        f"선호 메뉴: {', '.join(user['preferred_menus']) or '없음'}",
        f"왕복 이동시간 상한: {c['max_round_trip_minutes']}분" if c["max_round_trip_minutes"] is not None else "왕복 이동시간 상한: 없음",
        f"회피 메뉴: {', '.join(c['avoid_menus']) or '없음'}",
        f"어제 먹은 메뉴: {c['last_menu'] or '없음'}",
        f"특이사항: {', '.join(c['health_notes']) or '없음'}",
    ]
    return "\n".join(lines)


def _restaurant_tooltip_text(restaurant: dict) -> str:
    menu_names = ", ".join(m["name"] for m in restaurant["menus"])
    lines = [
        restaurant["name"],
        restaurant["address"],
        f"편도 거리: {restaurant['distance_km']}km",
        f"메뉴: {menu_names}",
    ]
    return "\n".join(lines)


class ListboxTooltip:
    """리스트박스 항목에 마우스를 올리면 그 항목에 대한 정보를 풍선말로 보여준다."""

    def __init__(self, listbox: tk.Listbox, text_for_index):
        self.listbox = listbox
        self.text_for_index = text_for_index
        self.tip_window = None
        self.shown_index = None
        listbox.bind("<Motion>", self._on_motion)
        listbox.bind("<Leave>", lambda e: self._hide())

    def _on_motion(self, event):
        index = self.listbox.nearest(event.y)
        bbox = self.listbox.bbox(index)
        if not bbox or not (bbox[1] <= event.y <= bbox[1] + bbox[3]):
            self._hide()
            return
        if index == self.shown_index:
            self._move(event)
            return
        self._hide()
        text = self.text_for_index(index)
        if not text:
            return
        self.shown_index = index
        self.tip_window = tk.Toplevel(self.listbox)
        self.tip_window.wm_overrideredirect(True)
        label = tk.Label(
            self.tip_window, text=text, justify=tk.LEFT, background="#ffffe0",
            relief=tk.SOLID, borderwidth=1, padx=6, pady=4, font=("맑은 고딕", 9),
        )
        label.pack()
        self._move(event)

    def _move(self, event):
        if self.tip_window:
            x = self.listbox.winfo_rootx() + event.x + 16
            y = self.listbox.winfo_rooty() + event.y + 12
            self.tip_window.wm_geometry(f"+{x}+{y}")

    def _hide(self):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None
        self.shown_index = None


class AgentLoopThread(threading.Thread):
    """tkinter(동기)와 에이전트(비동기)를 함께 쓰기 위한 전용 asyncio 이벤트 루프 스레드."""

    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()
        self.agent = None
        self.ready = threading.Event()
        self.error = None

    def run(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.agent = self.loop.run_until_complete(build_app())
        except Exception as e:
            self.error = e
        finally:
            self.ready.set()
        self.loop.run_forever()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


class QueryGUI(tk.Tk):
    def __init__(self, agent_thread: AgentLoopThread):
        super().__init__()
        self.title("점심 추천 Agent - 자유 질의")
        self.geometry("1020x600")
        self.agent_thread = agent_thread
        self.thread_id = f"gui_{uuid.uuid4().hex[:8]}"
        self.pending_interrupt = False

        chat_frame = tk.Frame(self)
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.output = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state="disabled")
        self.output.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        input_frame = tk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.entry = tk.Entry(input_frame)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda e: self.on_send())

        tk.Button(input_frame, text="전송", command=self.on_send).pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(input_frame, text="새 대화", command=self.on_reset).pack(side=tk.LEFT, padx=(4, 0))

        self._build_reference_panel()

        self._append("[안내] 에이전트 준비 중... (MCP 서버 기동, 잠시 기다려주세요)\n")
        threading.Thread(target=self._wait_ready, daemon=True).start()

    def _build_reference_panel(self):
        """검증용 참고 패널 — 사용자/식당 목록을 보여주고, 마우스를 올리면 상세 정보를 풍선말로
        띄운다. data/*.json을 직접 읽어서 표시만 하는 용도라, 에이전트가 세션 중에 임시로 바꾼
        값(update_user_preference)은 반영되지 않는다 (그 값은 MCP 서버 프로세스 메모리에만 있다)."""
        panel = tk.Frame(self, width=300)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        panel.pack_propagate(False)

        users = _load_json("users.json")
        restaurants = _load_json("restaurants.json")

        tk.Label(panel, text="사용자 목록 (마우스오버: 상세)", anchor="w").pack(fill=tk.X)
        user_list = tk.Listbox(panel, height=12, exportselection=False)
        for u in users:
            user_list.insert(tk.END, u["name"])
        user_list.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        ListboxTooltip(user_list, lambda i: _user_tooltip_text(users[i]) if i < len(users) else "")

        tk.Label(panel, text="식당 목록 (마우스오버: 상세)", anchor="w").pack(fill=tk.X)
        restaurant_list = tk.Listbox(panel, height=12, exportselection=False)
        for r in restaurants:
            restaurant_list.insert(tk.END, r["name"])
        restaurant_list.pack(fill=tk.BOTH, expand=True)
        ListboxTooltip(
            restaurant_list,
            lambda i: _restaurant_tooltip_text(restaurants[i]) if i < len(restaurants) else "",
        )

    def _wait_ready(self):
        self.agent_thread.ready.wait()
        if self.agent_thread.error:
            self.after(0, lambda: self._append(f"[오류] 에이전트 초기화 실패: {self.agent_thread.error}\n"))
        else:
            self.after(0, lambda: self._append("[안내] 준비 완료. 질문을 입력하세요 (예: 송덕삼이랑 점심 먹을만한 식당 추천해줘).\n"))

    def _append(self, text: str):
        self.output.configure(state="normal")
        self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.output.configure(state="disabled")

    def on_reset(self):
        self.thread_id = f"gui_{uuid.uuid4().hex[:8]}"
        self.pending_interrupt = False
        self._append("\n[안내] 새 대화로 초기화했습니다 (이전 맥락 없음).\n")

    def on_send(self):
        text = self.entry.get().strip()
        if not text or self.agent_thread.agent is None:
            return
        self.entry.delete(0, tk.END)
        self._append(f"\n> {text}\n")

        if self.pending_interrupt:
            decision_type = "approve" if text in ("승인", "approve", "yes", "y") else "reject"
            payload = Command(resume={"decisions": [{"type": decision_type}]})
            self.pending_interrupt = False
        else:
            payload = {"messages": [HumanMessage(text)]}

        config = {"configurable": {"thread_id": self.thread_id}}

        async def call():
            agent = self.agent_thread.agent
            r = await agent.ainvoke(payload, config=config)
            answer = get_text(r["messages"][-1])
            state = await agent.aget_state(config)
            interrupt_info = None
            if state.next and state.tasks and state.tasks[0].interrupts:
                interrupt_info = state.tasks[0].interrupts[0].value
            return answer, interrupt_info

        future = self.agent_thread.submit(call())

        def on_done(fut):
            try:
                answer, interrupt_info = fut.result()
            except Exception as e:
                answer, interrupt_info = f"[오류] {e}", None

            def update_ui():
                self._append(f"{answer}\n")
                if interrupt_info:
                    self.pending_interrupt = True
                    self._append(f"[승인 대기] {interrupt_info}\n(승인하려면 '승인', 거절하려면 '거절' 입력 후 전송)\n")

            self.after(0, update_ui)

        future.add_done_callback(on_done)


def main():
    agent_thread = AgentLoopThread()
    agent_thread.start()
    app = QueryGUI(agent_thread)
    app.mainloop()


if __name__ == "__main__":
    main()
