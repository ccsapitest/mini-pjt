# mcp_server.py - 점심 식당 추천 Agent용 구조화 데이터 MCP 서버 (fastmcp)
"""노출하는 도구:
  - get_user_preference(name)                사용자 선호·제약 조회
  - update_user_preference(name, field, value) 사용자 선호·제약 수정 (되돌리기 어려움, HITL 필수)
  - list_restaurants(max_distance_km)         식당 목록 조회
  - get_menu(name)                            표준 메뉴 조회 (칼로리/나트륨/주의사항)

실행: python mcp_server.py
(stdio 모드로 대기합니다. 단독으로 쓰는 파일이 아니라
 클라이언트(agent.py)가 자식 프로세스로 띄워서 사용합니다)
"""
import json
import os
from fastmcp import FastMCP

mcp = FastMCP("lunch-recommendation-data")

# data/ 는 이 파일(src/) 기준 한 단계 위 폴더에 있습니다 (실행 위치와 무관하게 안전한 절대경로)
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "data")
USERS_PATH = os.path.join(DATA_DIR, "users.json")
RESTAURANTS_PATH = os.path.join(DATA_DIR, "restaurants.json")
MENUS_PATH = os.path.join(DATA_DIR, "menus.json")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


USERS = _load(USERS_PATH)
RESTAURANTS = _load(RESTAURANTS_PATH)
MENUS = _load(MENUS_PATH)

UPDATABLE_FIELDS = {"preferred_menus", "max_round_trip_minutes", "avoid_menus", "last_menu", "health_notes"}
CONSTRAINT_FIELDS = {"max_round_trip_minutes", "avoid_menus", "last_menu", "health_notes"}


@mcp.tool()
def get_user_preference(name: str) -> str:
    """사용자 이름으로 선호 메뉴와 제약(왕복 이동시간 상한, 회피 메뉴, 어제 먹은 메뉴, 특이사항)을 조회한다.

    요청자 본인과, 대화에서 명시적으로 언급된 동행자 조회에만 사용한다.
    그 외 제3자 이름으로는 호출하지 않는다.

    Args:
        name: 조회할 사용자의 정확한 한국어 이름
    """
    for user in USERS:
        if user["name"] == name:
            return json.dumps(user, ensure_ascii=False)
    names = ", ".join(u["name"] for u in USERS)
    return f"'{name}' 사용자를 찾을 수 없습니다. 등록된 사용자: {names}"


@mcp.tool()
def update_user_preference(name: str, field: str, value) -> str:
    """이미 등록된 사용자의 선호·제약 정보를 수정한다.

    되돌리기 어려운 변경이므로, 이 도구를 호출하기 전에 반드시 사람의 승인을 받아야 한다
    (에이전트 레이어의 HITL 게이트가 이를 강제한다 — 이 함수 자체는 승인 여부를 판단하지 않는다).

    어떠한 경우에도 새 사용자를 추가하거나 기존 사용자를 삭제하지 않는다 — 등록된 사용자의
    필드 값만 바꾼다. name이 등록되지 않은 이름이면 새로 만들지 않고 실패로 처리한다.

    Args:
        name: 수정할 사용자의 정확한 한국어 이름
        field: preferred_menus, max_round_trip_minutes, avoid_menus, last_menu, health_notes 중 하나
        value: 새로 설정할 값 (field에 맞는 타입: 리스트 또는 숫자)
    """
    if field not in UPDATABLE_FIELDS:
        return f"'{field}'는 수정 가능한 항목이 아닙니다. 가능한 항목: {', '.join(sorted(UPDATABLE_FIELDS))}"

    for user in USERS:
        if user["name"] == name:
            if field in CONSTRAINT_FIELDS:
                user["constraints"][field] = value
            else:
                user[field] = value
            with open(USERS_PATH, "w", encoding="utf-8") as f:
                json.dump(USERS, f, ensure_ascii=False, indent=2)
            return f"'{name}'의 {field}를(을) {value}로 수정했습니다."
    names = ", ".join(u["name"] for u in USERS)
    return f"'{name}' 사용자를 찾을 수 없습니다. 등록된 사용자: {names}"


@mcp.tool()
def list_restaurants(max_distance_km: float = 0) -> str:
    """등록된 식당 목록을 조회한다. 각 식당의 이름, 주소, 거리, 취급 메뉴명(식당이 부르는 이름 그대로)을 포함한다.

    Args:
        max_distance_km: 이 거리(km) 이내의 식당만 반환. 0이면 전체 반환
    """
    rows = RESTAURANTS if not max_distance_km else [
        r for r in RESTAURANTS if r["distance_km"] <= max_distance_km
    ]
    if not rows:
        return f"왕복 거리 {max_distance_km}km 이내 등록된 식당이 없습니다."
    return json.dumps(rows, ensure_ascii=False)


@mcp.tool()
def get_menu(name: str) -> str:
    """표준 메뉴 이름으로 칼로리·나트륨·주의사항을 조회한다.

    식당이 실제로 부르는 메뉴명(예: "왕 등심 돈까스")이 아니라 표준 메뉴명(예: "돈까스") 기준이다.
    식당 메뉴명을 표준 메뉴명으로 바꾸는 작업은 retriever의 정규화 검색을 먼저 사용한다.

    Args:
        name: 표준 메뉴 이름
    """
    for menu in MENUS:
        if menu["name"] == name:
            return json.dumps(menu, ensure_ascii=False)
    names = ", ".join(m["name"] for m in MENUS)
    return f"'{name}' 표준 메뉴를 찾을 수 없습니다. 등록된 표준 메뉴: {names}"


if __name__ == "__main__":
    mcp.run()   # stdio 모드로 실행
