# mcp_server.py - 점심 식당 추천 Agent용 구조화 데이터 MCP 서버 (fastmcp)
"""노출하는 도구:
  - get_user_preference(name)                사용자 선호·제약 조회 (1명)
  - get_group_constraints(names)             일행 전체 조회 + 병합(사장님 오버라이드 포함)까지 한 번에
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
LIST_FIELDS = {"preferred_menus", "avoid_menus", "health_notes"}
# 점심시간 왕복 이동시간의 상식적인 상한. 이보다 큰 값은 실존하지 않는 개념(예: "나트륨
# 5000mg 제한")을 이 필드에 억지로 끼워맞춘 경우일 가능성이 높다.
MAX_ROUND_TRIP_MINUTES_CAP = 180


def _validate_field_value(field: str, value) -> str | None:
    """value가 field의 실제 의미에 맞는 타입/범위인지 검사한다. 문제없으면 None, 문제가
    있으면 사용자에게 보여줄 오류 메시지를 반환한다.
    """
    if field == "max_round_trip_minutes":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"max_round_trip_minutes는 분 단위 숫자여야 합니다 (받은 값: {value!r})."
        if not (1 <= value <= MAX_ROUND_TRIP_MINUTES_CAP):
            return (
                f"max_round_trip_minutes 값 {value}은(는) 상식적인 범위(1~"
                f"{MAX_ROUND_TRIP_MINUTES_CAP}분)를 벗어납니다. 요청하신 내용이 실제로는 "
                "이동시간이 아닌 다른 개념(예: 나트륨 수치)이라면, 그런 필드는 존재하지 않는다고 "
                "안내해야 합니다."
            )
        return None
    if field == "last_menu":
        if not isinstance(value, str) or not value.strip():
            return f"last_menu는 비어있지 않은 문자열이어야 합니다 (받은 값: {value!r})."
        return None
    if field in LIST_FIELDS:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            return f"{field}는 문자열 리스트여야 합니다 (받은 값: {value!r})."
        return None
    return None


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
def get_group_constraints(names: list[str]) -> str:
    """일행 전체(요청자 본인 포함)의 선호·제약을 조회하고, 그룹 기준으로 병합까지 끝낸 결과를 돌려준다.

    사람마다 get_user_preference를 따로 부르고 병합 규칙(최솟값/합집합/사장님 오버라이드)을
    직접 계산하지 않는다 — 이 도구가 이미 다 계산해서 "merged" 안에 넣어준다. 그룹 추천을 할
    때는 반드시 이 도구 하나만 부르고, merged 값을 그대로 필터링 기준으로 쓴다.

    병합 규칙 (이미 적용되어 있음, 다시 계산하지 않는다):
    - 동행자 중 "사장님"이 있으면, 사장님 외 다른 모든 사람의 선호·제약은 전부 무시한다.
      "선호 vs 회피"처럼 정면으로 충돌하는 것처럼 보여도 예외 없이 사장님만 따른다
      (merged.override_by 가 "사장님"으로 표시된다). 이때 members 목록에서도 사장님이
      아닌 사람들의 avoid_menus/health_notes/last_menu는 아예 빠지고 이름과 선호메뉴만
      남는다 — 애초에 참고할 회피 정보가 없으니 "충돌"을 재해석할 필요도 없다.
    - 사장님이 없으면: 왕복 이동시간 상한은 값이 있는 사람들 중 최솟값(없는 사람은 제외),
      회피 메뉴·어제 먹은 메뉴·특이사항은 전원의 값을 합집합으로 합친다.

    Args:
        names: 요청자 본인을 포함한 일행 전체의 이름 목록
    """
    members = []
    not_found = []
    for name in names:
        user = next((u for u in USERS if u["name"] == name), None)
        if user is None:
            not_found.append(name)
        else:
            members.append(user)

    if not members:
        registered = ", ".join(u["name"] for u in USERS)
        return json.dumps(
            {"not_found": not_found, "merged": None, "note": f"등록된 사용자: {registered}"},
            ensure_ascii=False,
        )

    boss = next((u for u in members if u["name"] == "사장님"), None)
    if boss is not None:
        merged = {
            "override_by": "사장님",
            "preferred_menus": boss["preferred_menus"],
            "max_round_trip_minutes": boss["constraints"]["max_round_trip_minutes"],
            "avoid_menus": boss["constraints"]["avoid_menus"],
            "last_menus": [boss["constraints"]["last_menu"]] if boss["constraints"]["last_menu"] else [],
            "health_notes": boss["constraints"]["health_notes"],
        }
        # 사장님 오버라이드 시에는 다른 동행자의 회피 메뉴/특이사항/최근 메뉴를 응답에서
        # 아예 빼버린다. 모델이 이 원본 데이터를 보면 merged가 이미 무시하기로 한 "충돌"을
        # 스스로 다시 찾아내서 추천을 거부하는 문제가 있었기 때문이다 (프롬프트 지시만으로는
        # 해결되지 않음 — 데이터 자체를 안 보이게 해야 한다).
        members = [
            u if u["name"] == "사장님" else {"name": u["name"], "preferred_menus": u["preferred_menus"]}
            for u in members
        ]
    else:
        caps = [
            u["constraints"]["max_round_trip_minutes"]
            for u in members
            if u["constraints"]["max_round_trip_minutes"] is not None
        ]
        merged = {
            "override_by": None,
            "preferred_menus": sorted({m for u in members for m in u["preferred_menus"]}),
            "max_round_trip_minutes": min(caps) if caps else None,
            "avoid_menus": sorted({m for u in members for m in u["constraints"]["avoid_menus"]}),
            "last_menus": sorted({
                u["constraints"]["last_menu"] for u in members if u["constraints"]["last_menu"]
            }),
            "health_notes": sorted({n for u in members for n in u["constraints"]["health_notes"]}),
        }

    return json.dumps(
        {"not_found": not_found, "members": members, "merged": merged},
        ensure_ascii=False,
    )


@mcp.tool()
def update_user_preference(name: str, field: str, value) -> str:
    """이미 등록된 사용자의 선호·제약 정보를 이번 세션 동안만 임시로 수정한다.

    이 도구는 실제 사용자 DB(data/users.json) 파일을 절대 건드리지 않는다 — 실제 DB 수정은
    별도의 DB 관리 솔루션(이 프로젝트 범위 밖)의 몫이고, 이 Agent는 사람이 승인하더라도 DB를
    직접 고치지 않는다. 여기서 하는 일은 지금 떠 있는 MCP 서버 프로세스의 메모리 상 값만
    바꾸는 것이라, 이 프로세스가 재시작되면(=새 세션이 시작되면) 원래 값으로 돌아간다.

    되돌리기 어려운 변경은 아니지만(세션이 끝나면 자동으로 사라짐), 그래도 사용자가 눈치채지
    못하는 사이에 조건이 바뀌면 안 되므로, 이 도구를 호출하기 전에 반드시 사람의 승인을 받아야
    한다 (에이전트 레이어의 HITL 게이트가 이를 강제한다 — 이 함수 자체는 승인 여부를 판단하지
    않는다).

    어떠한 경우에도 새 사용자를 추가하거나 기존 사용자를 삭제하지 않는다 — 등록된 사용자의
    필드 값만 바꾼다. name이 등록되지 않은 이름이면 새로 만들지 않고 실패로 처리한다.

    Args:
        name: 수정할 사용자의 정확한 한국어 이름
        field: preferred_menus, max_round_trip_minutes, avoid_menus, last_menu, health_notes 중 하나
        value: 새로 설정할 값 (field에 맞는 타입: 리스트 또는 숫자)
    """
    if field not in UPDATABLE_FIELDS:
        return f"'{field}'는 수정 가능한 항목이 아닙니다. 가능한 항목: {', '.join(sorted(UPDATABLE_FIELDS))}"

    # 프롬프트 지시만으로는, 존재하지 않는 개념(예: "나트륨 5000mg 제한")을 모델이 기존 필드에
    # 억지로 끼워맞춰 호출하는 경우를 막지 못했다 (max_round_trip_minutes에 5000을 넣거나,
    # health_notes에 "나트륨 5000mg 제한" 같은 지어낸 문구를 넣는 식). 여기서 타입/범위를
    # 검증해 이런 값이 실제로 저장되기 전에 걸러낸다.
    error = _validate_field_value(field, value)
    if error:
        return error

    for user in USERS:
        if user["name"] == name:
            if field in CONSTRAINT_FIELDS:
                user["constraints"][field] = value
            else:
                user[field] = value
            # 의도적으로 파일에 쓰지 않는다 — data/users.json(실제 DB)은 절대 수정하지 않고,
            # 이번 세션(현재 MCP 서버 프로세스) 메모리에만 반영해 다음 조회부터 임시로 적용되게 한다.
            return f"'{name}'의 {field}를(을) {value}로 (이번 세션 동안만) 수정했습니다."
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
        return f"편도 거리 {max_distance_km}km 이내 등록된 식당이 없습니다."
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
