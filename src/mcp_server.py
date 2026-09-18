# mcp_server.py - 점심 식당 추천 Agent용 구조화 데이터 MCP 서버 (fastmcp)
"""노출하는 도구:
  - get_user_preference(name)                사용자 선호·제약 조회 (1명)
  - get_group_constraints(names)             일행 전체 조회 + 하드 필터(이동시간 상한, 특이사항)만 병합
  - rank_restaurants(candidates, members, ...) 점수제로 식당 순위를 매겨 상위 N곳 반환
  - update_user_preference(name, field, value) 사용자 선호·제약 수정 (되돌리기 어려움, HITL 필수)
  - list_restaurants(max_distance_km)         식당 목록 조회
  - get_menu(name)                            표준 메뉴 조회 (칼로리/나트륨/주의사항)

실행: python mcp_server.py
(stdio 모드로 대기합니다. 단독으로 쓰는 파일이 아니라
 클라이언트(agent.py)가 자식 프로세스로 띄워서 사용합니다)
"""
import json
import os
import sys
from fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import _compute_minutes  # noqa: E402

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

# 표준 메뉴 이름 -> 카테고리 목록 (예: "돈까스" -> ["일식"]). preferred_menus/avoid_menus는
# "한식"/"고기" 같은 카테고리로도, "뼈해장국" 같은 정확한 메뉴명으로도 올 수 있어서
# rank_restaurants에서 두 가지 다 매칭해야 한다.
MENU_CATEGORIES = {m["name"]: m.get("categories", []) for m in MENUS}

BOSS_WEIGHT = 20
POINTS_PER_MATCH = 10
DISTANCE_PENALTY_PER_5MIN = 5

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


def _lookup_members(names: list[str]) -> tuple[list[dict], list[str]]:
    members = []
    not_found = []
    for name in names:
        user = next((u for u in USERS if u["name"] == name), None)
        if user is None:
            not_found.append(name)
        else:
            members.append(user)
    return members, not_found


def _hard_filters(members: list[dict]) -> dict:
    """max_round_trip_minutes/health_notes만 계산한다 (avoid_menus/last_menu/preferred_menus는
    더 이상 여기서 다루지 않는다 — rank_restaurants가 이름만 받아서 내부에서 직접 점수로 반영한다)."""
    boss = next((u for u in members if u["name"] == "사장님"), None)
    if boss is not None:
        return {
            "override_by": "사장님",
            "max_round_trip_minutes": boss["constraints"]["max_round_trip_minutes"],
            "health_notes": [
                {"note": n, "who": ["사장님"]} for n in boss["constraints"]["health_notes"]
            ],
        }
    caps = [
        u["constraints"]["max_round_trip_minutes"]
        for u in members
        if u["constraints"]["max_round_trip_minutes"] is not None
    ]
    notes_to_who: dict[str, list[str]] = {}
    for u in members:
        for n in u["constraints"]["health_notes"]:
            notes_to_who.setdefault(n, []).append(u["name"])
    return {
        "override_by": None,
        "max_round_trip_minutes": min(caps) if caps else None,
        "health_notes": [
            {"note": n, "who": who} for n, who in sorted(notes_to_who.items())
        ],
    }


@mcp.tool()
def get_group_constraints(names: list[str]) -> str:
    """일행 전체(요청자 본인 포함)의 하드 필터(이동시간 상한, 건강 특이사항)만 조회한다.

    이 도구는 일부러 사람별 선호·회피·최근 메뉴(preferred_menus/avoid_menus/last_menu)는
    돌려주지 않는다 — 그건 점수 계산용이라 rank_restaurants가 이름만 받아서 내부에서 직접
    처리한다. 여기서 받는 merged.health_notes는 search_menu_by_caution으로 검색해 해당 표준
    메뉴를 후보에서 제외하는 데 쓰고, merged.max_round_trip_minutes는 답변에서 안내하는 용도로만
    쓴다 (실제 필터링은 rank_restaurants가 이름을 받아 내부에서 다시 계산해 적용한다).

    merged.health_notes는 문자열 배열이 아니라 {"note": "날것 주의", "who": ["최민준"]} 형태의
    객체 배열이다 — 누구의 특이사항인지 반드시 이 who를 보고 판단하고, "일행 모두"나 "두 분 다"처럼
    함부로 일반화하지 않는다. who에 없는 사람에게는 그 특이사항이 없는 것이다.

    병합 규칙 (이미 적용되어 있음, 다시 계산하지 않는다):
    - 동행자 중 "사장님"이 있으면, 이동시간 상한·건강 특이사항은 사장님 기준으로만 정해진다
      (merged.override_by가 "사장님"으로 표시됨). 다른 동행자의 회피 메뉴나 특이사항 때문에
      "충돌"이라고 되묻지 않는다 — 그 사람들 정보는 애초에 이 도구가 보여주지 않는다.
    - 사장님이 없으면: 이동시간 상한은 값이 있는 사람들 중 최솟값, 건강 특이사항은 전원의
      값을 모으되 각 항목마다 who로 누구 것인지 표시한다.

    Args:
        names: 요청자 본인을 포함한 일행 전체의 이름 목록
    """
    members, not_found = _lookup_members(names)
    if not members:
        registered = ", ".join(u["name"] for u in USERS)
        return json.dumps(
            {"not_found": not_found, "merged": None, "note": f"등록된 사용자: {registered}"},
            ensure_ascii=False,
        )
    return json.dumps(
        {"not_found": not_found, "merged": _hard_filters(members)},
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


def _matches(target: str, standard_menu_name: str) -> bool:
    """avoid_menus/preferred_menus/last_menu의 값(target)이 표준 메뉴 이름이나
    카테고리("한식", "고기" 등) 어느 쪽으로 와도 그 표준 메뉴에 매칭되는지 판단한다."""
    if target == standard_menu_name:
        return True
    return target in MENU_CATEGORIES.get(standard_menu_name, [])


@mcp.tool()
def rank_restaurants(
    names: list[str] | str,
    candidates: list[dict] | str,
    weather: str = "맑음",
    mode: str = "도보",
    top_n: int = 3,
    ignore_time_cap: bool = False,
) -> str:
    """일행 이름과 식당 후보만 주면, 하드 필터·점수 계산·순위 결정을 전부 이 도구가 끝내서
    상위 N곳을 돌려준다. 호출한 뒤에는 이 결과를 그대로 답변에 옮기면 된다 (다시 필터링하거나
    직접 순위를 재계산하지 않는다).

    일행 각자의 preferred_menus/avoid_menus/last_menu와 사장님 가중치(20)는 이 도구가 이름으로
    직접 조회해서 내부에서만 쓴다 — get_group_constraints처럼 그 값을 텍스트로 돌려주지 않는다.
    그러니 "이 사람이 이걸 회피하니까 문제 아닌가?" 같은 판단을 스스로 하지 않는다 — 그 데이터를
    볼 필요도, 볼 수도 없다. 사장님이 일행에 있으면 다른 사람의 회피/최근 메뉴가 이 식당에
    있어도 그냥 이 도구가 알아서 가중치 계산에 반영할 뿐이니, 결과가 나오기 전에 미리
    "괜찮을까요?"라고 되묻지 않는다.

    처리 순서:
    1. 하드 필터: 이동시간 상한(사장님이 있으면 사장님 기준, 없으면 최솟값)을 넘는 식당은
       제외한다 (health_notes 기반 주의 메뉴 제외는 이 도구를 부르기 전에
       search_menu_by_caution으로 미리 처리해서, candidates의 standard_menus에서
       빼놓고 넘긴다 — 이 도구는 그 목록을 그대로 신뢰한다).
    2. 점수 계산: 살아남은 각 식당에 대해, 일행 각자마다 다음을 한 번씩만 적용한다
       (같은 카테고리 메뉴가 여러 개 있어도 중복 가점/감점하지 않는다). 사장님이 있으면
       사장님의 가중치는 20, 나머지는 1이다:
       - 그 식당에 preferred_menus와 매칭되는(정확한 이름 또는 카테고리) 표준 메뉴가
         하나라도 있으면 +10 * weight
       - avoid_menus와 매칭되는 메뉴가 하나라도 있으면 -10 * weight
       - last_menu와 매칭되는 메뉴가 있으면 -10 * weight
       거리 페널티: mode가 "도보"일 때만, 편도 이동시간 5분마다 -5점 (대중교통/차량은 감점 없음).
    3. 점수 내림차순으로 정렬해 상위 N곳만 반환한다.

    Args:
        names: 요청자 본인을 포함한 일행 전체의 이름 목록 (get_group_constraints에 넘긴 것과 동일)
        candidates: [{"name": 식당명, "distance_km": 편도 거리,
                      "standard_menus": [이미 정규화되고 주의 메뉴는 제외된 표준 메뉴명 목록]}]
        weather: 오늘 날씨 (왕복 이동시간 계산에 사용)
        mode: 이동수단 ("도보", "대중교통", "차량")
        top_n: 상위 몇 곳을 반환할지 (기본 3, 사용자가 다른 개수를 요청하면 그 값으로)
        ignore_time_cap: True면 이동시간 상한 하드 필터를 이번 호출에서만 건너뛴다
            ("OO 이동시간 제한 무시하고 추천해줘"처럼 이번 요청 한정 완화일 때만 True로 준다)
    """
    # 모델이 가끔 리스트 인자를 JSON 문자열로 직렬화해서 넘길 때가 있다 (예: candidates를
    # '[{"name": ...}]' 같은 문자열로 전달). 스키마에서 바로 거부하지 않고 여기서 파싱해
    # 구제한다 — 어차피 그 다음 코드가 기대하는 모양은 리스트/딕셔너리이기 때문이다.
    if isinstance(names, str):
        names = json.loads(names)
    if isinstance(candidates, str):
        candidates = json.loads(candidates)

    raw_members, not_found = _lookup_members(names)
    boss = next((u for u in raw_members if u["name"] == "사장님"), None)
    members = [
        {
            "name": u["name"],
            "preferred_menus": u["preferred_menus"],
            "avoid_menus": u["constraints"]["avoid_menus"],
            "last_menu": u["constraints"]["last_menu"],
            "weight": BOSS_WEIGHT if u["name"] == "사장님" else 1,
        }
        for u in raw_members
    ]
    if boss is not None:
        max_round_trip_minutes = boss["constraints"]["max_round_trip_minutes"]
    else:
        caps = [
            u["constraints"]["max_round_trip_minutes"]
            for u in raw_members
            if u["constraints"]["max_round_trip_minutes"] is not None
        ]
        max_round_trip_minutes = min(caps) if caps else None
    if ignore_time_cap:
        max_round_trip_minutes = None

    ranked = []
    for c in candidates:
        if not c.get("standard_menus"):
            continue  # 안전하게 먹을 수 있는 메뉴가 하나도 없는 식당은 후보에서 제외
        minutes, multiplier = _compute_minutes(c["distance_km"], weather, mode)
        if max_round_trip_minutes is not None and minutes > max_round_trip_minutes:
            continue

        score = 0
        breakdown = []
        for m in members:
            weight = m.get("weight", 1)
            std_menus = c.get("standard_menus", [])
            if any(_matches(p, sm) for p in m.get("preferred_menus", []) for sm in std_menus):
                score += POINTS_PER_MATCH * weight
                breakdown.append(f"{m['name']} 선호 메뉴 매칭 +{POINTS_PER_MATCH * weight}")
            if any(_matches(a, sm) for a in m.get("avoid_menus", []) for sm in std_menus):
                score -= POINTS_PER_MATCH * weight
                breakdown.append(f"{m['name']} 회피 메뉴 존재 -{POINTS_PER_MATCH * weight}")
            last = m.get("last_menu")
            if last and any(_matches(last, sm) for sm in std_menus):
                score -= POINTS_PER_MATCH * weight
                breakdown.append(f"{m['name']} 어제 먹은 메뉴 존재 -{POINTS_PER_MATCH * weight}")

        if mode == "도보":
            one_way_minutes = minutes / 2
            penalty_units = int(one_way_minutes // 5)
            distance_penalty = penalty_units * DISTANCE_PENALTY_PER_5MIN
            if distance_penalty:
                score -= distance_penalty
                breakdown.append(f"편도 {one_way_minutes:.0f}분 이동 -{distance_penalty}")

        ranked.append({
            "name": c["name"],
            "distance_km": c["distance_km"],
            "round_trip_minutes": minutes,
            "standard_menus": c.get("standard_menus", []),
            "score": score,
            "score_breakdown": breakdown,
        })

    ranked.sort(key=lambda r: r["score"], reverse=True)
    return json.dumps(ranked[:top_n], ensure_ascii=False)


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
