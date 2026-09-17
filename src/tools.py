# tools.py - 점심 식당 추천 Agent의 도메인 계산 도구
"""날씨 조회 자체는 외부 MCP 서버에서 가져온다 (agent.py에서 별도 연동).
이 파일은 그 날씨 값을 입력받아 왕복 이동시간을 계산하는 순수 계산 도구만 담는다.
"""
import json

from langchain_core.tools import tool

# 이동수단별 평균 속도(km/h). 실제 조사값이 아닌 실습용 가정치이며 임의로 바꾸지 않는다.
SPEED_KMH = {"도보": 4, "대중교통": 20, "차량": 30}

# 날씨 배율표: 맑음 외 모든 날씨는 왕복 이동시간에 동일하게 1.2배를 적용한다 (이동수단 구분 없음).
CLEAR_WEATHER = "맑음"
BAD_WEATHER_MULTIPLIER = 1.2


def _compute_minutes(distance_km: float, weather: str, mode: str) -> tuple[int, float]:
    speed_kmh = SPEED_KMH.get(mode, SPEED_KMH["도보"])
    base_minutes = (distance_km / speed_kmh) * 60 * 2
    multiplier = 1.0 if weather == CLEAR_WEATHER else BAD_WEATHER_MULTIPLIER
    return round(base_minutes * multiplier), multiplier


@tool
def estimate_round_trip_minutes(distances_km: list[float], weather: str, mode: str = "도보") -> str:
    """여러 식당까지의 왕복 이동시간(분)을 한 번에 계산한다 (거리 목록 전체를 한 번만 호출).

    후보 식당이 여러 곳이어도 이 도구를 식당 개수만큼 따로 부르지 않는다 — list_restaurants가
    돌려준 distance_km 값들을 한 번에 모아 이 도구 하나만 호출한다 (도구 호출 횟수 제한에
    빨리 도달하는 것을 방지). 결과 리스트의 순서는 입력한 distances_km 순서와 동일하다.

    맑음이 아닌 모든 날씨(비, 눈, 흐림 등)에는 이동수단과 무관하게 1.2배를 적용한다.

    Args:
        distances_km: 사무실에서 각 식당까지의 편도 거리(km) 목록
        weather: 오늘 날씨 (예: "맑음", "비", "눈", "흐림") — 모든 식당에 동일하게 적용된다
        mode: 이동수단 ("도보", "대중교통", "차량" 중 하나, 기본값 "도보")
    """
    results = []
    for distance_km in distances_km:
        minutes, multiplier = _compute_minutes(distance_km, weather, mode)
        results.append({
            "distance_km": distance_km,
            "round_trip_minutes": minutes,
            "mode": mode,
            "weather": weather,
            "multiplier": multiplier,
        })
    return json.dumps(results, ensure_ascii=False)
