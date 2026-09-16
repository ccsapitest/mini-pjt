# tools.py - 점심 식당 추천 Agent의 도메인 계산 도구
"""날씨 조회 자체는 외부 MCP 서버에서 가져온다 (agent.py에서 별도 연동).
이 파일은 그 날씨 값을 입력받아 왕복 이동시간을 계산하는 순수 계산 도구만 담는다.
"""
from langchain_core.tools import tool

# 이동수단별 평균 속도(km/h). 실제 조사값이 아닌 실습용 가정치이며 임의로 바꾸지 않는다.
SPEED_KMH = {"도보": 4, "대중교통": 20, "차량": 30}

# 날씨 배율표: 맑음 외 모든 날씨는 왕복 이동시간에 동일하게 1.2배를 적용한다 (이동수단 구분 없음).
CLEAR_WEATHER = "맑음"
BAD_WEATHER_MULTIPLIER = 1.2


@tool
def estimate_round_trip_minutes(distance_km: float, weather: str, mode: str = "도보") -> str:
    """식당까지의 왕복 이동시간(분)을 계산한다.

    맑음이 아닌 모든 날씨(비, 눈, 흐림 등)에는 이동수단과 무관하게 1.2배를 적용한다.

    Args:
        distance_km: 사무실에서 식당까지의 편도 거리(km)
        weather: 오늘 날씨 (예: "맑음", "비", "눈", "흐림")
        mode: 이동수단 ("도보", "대중교통", "차량" 중 하나, 기본값 "도보")
    """
    speed_kmh = SPEED_KMH.get(mode, SPEED_KMH["도보"])
    base_minutes = (distance_km / speed_kmh) * 60 * 2
    multiplier = 1.0 if weather == CLEAR_WEATHER else BAD_WEATHER_MULTIPLIER
    minutes = round(base_minutes * multiplier)
    return f"{minutes}분 (편도 {distance_km}km, {mode}, 날씨: {weather}, 배율 x{multiplier})"
