# weather_mcp_server.py - 기상청 단기예보 API를 감싼 날씨 MCP 서버 (fastmcp)
"""노출하는 도구:
  - get_weather(region)   오늘 날씨 조회 (기본 지역: 경남 거제시 장평동)

실행: python weather_mcp_server.py
(stdio 모드로 대기합니다. agent.py가 자식 프로세스로 띄워서 사용합니다)

주의:
- AWS/Bedrock이 제공하는 기성 날씨 MCP 서버는 없어서, 기상청 공공데이터포털
  단기예보 조회서비스(getVilageFcst)를 직접 감싼 것이다.
- .env 의 KMA_SERVICE_KEY (공공데이터포털에서 발급받은 서비스 키)가 필요하다.
- DEFAULT_NX/DEFAULT_NY(장평동 격자 좌표)는 기상청이 배포하는 "기상청 격자 좌표" 엑셀
  기준으로 다시 확인이 필요한 임시값이다 (아래 TODO 참조). 확정 전까지는 이 값으로 조회한
  결과가 실제 장평동과 정확히 일치하지 않을 수 있다.
- evaluation 실행 시에는 이 서버를 실제로 호출하지 않고, 같은 인터페이스의 스텁으로
  교체해서 고정값을 반환한다 (CLAUDE.md 코드 규칙 참조).
"""
import os
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("weather")

KMA_SERVICE_KEY = os.environ.get("KMA_SERVICE_KEY", "")
BASE_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"

# TODO: 기상청 격자 좌표 엑셀(공공데이터포털 getVilageFcst 안내 페이지 첨부파일)에서
#       "경상남도 거제시 장평동"의 정확한 격자 X/Y로 다시 확인하고 교체할 것.
DEFAULT_REGION = "경남 거제시 장평동"
DEFAULT_NX = 90
DEFAULT_NY = 63

# 단기예보는 하루 8번(02,05,08,11,14,17,20,23시)만 발표된다.
FORECAST_ISSUE_HOURS = [2, 5, 8, 11, 14, 17, 20, 23]

# PTY(강수형태) 코드 -> 날씨 상태
PTY_MAP = {
    "0": None,      # 없음: SKY 코드로 판단
    "1": "비",
    "2": "비/눈",
    "3": "눈",
    "4": "소나기",
    "5": "빗방울",
    "6": "빗방울눈날림",
    "7": "눈날림",
}

# SKY(하늘상태) 코드 -> 날씨 상태 (PTY가 강수 없음일 때만 사용)
SKY_MAP = {
    "1": "맑음",
    "3": "구름많음",
    "4": "흐림",
}


def _latest_base_datetime(now: datetime) -> tuple[str, str]:
    """가장 최근에 발표된 단기예보 base_date, base_time을 계산한다."""
    candidates = [h for h in FORECAST_ISSUE_HOURS if h <= now.hour]
    if candidates:
        base_hour = max(candidates)
        base_date = now
    else:
        base_hour = FORECAST_ISSUE_HOURS[-1]
        base_date = now - timedelta(days=1)
    return base_date.strftime("%Y%m%d"), f"{base_hour:02d}00"


@mcp.tool()
def get_weather(region: str = DEFAULT_REGION) -> str:
    """오늘 날씨를 조회한다. 기본 지역은 경남 거제시 장평동이다.

    Args:
        region: 조회할 지역명 (현재는 기본 지역(장평동)의 격자 좌표만 정확히 매핑되어 있다)
    """
    if not KMA_SERVICE_KEY:
        return "날씨 조회 실패: KMA_SERVICE_KEY 가 설정되지 않았습니다 (.env 확인 필요)."
    if region != DEFAULT_REGION:
        return f"'{region}'은(는) 아직 격자 좌표가 매핑되지 않았습니다. 현재는 '{DEFAULT_REGION}'만 지원합니다."

    now = datetime.now()
    base_date, base_time = _latest_base_datetime(now)

    params = {
        "serviceKey": KMA_SERVICE_KEY,
        "dataType": "JSON",
        "numOfRows": 100,
        "pageNo": 1,
        "base_date": base_date,
        "base_time": base_time,
        "nx": DEFAULT_NX,
        "ny": DEFAULT_NY,
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=5)
        resp.raise_for_status()
        items = resp.json()["response"]["body"]["items"]["item"]
    except Exception as e:
        return f"날씨 조회 실패: {e}"

    today_str = now.strftime("%Y%m%d")
    today_items = [it for it in items if it.get("fcstDate") == today_str]

    pty = next((it["fcstValue"] for it in today_items if it["category"] == "PTY"), "0")
    sky = next((it["fcstValue"] for it in today_items if it["category"] == "SKY"), "1")

    condition = PTY_MAP.get(pty) or SKY_MAP.get(sky, "알 수 없음")
    return condition


if __name__ == "__main__":
    mcp.run()   # stdio 모드로 실행
