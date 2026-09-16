# 미니 PJT: 점심 식당 추천 Agent

## 무엇을 푸나
경남 거제시 장평동 사무실 기준, 일행의 선호·건강 특이사항·이동 가능 범위와 오늘 날씨를 반영해 점심 식당·메뉴를 추천한다.

## 활용한 패턴 (Day 1~7)
- Day 2: RAG(Chroma + BedrockEmbeddings) — 식당 메뉴명을 표준 메뉴명으로 정규화하고, 사용자 특이사항으로 관련 메뉴 주의사항을 검색하는 데 응용 (`src/retriever.py`)
- Day 3: ReAct 기본 에이전트/도구 패턴 — `create_agent` + 도구 목록 조합의 뼈대로 사용
- Day 4~5: MCP 서버(fastmcp) — 구조화 데이터(식당/메뉴/사용자) 조회·수정을 별도 서버로 분리 (`src/mcp_server.py`), 날씨 조회도 같은 방식으로 별도 서버화 (`src/weather_mcp_server.py`)
- Day 5: 미들웨어 체인(로깅/마스킹/입출력 가드레일) + HITL(`HumanInTheLoopMiddleware`) — 5일차 `guarded_agent` 계열을 시작점으로 복사해 도메인에 맞게 재배선
- Day 6: 여러 MCP 서버를 `MultiServerMCPClient`로 동시에 연결하고, RAG와 MCP 도구를 하나의 에이전트에 함께 붙이는 통합 패턴

## 아키텍처
```
사용자 질의
  └─ InputGuardrailMiddleware (프롬프트 인젝션 1차 차단)
  └─ MaskingMiddleware (입력 민감정보 마스킹)
  └─ ToolCallLimitMiddleware (도구 호출 상한)
  └─ CompanionAuthorizationMiddleware (본인/명시된 동행자 외 조회 차단)
  └─ ReAct 루프 (create_agent)
       ├─ mcp_server.py       : get_user_preference / update_user_preference / list_restaurants / get_menu
       ├─ weather_mcp_server.py: get_weather (기상청 단기예보 API, 기본 지역: 장평동)
       ├─ tools.py            : estimate_round_trip_minutes (날씨 배율 적용)
       └─ retriever.py        : normalize_menu_names / search_menu_by_caution (RAG)
  └─ HumanInTheLoopMiddleware (update_user_preference 승인 게이트)
  └─ OutputGuardrailMiddleware (근거 없는 단정 등 출력 검증)
  └─ LoggingMiddleware (정제된 내용만 기록)
```

## 실행 방법
```bash
pip install -r requirements.txt
# 루트(sds-ax-practice/)의 .env 에 AWS 자격증명 + KMA_SERVICE_KEY 필요
cd src
python agent.py
```
`agent.py`는 현재 3가지 시나리오(정상 추천 → 선호도 수정 HITL 승인 → 프롬프트 인젝션 차단)를 실행하는 데모 스크립트다. `POST /query` HTTP API 서버는 아직 구현 전이다.

## RAGAS 평가 결과
아직 실행 전이다. 구조화 데이터(식당/메뉴) 추천 로직은 RAGAS 적용 대상이 아니라고 판단했고(검색 대상 문서가 없음), `retriever.py`의 특이사항 검색 부분만 RAGAS로 평가할 계획이다.
- context_recall: 미실행
- context_precision: 미실행
- faithfulness: 미실행
- answer_relevancy: 미실행

## 인-아웃 세트 통과율 (자체 평가)
`evaluation/test_queries.csv`에 17개 케이스(positive/negative/edge/guardrail)를 작성했다. 개별 시나리오(추천 흐름, HITL 미발동 케이스, 프롬프트 인젝션)는 `agent.py` 데모로 개별 확인했지만, 17건 전체를 채점 스크립트로 자동 실행한 적은 아직 없다.
- 1차 (Day 9 종료): 미실시 — `evaluation/round1_report.md` 작성 예정
- 2차 (Day 10 개선 후): 미실시
- 개선폭: -

## 트라이앤에러 회고
- **시도했지만 실패한 접근**: `AsyncSqliteSaver`를 `build_app()` 안에서 `async with ... as checkpointer: return agent`로 감쌌더니, `return` 시점에 컨텍스트 매니저가 닫혀 연결이 끊겼다. 이후 `main()`에서 `agent.ainvoke()`를 호출하면 "threads can only be started once" / "Connection closed" 에러가 났다. `build_app()`이 만든 에이전트를 다른 함수에서 계속 써야 하는 구조라면, 컨텍스트 매니저를 함수 스코프 안에서 닫으면 안 된다는 걸 확인했다.
- **최종 채택한 접근**: `AsyncSqliteSaver.from_conn_string(...)`이 돌려주는 컨텍스트 매니저 객체를 모듈 전역 변수에 보관하고 `__aenter__()`만 호출해, 앱 생명주기 동안 연결을 열어둔 채로 쓴다 (`src/agent.py`).
- **또 다른 시행착오**: Bedrock Titan 임베딩으로 Chroma를 만들 때 기본(L2) 거리 기준 relevance score가 음수로 깨져서, 진짜 정답도 전부 "매칭 실패"로 처리되고 있었다. `collection_metadata={"hnsw:space": "cosine"}`으로 코사인 유사도 기준으로 바꿔서 해결했다.
- **또 다른 시행착오**: 식당 메뉴 하나하나마다 `normalize_menu_name`을 개별 호출하다 보니 `ToolCallLimitMiddleware` 상한(15)에 금방 도달해 추천이 중간에 끊겼다. 여러 메뉴명을 한 번에 처리하는 `normalize_menu_names` 배치 도구로 바꿔서 해결했다.
- **남은 한계**: (1) 짧은 한국어 메뉴명끼리는 임베딩 유사도가 낮게 나와(진짜 정답도 0.3대) 임계값 하나로 완벽히 분리하기 어렵다. (2) `weather_mcp_server.py`의 격자 좌표(nx/ny)는 기상청 공식 좌표표로 재확인이 필요한 임시값이다. (3) RAGAS 평가와 test_queries.csv 17건 자동 채점이 아직 없다.
- **향후 개선 방향**: 격자 좌표 확정, `evaluation/` 자동 채점 스크립트 작성 후 1차 자체평가 실시, RAG 파트에 RAGAS 도입.

## 핵심 코드 위치
- `src/agent.py:121` — `build_app()`, 메인 에이전트 조립 (미들웨어·MCP·도구 배선)
- `src/agent.py:76` — `CompanionAuthorizationMiddleware`, 인가 가드레일
- `src/mcp_server.py` — 식당·메뉴·사용자 조회/수정 MCP 서버
- `src/weather_mcp_server.py` — 기상청 단기예보 API 기반 날씨 MCP 서버
- `src/tools.py` — `estimate_round_trip_minutes`, 날씨 배율 적용 왕복 이동시간 계산
- `src/retriever.py:92` — `normalize_menu_names`, 메뉴명 정규화 (RAG)
- `src/retriever.py:109` — `search_menu_by_caution`, 특이사항 기반 메뉴 검색 (RAG)
