# 미니 PJT: 점심 식당 추천 Agent

## 무엇을 푸나
경남 거제시 장평동 사무실 기준, 일행의 선호·건강 특이사항·이동 가능 범위와 오늘 날씨를 반영해 점심 식당·메뉴를 추천한다.

## 활용한 패턴 (Day 1~7)
- Day 2: RAG(Chroma + BedrockEmbeddings) — 식당 메뉴명을 표준 메뉴명으로 정규화하고, 사용자 특이사항으로 관련 메뉴 주의사항을 검색하는 데 응용 (`src/retriever.py`)
- Day 3: ReAct 기본 에이전트/도구 패턴 — `create_agent` + 도구 목록 조합의 뼈대로 사용
- Day 4~5: MCP 서버(fastmcp) — 구조화 데이터(식당/메뉴/사용자) 조회·수정을 별도 서버로 분리 (`src/mcp_server.py`), 날씨 조회도 같은 방식으로 별도 서버화 (`src/weather_mcp_server.py`)
- Day 5: 미들웨어 체인(로깅/마스킹/입출력 가드레일) + HITL(`interrupt()` 기반 `HumanInTheLoopMiddleware`, `update_user_preference` 승인 게이트) — 5일차 `guarded_agent` 계열을 시작점으로 복사해 도메인에 맞게 재배선
- Day 6: 여러 MCP 서버를 `MultiServerMCPClient`로 동시에 연결하고, RAG와 MCP 도구를 하나의 에이전트에 함께 붙이는 통합 패턴
- 도구 호출 배치화 — 식당/메뉴 개수만큼 도구를 따로 부르던 것을 리스트 인자 하나로 묶어 한 번에 처리(`normalize_menu_names`, `estimate_round_trip_minutes`), 턴당 도구 호출 수와 토큰 사용량을 크게 절감
- Day 7: Observability(LangSmith 트레이스 — 코드 수정 없이 `.env`의 `LANGSMITH_TRACING`/`LANGSMITH_API_KEY`/`LANGSMITH_PROJECT`만으로 전체 도구 호출·미들웨어 훅·모델 호출이 자동 기록됨) + LLM-as-Judge(`evaluation/judge_eval.py` — `run_eval.py` 결과를 사람이 아니라 LLM이 `expected_traits`/`forbidden` 기준으로 재현 가능하게 자동 채점)

## 아키텍처
```
사용자 질의
  └─ InputGuardrailMiddleware (프롬프트 인젝션 1차 차단 — 규칙 + LLM 판별)
  └─ MaskingMiddleware (입력 민감정보 마스킹)
  └─ ToolCallLimitMiddleware (run_limit=25, exit_behavior="end" — 턴 단위로 리셋되는 도구 호출 상한)
  └─ CompanionAuthorizationMiddleware (요청자 본인 + 대화 전체에서 한 번이라도 언급된 동행자 외 조회 차단)
  └─ ReAct 루프 (create_agent)
       ├─ mcp_server.py       : get_group_constraints(일행 병합·사장님 오버라이드 포함) / get_user_preference /
       │                        update_user_preference(세션 동안만 임시 반영, 실제 DB 미기록) / list_restaurants / get_menu
       ├─ weather_mcp_server.py: get_weather (기상청 단기예보 API, 기본 지역: 장평동)
       ├─ tools.py            : estimate_round_trip_minutes (날씨 배율 적용, 여러 식당 거리 일괄 계산)
       └─ retriever.py        : normalize_menu_names / search_menu_by_caution (RAG)
  └─ HumanInTheLoopMiddleware (update_user_preference 승인 게이트)
  └─ OutputGuardrailMiddleware (민감정보 마스킹, 범위 이탈 차단, 근거 없는 단정 검증)
  └─ LoggingMiddleware (정제된 내용만 기록)
```

`update_user_preference`는 사람이 승인하더라도 실제 사용자 DB(`data/users.json`) 파일을 직접 수정하지 않는다. 지금 떠 있는 MCP 서버 프로세스의 메모리에만 반영되어 이번 세션 동안의 추천에 임시로 적용되고, 세션(프로세스)이 끝나면 원래 값으로 돌아간다 — 실제 DB 반영은 이 프로젝트 범위 밖의 별도 DB 관리 솔루션의 몫으로 가정한다.

## 실행 방법
```bash
pip install -r requirements.txt
# 루트(sds-ax-practice/)의 .env 에 AWS 자격증명 + KMA_SERVICE_KEY 필요
cd src
python agent.py       # 3가지 시나리오(정상 추천 → 선호도 수정 HITL 승인 → 프롬프트 인젝션 차단) 데모
```
자유롭게 이것저것 물어보고 싶으면 루트에서 `python query_gui.py`로 tkinter GUI를 띄운다 (HITL 승인 대기 중에는 입력창에 "승인"/"거절" 입력). `POST /query` HTTP API 서버는 아직 구현 전이다.

## 평가 결과

### 인-아웃 세트 통과율 (자체 평가)
`evaluation/test_queries.csv`에 18개 케이스(positive/negative/edge/guardrail)를 작성하고, `evaluation/run_eval.py`로 실제 에이전트에 돌려 결과를 수집한다.

| 라운드 | 모델/채점 방식 | 결과 | 핵심 내용 |
|---|---|---|---|
| 1차 (`round1_report.md`) | Haiku 4.5, 사람 채점 | 18/18 PASS | 체크포인트를 초기화하지 않고 반복 평가하면 이전 실행의 대화 상태가 남아 회귀처럼 보이는 문제(가짜 회귀)를 발견·수정. 이후 재실행에서 18/18 |
| 2차 (`round2_report.md`) | Haiku 4.5, 사람 채점 | 18/18 PASS | 사장님 오버라이드 신뢰성 문제를 프롬프트가 아니라 `get_group_constraints` 도구(서버 계산 + 원본 회피 데이터 리댁션)로 옮겨 해결. 도구 배치화로 토큰 사용량 절감. Amazon Nova 2 Lite로 모델 교체 실험은 다단계 도구 체인에서 빈 답변을 내는 문제로 부적합 판정 |
| 3차 | Sonnet 4.5 전체 전환, `judge_eval.py` 자동 채점(LLM-as-Judge) | 18/18 PASS | 모든 LLM 호출(에이전트 본체 + 가드레일 + 로깅)을 `global.anthropic.claude-sonnet-4-5-20250929-v1:0`으로 통일. 최초 채점 13/18 → judge 프롬프트에 요청자 본인 식별·HITL 상세정보·출처 요구 완화를 반영해 재채점하니 3건이 judge 자체 오판이었음이 확인되어 16/18. 남은 진짜 이슈 2건 중 1건(메뉴 배치 결과 오독)은 재현 안 됨, 1건(미등록 동행자 "동훈" 안내 누락)은 3/3 재현되어 프롬프트 수정 후 18/18 달성 |

같은 세션에서 추가로 발견·수정한 것(별도 보고서 미작성, 커밋 이력에 기록):
- `CompanionAuthorizationMiddleware`가 거절 시 문자열을 반환해 `HumanMessage`로 잘못 강제 변환되며 tool_use/tool_result 짝이 깨지던 Bedrock `ValidationException`
- `ToolCallLimitMiddleware(thread_limit=...)`가 대화 스레드 전체에서 누적되어 긴 대화에서 결국 한도를 넘기던 문제 → `run_limit`(턴 단위 리셋)으로 변경
- 인가 판정이 "이번 메시지"만 보고 있어, 이전 턴에 이미 등장한 동행자를 이어가는 정상 요청("사장님 대신 배트맨 불러서")까지 오차단하던 문제
- "사장님"을 존칭/일반명사로 오인해 "정확한 이름이 뭔가요?"라고 되묻던 문제
- "OO 조건 무시하고 추천해줘" 같은 도메인 제약 완화 요청을 프롬프트 인젝션으로 오탐하던 문제
- 동행자가 미등록일 때(`not_found`) 사용자에게 알리지 않고 조용히 나머지 인원만으로 추천을 진행하던 문제 (Sonnet에서 재현, 프롬프트에 명시적 규칙 추가로 해결)

### Observability (LangSmith)
`.env`에 `LANGSMITH_TRACING=true`/`LANGSMITH_API_KEY`/`LANGSMITH_PROJECT`만 설정하면 코드 수정 없이 전체 트레이스가 기록된다. 도구 호출, 미들웨어 훅(`MaskingMiddleware.before_model` 등), 모델 호출, 실패한 호출(Bedrock 스로틀링 등)까지 전부 LangSmith 대시보드에서 확인 가능함을 실제 호출로 검증했다.

### LLM-as-Judge (`evaluation/judge_eval.py`)
지금까지(1·2차)는 `expected_traits`/`forbidden` 대조를 사람이 대화 중에 직접 했는데, 이를 고정된 판정 프롬프트로 재현 가능하게 자동화했다. 회고: 처음 짠 judge 프롬프트는 이 시스템의 도메인 맥락(요청자 본인 이름, HITL 승인 대기 시 실제 변경 내용이 `interrupt_value`에 있다는 것, 외부 출처 인용이 없는 내부 데이터 구조)을 몰라서 3건을 잘못 FAIL 처리했다 — 채점기 프롬프트에도 도메인 배경지식을 넣어줘야 한다는 교훈.

### RAG 평가 (RAGAS)
도입을 시도했으나 보류했다. `ragas`가 의존하는 `scikit-network`가 현재 환경의 Python 3.14용 사전빌드 wheel을 제공하지 않아 소스 빌드가 필요한데, Microsoft Visual C++ Build Tools가 없어 실패했다. `--no-deps`로 우회해 나머지 의존성을 수동 설치해도, `ragas`가 import 시점에 요구하는 `langchain_community.chat_models.vertexai` 서브모듈이 최신 `langchain_community`에서 이미 제거되어 있어 또 막혔다 — `ragas 0.4.3`이 기대하는 `langchain_community` 버전이 이 프로젝트가 쓰는 `langchain 1.4.0`과 맞지 않는다. Python 3.14가 아직 이 생태계와 충분히 맞물리지 않은 것이 근본 원인으로 보인다. 대신 커스텀 LLM-as-Judge(`judge_eval.py`)로 평가 자동화 자체는 달성했다.
- context_recall / context_precision / faithfulness / answer_relevancy: 미실행

### RAG 평가 (RAGAS)
도입을 시도했으나 보류했다. `ragas`가 의존하는 `scikit-network`가 현재 환경의 Python 3.14용 사전빌드 wheel을 제공하지 않아 소스 빌드가 필요한데, Microsoft Visual C++ Build Tools가 없어 실패했다. `--no-deps`로 우회해 나머지 의존성을 수동 설치해도, `ragas`가 import 시점에 요구하는 `langchain_community.chat_models.vertexai` 서브모듈이 최신 `langchain_community`에서 이미 제거되어 있어 또 막혔다 — `ragas 0.4.3`이 기대하는 `langchain_community` 버전이 이 프로젝트가 쓰는 `langchain 1.4.0`과 맞지 않는다. Python 3.14가 아직 이 생태계와 충분히 맞물리지 않은 것이 근본 원인으로 보이며, 환경이 안정화되면 재시도하거나, RAGAS 대신 `retriever.py` 전용 커스텀 채점 스크립트(정규화 정확도 등)로 대체하는 방안을 검토 중이다.
- context_recall / context_precision / faithfulness / answer_relevancy: 미실행

## 트라이앤에러 회고
- **컨텍스트 매니저 조기 종료**: `AsyncSqliteSaver`를 `build_app()` 안에서 `async with ... as checkpointer: return agent`로 감쌌더니, `return` 시점에 컨텍스트 매니저가 닫혀 연결이 끊겼다. `build_app()`이 만든 에이전트를 다른 함수에서 계속 써야 하는 구조라면, 컨텍스트 매니저를 함수 스코프 안에서 닫으면 안 된다 — 전역 변수에 컨텍스트 매니저 객체를 보관하고 `__aenter__()`만 호출해 앱 생명주기 동안 열어두는 방식으로 해결했다.
- **Chroma relevance score가 음수로 깨짐**: Bedrock Titan 임베딩으로 Chroma를 만들 때 기본(L2) 거리 기준 relevance score가 음수로 나와, 진짜 정답도 전부 "매칭 실패"로 처리됐다. `collection_metadata={"hnsw:space": "cosine"}`으로 해결했다.
- **도구 호출 배치화**: 식당 메뉴 하나하나마다, 식당 하나하나마다 도구를 개별 호출하다 보니 `ToolCallLimitMiddleware` 상한에 금방 도달하고 토큰도 많이 썼다. 리스트 인자를 받는 배치 도구(`normalize_menu_names`, `estimate_round_trip_minutes`)로 바꿔 호출 수를 8회→1회 수준으로 줄였다.
- **`wrap_tool_call`은 반드시 `ToolMessage`를 반환해야 한다**: 인가 미들웨어가 거절 시 그냥 문자열을 반환했는데, 이게 어떤 경로에서는 `ToolMessage`가 아니라 `HumanMessage`로 잘못 강제 변환되면서 해당 tool_use에 대응하는 tool_result가 사라져 Bedrock `ValidationException`("tool_use ids were found without tool_result blocks")이 발생했다. 라이브러리 소스(`langchain/agents/middleware/types.py`)를 직접 확인해 `wrap_tool_call`/`awrap_tool_call`의 반환 타입 계약이 `ToolMessage | Command`임을 확인하고, `tool_call_id`가 일치하는 `ToolMessage`를 명시적으로 만들어 반환하도록 고쳤다.
- **`thread_limit`은 대화 전체에서 계속 누적된다**: `ToolCallLimitMiddleware(thread_limit=25)`는 체크포인트로 유지되는 대화 스레드 생애주기 전체에서 누적되는 카운터라, GUI처럼 같은 thread_id로 오래 대화할수록 결국 정상적인 턴에서도 한도를 넘긴다. 턴(`.ainvoke()` 1회)마다 리셋되는 `run_limit`으로 바꿔야 원래 의도(한 턴 안에서 도구 무한 호출 방지)에 맞는다.
- **LLM이 명시적 지시를 계속 어기면, 지시를 더 세게 쓰기보다 판단 근거가 되는 데이터 자체를 안 보이게 한다**: "사장님이 있으면 다른 사람 조건은 무시한다"는 규칙을 프롬프트로 두 차례 강화해도, 모델이 동행자 개인의 원본 회피 정보를 다시 읽고 "충돌"이라고 재해석하는 문제가 반복됐다. 병합 로직을 `get_group_constraints` 도구로 옮기고, 사장님이 있을 때는 응답에서 다른 사람의 회피 정보 자체를 지워버리자 문제가 사라졌다.
- **LLM 채점기(judge)도 도메인 배경지식이 없으면 오판한다**: `judge_eval.py`의 첫 채점에서 5건 중 3건이 판정 실수였다 — "본인" 이름이 시스템에 고정값으로 등록되어 있다는 것, HITL 승인 대기 중엔 실제 변경 내용이 채팅 텍스트가 아니라 `interrupt_value`에 있다는 것, 이 서비스엔 애초에 "출처 인용" 개념이 없다는 것을 judge 프롬프트가 몰랐다. 사람 채점자에게 당연한 맥락도 자동 채점 프롬프트엔 명시적으로 넣어줘야 한다.
- **남은 한계**:
  1. 짧은 한국어 메뉴명끼리는 임베딩 유사도가 낮게 나와(진짜 정답도 0.3대) 임계값 하나로 완벽히 분리하기 어렵다 — 정확 문자열 매칭을 우선 시도하고 임베딩은 폴백으로만 쓰는 방식으로 보완했다.
  2. `weather_mcp_server.py`의 격자 좌표(nx/ny)는 기상청 공식 좌표표로 재확인이 필요한 임시값이다.
  3. 기상청 API가 간헐적으로 실패한다(외부 API 신뢰성 문제, 실패 시 폴백 응답 자체는 정상 동작).
  4. RAGAS 도입은 Python 3.14 환경 문제로 보류 중이다.
- **향후 개선 방향**: 격자 좌표 확정, RAG 파트 평가(RAGAS 또는 커스텀 스크립트) 착수, 토큰 비용 추가 최적화.

## 핵심 코드 위치
- `src/agent.py:243` — `build_app()`, 메인 에이전트 조립 (미들웨어·MCP·도구 배선)
- `src/agent.py:172` — `CompanionAuthorizationMiddleware`, 인가 가드레일
- `src/mcp_server.py:88` — `get_group_constraints`, 일행 병합(최솟값/합집합/사장님 오버라이드) 서버 계산
- `src/mcp_server.py:165` — `update_user_preference`, 세션 한정(비영구) 선호·제약 수정
- `src/mcp_server.py:44` — `_validate_field_value`, 존재하지 않는 개념을 다른 필드에 억지로 끼워맞추는 것 방지
- `src/weather_mcp_server.py` — 기상청 단기예보 API 기반 날씨 MCP 서버
- `src/tools.py:25` — `estimate_round_trip_minutes`, 날씨 배율 적용 왕복 이동시간 일괄 계산
- `src/retriever.py:147` — `normalize_menu_names`, 메뉴명 정규화 (RAG)
- `src/retriever.py:164` — `search_menu_by_caution`, 특이사항 기반 메뉴 검색 (RAG)
- `query_gui.py` — 자유 질의 GUI (tkinter)
- `evaluation/judge_eval.py` — LLM-as-Judge 자동 채점 스크립트
