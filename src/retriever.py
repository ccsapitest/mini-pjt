# retriever.py - 메뉴명 정규화 검색 + 특이사항 기반 메뉴 필터링 (RAG 파이프라인)
"""표준 메뉴(menus.json)를 임베딩해 두 가지 용도로 검색한다.

  1. normalize_menu_names: 식당이 부르는 메뉴명들(예: "왕 등심 돈까스")을
     표준 메뉴명(예: "돈까스")으로 정규화한다. 한 번에 여러 개를 배치로 처리해서,
     식당 하나당 도구 호출을 1회로 줄인다 (메뉴마다 따로 부르면 도구 호출 제한에
     금방 도달한다). 유사도가 낮으면 매칭하지 않고 보수적으로 "정보 없음"을 반환한다
     (CLAUDE.md 코드 규칙 참조).

  2. search_menu_by_caution: 사용자의 특이사항(자유 텍스트, 예: "고나트륨 주의")을
     질의로, 관련된 표준 메뉴의 주의사항을 검색해 후보 제외에 쓸 메뉴 목록을 돌려준다.
"""
import json
import os
import threading

from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool

BASE = os.path.dirname(os.path.abspath(__file__))
MENUS_PATH = os.path.join(BASE, "..", "data", "menus.json")
PERSIST_DIR = os.path.join(BASE, "..", "chroma_db")

# Bedrock Titan 임베딩 기준 실측 점수대로 보정한 임계값이다 (짧은 한국어 메뉴명끼리는
# 코사인 유사도가 0.3~0.4대만 되어도 진짜 정답인 경우가 많다). 정규화(이름 매칭)는
# 다른 메뉴로 오매칭되면 건강 주의사항이 누락될 수 있어 더 엄격하게 잡는다.
NAME_MATCH_THRESHOLD = 0.30
CAUTION_MATCH_THRESHOLD = 0.5

def _load_keyword_map() -> dict:
    """표준 메뉴명·별칭 -> 메뉴 dict. 정확한 문자열 포함 매칭에 쓴다 (임베딩 호출 불필요)."""
    with open(MENUS_PATH, encoding="utf-8") as f:
        menus = json.load(f)
    keyword_map = {}
    for menu in menus:
        keyword_map[menu["name"]] = menu
        for alias in menu.get("aliases", []):
            keyword_map[alias] = menu
    return keyword_map


_KEYWORD_MAP = _load_keyword_map()


def _exact_match(query: str) -> dict | None:
    """표준 메뉴명이나 별칭이 질의에 그대로 포함되어 있으면 그 메뉴를 확정 반환한다.

    예: "에비덴 왕새우 커리"에 별칭 "커리"가 그대로 들어있으므로 "카레"로 확정 매칭한다.
    공백은 무시하고 비교한다 (예: "우거지 뼈 해장국"과 표준명 "뼈해장국"의 띄어쓰기 차이).
    임베딩 유사도보다 정확한 문자열 포함이 항상 더 신뢰할 수 있어 먼저 확인한다.
    """
    normalized_query = query.replace(" ", "")
    # 긴 키워드부터 확인해 "치즈돈까스"가 "돈까스"보다 먼저 매칭되게 한다.
    for keyword in sorted(_KEYWORD_MAP, key=len, reverse=True):
        if keyword.replace(" ", "") in normalized_query:
            return _KEYWORD_MAP[keyword]
    return None


_embeddings = None
_vectorstore = None
# 에이전트가 한 턴에 여러 도구를 병렬(스레드풀)로 호출할 때, _vectorstore가 아직 비어있으면
# 두 스레드가 동시에 같은 경로로 Chroma 클라이언트를 만들려다 충돌한다
# ("AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'"). 락으로 막는다.
_vectorstore_lock = threading.Lock()


def _get_vectorstore() -> Chroma:
    """menus.json 을 임베딩한 Chroma 벡터스토어를 만들거나 캐시된 것을 돌려준다."""
    global _embeddings, _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    with _vectorstore_lock:
        if _vectorstore is not None:  # 락 대기 중 다른 스레드가 이미 만들었을 수 있다
            return _vectorstore
        return _build_vectorstore()


def _build_vectorstore() -> Chroma:
    global _embeddings, _vectorstore
    _embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1",
    )

    with open(MENUS_PATH, encoding="utf-8") as f:
        menus = json.load(f)

    documents = []
    for menu in menus:
        aliases = menu.get("aliases", [])
        # "커리"처럼 식당에서 자주 쓰는 동의어를 문서 텍스트에 포함시켜야 유사도 검색이
        # 그 표현으로 물어봐도 정답을 찾는다 (예: "카레" <-> "커리").
        display_name = f"{menu['name']}({', '.join(aliases)})" if aliases else menu["name"]
        documents.append(Document(
            page_content=f"{display_name}: {menu['caution']}",
            # Chroma 메타데이터는 리스트를 못 받으므로 aliases는 콤마로 이어 붙인 문자열로 저장한다.
            metadata={**menu, "aliases": ", ".join(aliases)},
        ))
    ids = [menu["name"] for menu in menus]

    store = Chroma(
        collection_name="standard_menus",
        embedding_function=_embeddings,
        persist_directory=PERSIST_DIR,
        # Bedrock Titan 임베딩은 기본(L2) 거리 기준 relevance score가 음수로 깨진다.
        # 코사인 유사도 기준으로 바꿔야 0~1 범위의 정상적인 유사도 점수가 나온다.
        collection_metadata={"hnsw:space": "cosine"},
    )
    # 메뉴명을 id로 써서, 프로세스를 다시 띄워도 같은 메뉴가 중복 적재되지 않게 한다.
    if store._collection.count() != len(menus):
        existing_ids = store.get()["ids"]
        if existing_ids:
            store.delete(ids=existing_ids)
        store.add_documents(documents, ids=ids)

    _vectorstore = store
    return _vectorstore


def _normalize_one(restaurant_menu_name: str) -> dict:
    exact = _exact_match(restaurant_menu_name)
    if exact is not None:
        return {
            "restaurant_menu_name": restaurant_menu_name,
            "standard_menu": exact,
            "match_score": 1.0,
        }

    results = _get_vectorstore().similarity_search_with_relevance_scores(
        restaurant_menu_name, k=1
    )
    if not results or results[0][1] < NAME_MATCH_THRESHOLD:
        return {"restaurant_menu_name": restaurant_menu_name, "standard_menu": None}
    doc, score = results[0]
    return {
        "restaurant_menu_name": restaurant_menu_name,
        "standard_menu": doc.metadata,
        "match_score": round(score, 3),
    }


@tool
def normalize_menu_names(restaurant_menu_names: list[str]) -> str:
    """식당이 부르는 메뉴명들을 표준 메뉴명으로 정규화한다 (여러 개를 한 번에 처리).

    예: ["왕 등심 돈까스", "쫄우동"] -> [{"...": "돈까스"}, {"...": "우동"}].
    식당 하나의 메뉴 목록은 이 도구를 한 번만 호출해서 전부 정규화한다
    (메뉴마다 따로 호출하지 않는다 — 도구 호출 횟수 제한에 빨리 도달한다).
    유사도가 낮아 매칭이 애매하면 별칭을 지어내지 않고 standard_menu를 null로 반환해
    후보에서 제외되도록 한다.

    Args:
        restaurant_menu_names: 식당이 실제로 쓰는 메뉴명 목록
    """
    results = [_normalize_one(name) for name in restaurant_menu_names]
    return json.dumps(results, ensure_ascii=False)


@tool
def search_menu_by_caution(health_note: str) -> str:
    """사용자의 특이사항으로 관련된 표준 메뉴의 주의사항을 검색한다.

    예: "고나트륨 주의" -> 나트륨이 높은 표준 메뉴 목록. 이 결과를 근거로
    추천 후보에서 제외할 메뉴를 정한다.

    Args:
        health_note: 사용자의 특이사항 자유 텍스트 (예: "당뇨", "날것 주의")
    """
    results = _get_vectorstore().similarity_search_with_relevance_scores(
        health_note, k=5
    )
    matched = [doc.metadata for doc, score in results if score >= CAUTION_MATCH_THRESHOLD]
    if not matched:
        return f"'{health_note}'와 관련해 주의가 필요한 표준 메뉴를 찾지 못했습니다."
    return json.dumps(matched, ensure_ascii=False)
