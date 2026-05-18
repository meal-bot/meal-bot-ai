"""Retriever → Rerank → QA 통합 동작 검증 (일회성).

멀티턴 후속 질문 흐름을 콘솔로 출력해 다음을 눈으로 확인한다.
1. Turn 1: HybridRetriever → rerank top-N
2. Turn 2: rerank의 첫 번째 레시피에 대한 재료 질문
3. Turn 3: 동일 레시피의 조리법 질문 (chat_history 누적 상태)

파일 저장 없이 콘솔 출력만. 외부 의존성 추가 없음.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import DenseRetriever, BM25Retriever, HybridRetriever, Hit
from rag.reranker import rerank
from rag.qa import answer


# ── 경로 / 상수 ───────────────────────────────────────────────────────────────

ENRICHED_DATA = Path("data/recipes_enriched_v2.json")
HYBRID_TOP_K  = 10

TURN1_QUERY = "저녁에 먹을 안 매운 국물 요리 추천해줘"
TURN2_QUERY = "첫 번째 거 재료 뭐야?"
TURN3_QUERY = "어떻게 만들어?"

SEP = "=" * 64


# ── lookup 헬퍼 ──────────────────────────────────────────────────────────────

def load_full_recipe_lookup(path: Path) -> dict[str, dict]:
    """recipes_enriched_v2.json을 rcp_seq → 레시피 dict 전체로 매핑."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {str(r["rcp_seq"]): r for r in data if r.get("rcp_seq")}


# ── candidate 변환 ───────────────────────────────────────────────────────────
#
# NOTE: run_rerank_baseline.py는 baseline JSON(이미 평탄화된 후보 dict)에
#       4필드(spicy_level/main_ingredients/meal_time/purpose)만 보강했다.
#       이 통합 스크립트는 HybridRetriever를 직접 호출하므로 Hit.metadata가
#       dense(ChromaDB JSON 문자열) / bm25(json 원본)로 섞여 있다.
#       full_lookup 한 곳에서 모든 필드를 가져오면 JSON 파싱과 4필드 보강이
#       동시에 해결되므로 여기서는 단일 lookup으로 통합한다.

def hit_to_candidate(hit: Hit, full_lookup: dict[str, dict]) -> dict:
    """Hit + full_lookup → reranker가 기대하는 후보 dict."""
    rid = hit.recipe_id
    full = full_lookup.get(rid)
    if full is None:
        raise KeyError(
            f"recipe_id={rid} 가 recipes_enriched_v2.json에 없음 "
            f"(rerank 후보 구성 불가)"
        )

    candidate = dict(full)
    candidate["recipe_id"]  = rid
    candidate["dense_rank"] = hit.dense_rank
    candidate["bm25_rank"]  = hit.bm25_rank
    candidate["rrf_score"]  = hit.score
    return candidate


def build_recommendation_text(rerank_response, full_lookup: dict[str, dict]) -> str:
    """rerank 응답을 assistant 메시지용 텍스트로 변환."""
    lines = ["다음 메뉴를 추천드립니다."]
    for r in rerank_response.recommendations:
        name = full_lookup.get(r.recipe_id, {}).get("name", "?")
        lines.append(f"{r.rank}. {name}")
    return "\n".join(lines)


# ── 출력 헬퍼 ─────────────────────────────────────────────────────────────────

def print_header(turn_n: int, query: str) -> None:
    print(SEP)
    print(f"=== Turn {turn_n}: {query} ===")
    print(SEP)


def print_history_state(chat_history: list[dict]) -> None:
    print(f"[chat_history 누적 길이: {len(chat_history)} 메시지]")
    print()


def print_qa_response(qa_resp) -> None:
    print("answer:")
    print(qa_resp.answer)
    print()
    print(f"used_fields: {qa_resp.used_fields}")
    print(f"refused:     {qa_resp.refused}")
    print(f"qa_failed:   {qa_resp.qa_failed}")
    print(f"is_fallback: {qa_resp.is_fallback}")
    print()


# ── Turn 함수 ────────────────────────────────────────────────────────────────

async def run_turn1(
    query: str,
    hybrid: HybridRetriever,
    full_lookup: dict[str, dict],
):
    """Turn 1: hybrid search → rerank."""
    print_header(1, query)

    hits = hybrid.search(query, top_k=HYBRID_TOP_K)
    print(f"hybrid hits: {len(hits)}")
    if not hits:
        raise RuntimeError("hybrid 결과 비어 있음")

    candidates = [hit_to_candidate(h, full_lookup) for h in hits]

    rerank_resp = await rerank(query, candidates)
    print(
        f"rerank: {len(rerank_resp.recommendations)}개 반환, "
        f"insufficient={rerank_resp.insufficient_matches}"
    )
    print()

    for r in rerank_resp.recommendations:
        name = full_lookup.get(r.recipe_id, {}).get("name", "?")
        intents = ", ".join(r.matched_intents)
        print(f"  {r.rank}. recipe_id={r.recipe_id} | {name}")
        print(f"     reason: {r.reason}")
        print(f"     intents: [{intents}]")
    print()
    return rerank_resp


async def run_qa_turn(
    turn_n:  int,
    query:   str,
    target_doc: dict,
    chat_history: list[dict],
):
    """Turn 2/3: qa.answer 호출."""
    print_header(turn_n, query)
    qa_resp = await answer(query, [target_doc], chat_history)
    print_qa_response(qa_resp)
    return qa_resp


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def main():
    print("== Loading retrievers ==")
    dense  = DenseRetriever()
    bm25   = BM25Retriever()
    hybrid = HybridRetriever(dense, bm25)

    print("== Loading full recipe lookup ==")
    full_lookup = load_full_recipe_lookup(ENRICHED_DATA)
    print(f"lookup entries: {len(full_lookup)}")
    print()

    chat_history: list[dict] = []

    # ── Turn 1 ──────────────────────────────────────────────────────────────
    try:
        rerank_resp = await run_turn1(TURN1_QUERY, hybrid, full_lookup)
    except Exception as e:
        print(f"[FATAL] Turn 1 실패: {e}")
        return

    if not rerank_resp.recommendations:
        print("[FATAL] rerank 추천 비어 있음. 후속 Turn 진행 불가.")
        return

    first_rid = rerank_resp.recommendations[0].recipe_id
    first_doc = full_lookup.get(first_rid)
    if first_doc is None:
        print(
            f"[FATAL] rerank top-1 recipe_id={first_rid} 가 "
            f"recipes_enriched_v2.json에 없음. 중단."
        )
        return

    rec_text = build_recommendation_text(rerank_resp, full_lookup)
    chat_history.append({"role": "user",      "content": TURN1_QUERY})
    chat_history.append({"role": "assistant", "content": rec_text})
    print_history_state(chat_history)

    # ── Turn 2 ──────────────────────────────────────────────────────────────
    try:
        qa2 = await run_qa_turn(2, TURN2_QUERY, first_doc, chat_history)
    except Exception as e:
        print(f"[FATAL] Turn 2 실패: {e}")
        return

    chat_history.append({"role": "user",      "content": TURN2_QUERY})
    chat_history.append({"role": "assistant", "content": qa2.answer})
    print_history_state(chat_history)

    # ── Turn 3 ──────────────────────────────────────────────────────────────
    try:
        qa3 = await run_qa_turn(3, TURN3_QUERY, first_doc, chat_history)
    except Exception as e:
        print(f"[FATAL] Turn 3 실패: {e}")
        return

    chat_history.append({"role": "user",      "content": TURN3_QUERY})
    chat_history.append({"role": "assistant", "content": qa3.answer})
    print_history_state(chat_history)

    print(SEP)
    print("DONE")
    print(SEP)


if __name__ == "__main__":
    asyncio.run(main())