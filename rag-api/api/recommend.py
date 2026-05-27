"""recommend 분기 핸들러 v0.3.

기존 api/main.py의 /recommend 로직을 모듈화한 버전.
- spicy_max 제거 (v0.3 슬롯에서 삭제)
- top_k=2 (응답 스키마가 2개로 제한)
- v0.3 Recommendation 스키마에 맞춰 매핑
- 0건 retrieval 시 slot_fill 폴백 (orchestrator-v0.3.md 정책)
"""

from __future__ import annotations

import logging
import time

from api.handler_result import HandlerResult
from api.schemas import Recommendation, Slots
from rag.answer_generator import (
    AnswerGeneratorError,
    AnswerTimeoutError,
    AnswerValidationError,
    generate_answer,
)
from rag.query_builder import build_retrieval_query
from rag.recipe_store import RecipeStore
from rag.reranker import rerank
from rag.retriever import HybridRetriever


# 기존 정적 answer 템플릿. answer_generator가 실패하면 이 값으로 silent fallback.
_RECOMMEND_FALLBACK_ANSWER = "조건에 맞춰 2개 골라봤어요."


logger = logging.getLogger(__name__)


def _normalize_recipe_id(recipe_id: str) -> str:
    """recipe_42 / 42 / 'recipe_42 ' 모두 '42'로 정규화."""
    return str(recipe_id).strip().removeprefix("recipe_").strip()


def _to_cooking_time(raw) -> int | None:
    """raw 값을 int로 안전 변환, 실패 시 None."""
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def handle_recommend(
    slots: Slots,
    free_text_delta: str | None,
    user_query: str,
    retriever: HybridRetriever,
    recipe_store: RecipeStore,
) -> HandlerResult:
    """recommend 의도를 처리한다.

    Parameters
    ----------
    slots : Slots
        v0.3 slots 스냅샷. meal_times/purpose는 충족 상태 가정
        (orchestrator가 슬롯 충족 검사 후 호출).
    free_text_delta : str | None
        이번 턴 extract_slots가 뽑은 자유텍스트 delta.
        slots.free_text(Spring 누적)가 비어 있을 때 신호 보강용 폴백.
    user_query : str
        이번 턴 사용자 원문 발화. answer 본문 LLM 생성 시 입력으로 전달.
    retriever : HybridRetriever
    recipe_store : RecipeStore

    Returns
    -------
    HandlerResult
        정상: intent="recommend", recommendations=2개
        0건: intent="slot_fill", recommendations=[], flags_override={needs_more_slots, is_fallback}
        rerank 실패: intent="recommend", recommendations=retrieval score top2, is_fallback=true
    """
    timings: dict[str, int] = {}

    # slots.free_text가 비어 있으면 이번 턴 delta로 폴백.
    # Spring이 누적해서 보내주면 slots.free_text 우선, 아니면 delta로라도 신호 살림.
    effective_free_text = slots.free_text or free_text_delta

    # 1. query 빌드
    t0 = time.perf_counter()
    query = build_retrieval_query(
        meal_times=slots.meal_times,
        purpose=slots.purpose,
        free_text=effective_free_text,
    )
    t1 = time.perf_counter()
    timings["query_build_ms"] = int((t1 - t0) * 1000)

    # 2. hybrid retrieval
    t2 = time.perf_counter()
    hits = retriever.search(query)
    t3 = time.perf_counter()
    timings["retrieval_ms"] = int((t3 - t2) * 1000)

    # Hit의 retrieval 신호(dense_rank/bm25_rank/RRF score)를 rerank LLM에 전달하기 위해
    # metadata를 얕은 복사한 뒤 세 필드를 머지. 한쪽에만 잡힌 후보는 반대편 rank가 None.
    candidates: list[dict] = []
    for h in hits:
        if not h.metadata:
            continue
        cand = dict(h.metadata)
        cand["dense_rank"] = h.dense_rank
        cand["bm25_rank"] = h.bm25_rank
        cand["rrf_score"] = h.score  # source="hybrid"일 때 score가 RRF 누적 점수
        candidates.append(cand)

    # 3. 0건이면 즉시 slot_fill 폴백
    if not candidates:
        logger.warning(
            "recommend: 0 candidates from retrieval. query=%r slots=%r",
            query, slots.model_dump(),
        )
        return HandlerResult(
            intent="slot_fill",
            answer="조건에 맞는 메뉴를 못 찾았어요. 조건을 조금 풀어볼까요?",
            recommendations=[],
            flags_override={"needs_more_slots": True, "is_fallback": True},
            timings=timings,
        )

    # 4. LLM rerank (top_k=2)
    structured_inputs = {
        "meal_times": slots.meal_times,
        "purpose": slots.purpose,
        "free_text": effective_free_text,
    }
    t4 = time.perf_counter()
    rerank_resp = await rerank(
        query=query,
        candidates=candidates,
        structured_inputs=structured_inputs,
        top_k=2,
        previously_recommended=None,  # recommend 분기에서는 None
    )
    t5 = time.perf_counter()
    timings["rerank_ms"] = int((t5 - t4) * 1000)

    # 5. rerank 결과 → Recommendation 매핑
    # answer_generator에 raw 레시피 dict가 필요하므로 lookup 결과를 병렬 보관.
    # RecipeStore.get_recipe_by_id 추가 호출 없이 같은 lookup 결과를 재사용한다.
    t6 = time.perf_counter()
    recommendations: list[Recommendation] = []
    recipe_details: list[dict] = []
    missing_ids: list[str] = []
    for item in rerank_resp.recommendations:
        normalized = _normalize_recipe_id(item.recipe_id)
        recipe = recipe_store.get_recipe_by_id(normalized)
        if recipe is None:
            missing_ids.append(item.recipe_id)
            continue

        main_ing_raw = recipe.get("main_ingredients")
        main_ingredients = list(main_ing_raw) if isinstance(main_ing_raw, list) else []

        recommendations.append(
            Recommendation(
                recipe_id=str(recipe.get("rcp_seq", normalized)),
                name=recipe.get("name", ""),
                summary=recipe.get("summary", ""),
                main_ingredients=main_ingredients,
                cooking_time=_to_cooking_time(recipe.get("cooking_time")),
                reason=item.reason,
            )
        )
        recipe_details.append(recipe)
    t7 = time.perf_counter()
    timings["lookup_ms"] = int((t7 - t6) * 1000)

    if missing_ids:
        logger.warning(
            "recommend: missing recipe_ids in store lookup: requested=%d missing=%s",
            len(rerank_resp.recommendations),
            missing_ids,
        )

    # 6. recommendations가 2개 미만이면 fallback 처리
    # 응답 불변식: recommendations=[] OR recommendations.length=2
    # 1개만 잡혔으면 0건과 동일하게 slot_fill 폴백 (불변식 위반 방지)
    if len(recommendations) < 2:
        logger.warning(
            "recommend: insufficient recommendations after lookup: got=%d (expected 2)",
            len(recommendations),
        )
        return HandlerResult(
            intent="slot_fill",
            answer="조건에 맞는 메뉴를 충분히 못 찾았어요. 조건을 조금 풀어볼까요?",
            recommendations=[],
            flags_override={"needs_more_slots": True, "is_fallback": True},
            timings=timings,
        )

    # 7. answer 본문 LLM 생성 (실패 시 정적 템플릿으로 silent fallback)
    # 추천 결과 자체는 정상이며 본문 텍스트 톤만 강등되는 케이스이므로,
    # recommendations / 응답 스키마 불변식은 영향 받지 않는다.
    final_recommendations = recommendations[:2]
    final_recipe_details = recipe_details[:2]

    t8 = time.perf_counter()
    answer_is_fallback = False
    try:
        answer_resp = await generate_answer(
            intent="recommend",
            user_query=user_query,
            slots=slots,
            free_text_delta=free_text_delta,
            recommendations=final_recommendations,
            recipe_details=final_recipe_details,
            previously_recommended_names=None,
        )
        answer_text = answer_resp.answer
    except (AnswerTimeoutError, AnswerValidationError, AnswerGeneratorError) as e:
        logger.warning(
            "recommend: answer_generator failed (%s: %s), using static fallback",
            type(e).__name__, e,
        )
        answer_text = _RECOMMEND_FALLBACK_ANSWER
        answer_is_fallback = True
    t9 = time.perf_counter()
    timings["answer_ms"] = int((t9 - t8) * 1000)

    # 8. 정상 응답
    # is_fallback은 rerank 또는 answer 생성 어느 한쪽이라도 fallback이면 True.
    # answer fallback은 텍스트 톤이 정적 템플릿으로 강등된 케이스로,
    # recommendations 자체는 정상이지만 클라이언트 호환을 위해 같은 flag로 합성.
    flags_override: dict[str, bool] = {}
    if rerank_resp.is_fallback or answer_is_fallback:
        flags_override["is_fallback"] = True

    return HandlerResult(
        intent="recommend",
        answer=answer_text,
        recommendations=final_recommendations,  # 안전장치 (이미 [:2] 슬라이스됨)
        flags_override=flags_override,
        timings=timings,
    )
