"""refine 분기 핸들러 v0.3.

8단계 흐름:
1. LLM query 재구성 (gpt-5-mini, JSON 강제)
2. exclude_ids 구성 (직전 턴 추천 id만)
3. hybrid_retrieve_with_exclusion 실행
4. 검색 결과 0건이면 즉시 fallback 반환 (intent=ask)
5. LLM rerank (previously_recommended 전달, top_k=2)
6. top2 선정 + Recommendation 매핑
7. answer 템플릿 생성 (free_text_delta 변수)
8. HandlerResult 반환

LLM query 재구성 실패 시 결정론적 concat query로 자동 fallback (is_fallback=true).
rerank 실패는 reranker 내부 fallback에 위임 (응답의 is_fallback 플래그로 전파).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Sequence

from openai import AsyncOpenAI

from api.errors import QueryRebuildError
from api.handler_result import HandlerResult
from api.prompts.refine_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from api.schemas import LastRecommendation, Recommendation, Slots
from rag.answer_generator import (
    AnswerGeneratorError,
    AnswerTimeoutError,
    AnswerValidationError,
    generate_answer,
)
from rag.config import REFINE_TIMEOUT_SECONDS
from rag.recipe_store import RecipeStore
from rag.reranker import rerank
from rag.retriever import HybridRetriever


# answer_generator 실패 시 사용할 정적 fallback 베이스 문구.
# 기존 _build_refine_answer는 free_text_delta를 본문에 echo했지만
# baseline에서 "다른거 추천해줘 반영해서…" 같은 비문이 발견되어
# fallback에서는 그 echo를 제거한 단순 문구를 사용한다.
_REFINE_FALLBACK_ANSWER_BASE = "조건에 맞춰 다시 골라봤어요."


logger = logging.getLogger(__name__)


# ── OpenAI client lazy init (rag/ 패턴 동일) ──────────────────────────────

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise QueryRebuildError("OPENAI_API_KEY not set")
        _client = AsyncOpenAI(api_key=api_key)
    return _client


# ── 유틸 ──────────────────────────────────────────────────────────────────


def _normalize_recipe_id(recipe_id: str) -> str:
    return str(recipe_id).strip().removeprefix("recipe_").strip()


def _to_cooking_time(raw) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _format_meal_times(meal_times) -> str:
    if meal_times is None or len(meal_times) == 0:
        return "null"
    return ", ".join(meal_times)


def _format_purpose(purpose) -> str:
    return purpose if purpose else "null"


def _format_free_text(free_text) -> str:
    if free_text is None or free_text == "":
        return "null"
    return free_text


def _format_free_text_delta(free_text_delta) -> str:
    if free_text_delta is None or free_text_delta == "":
        return "null"
    return free_text_delta


def _format_last_rec_names(last_recommendations) -> str:
    if not last_recommendations:
        return "null"
    return ", ".join(lr.name for lr in last_recommendations)


def _fallback_concat_query(
    slots: Slots,
    free_text_delta: str | None,
    message: str,
) -> str:
    """LLM 실패 시 결정론적 query 구성. 단순 concat."""
    parts: list[str] = []
    if slots.meal_times:
        parts.extend(slots.meal_times)
    if slots.purpose:
        parts.append(slots.purpose)
    if slots.free_text:
        parts.append(slots.free_text)
    if free_text_delta:
        parts.append(free_text_delta)
    # message는 메타 발화가 섞여 있을 수 있지만 fallback이므로 보수적으로 포함
    parts.append(message)
    return " ".join(p for p in parts if p).strip()


# ── LLM query 재구성 ──────────────────────────────────────────────────────


async def _rebuild_query_with_llm(
    slots: Slots,
    free_text_delta: str | None,
    message: str,
    last_recommendations: Sequence[LastRecommendation],
) -> str:
    """gpt-5-mini로 검색용 자연어 query 한 줄을 생성한다.

    실패 시 QueryRebuildError 발생. 호출자가 fallback 처리.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        meal_times=_format_meal_times(slots.meal_times),
        purpose=_format_purpose(slots.purpose),
        free_text=_format_free_text(slots.free_text),
        free_text_delta=_format_free_text_delta(free_text_delta),
        message=message,
        last_rec_names=_format_last_rec_names(last_recommendations),
    )

    client = _get_client()

    try:
        # SDK timeout과 asyncio.wait_for 둘 다 같은 값으로 호출 전체(SDK 내부 hang 포함)를 보호한다.
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-5-mini",
                max_completion_tokens=1000,
                reasoning_effort="minimal",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=REFINE_TIMEOUT_SECONDS,
            ),
            timeout=REFINE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as e:
        logger.warning(
            "refine._rebuild_query_with_llm: timeout after %.1fs",
            REFINE_TIMEOUT_SECONDS,
        )
        raise QueryRebuildError(
            f"timeout after {REFINE_TIMEOUT_SECONDS}s"
        ) from e
    except Exception as e:
        logger.warning("refine._rebuild_query_with_llm: LLM call failed: %s", e)
        raise QueryRebuildError(f"LLM call failed: {e}") from e

    raw = response.choices[0].message.content
    if not raw:
        raise QueryRebuildError("LLM returned empty content")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise QueryRebuildError(f"JSON parse failed: {e}, raw={raw[:200]}") from e

    if not isinstance(parsed, dict):
        raise QueryRebuildError(f"Expected dict, got {type(parsed).__name__}")

    search_query = parsed.get("search_query")
    if not isinstance(search_query, str) or not search_query.strip():
        raise QueryRebuildError(f"Invalid search_query: {search_query!r}")

    return search_query.strip()


# ── answer 템플릿 ────────────────────────────────────────────────────────


def _build_refine_answer(free_text_delta: str | None, count: int = 2) -> str:
    """refine 정상 응답의 answer 메시지 (템플릿 + 변수). [DEPRECATED]

    answer_generator 도입(2-3단계) 이후 호출 경로에서 제외되었으나,
    rollback 대비 및 외부 import 대비를 위해 함수는 보존한다.

    count는 최종 recommendations 개수. 데이터셋 한계로 1개만 잡힌 경우
    조건이 좁다는 안내 톤으로 분기한다.
    """
    if count == 1:
        return "조건이 조금 좁아서 가장 잘 맞는 메뉴 1개를 골라봤어요."
    if free_text_delta:
        return f"{free_text_delta} 반영해서 다시 골라봤어요."
    return "조건 반영해서 다시 골라봤어요."


def _build_refine_fallback_answer(count: int) -> str:
    """answer_generator 실패 시 사용할 정적 fallback. count는 최종 추천 개수."""
    if count == 1:
        return "조건이 좁아서 1개만 추천드려요."
    return _REFINE_FALLBACK_ANSWER_BASE


# ── 핸들러 본체 ──────────────────────────────────────────────────────────


async def handle_refine(
    slots: Slots,
    free_text_delta: str | None,
    message: str,
    last_recommendations: Sequence[LastRecommendation],
    retriever: HybridRetriever,
    recipe_store: RecipeStore,
) -> HandlerResult:
    """refine 의도를 처리한다.

    last_recommendations 비어있는 케이스는 orchestrator 3단계에서 recommend로
    재분류되어 이 핸들러에 도달하지 않는다.
    """
    timings: dict[str, int] = {}
    is_fallback = False

    # 1. LLM query 재구성 (실패 시 결정론적 fallback)
    t0 = time.perf_counter()
    try:
        search_query = await _rebuild_query_with_llm(
            slots=slots,
            free_text_delta=free_text_delta,
            message=message,
            last_recommendations=last_recommendations,
        )
    except QueryRebuildError as e:
        logger.warning("refine: query rebuild fallback to concat: %s", e)
        search_query = _fallback_concat_query(slots, free_text_delta, message)
        is_fallback = True
    t1 = time.perf_counter()
    timings["query_rebuild_ms"] = int((t1 - t0) * 1000)

    # 2. exclude_ids 구성 (직전 턴 추천만)
    exclude_ids = [_normalize_recipe_id(lr.recipe_id) for lr in last_recommendations]

    # 3. hybrid retrieval with exclusion
    t2 = time.perf_counter()
    hits = retriever.search(search_query, exclude_ids=exclude_ids)
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
        cand["rrf_score"] = h.score
        candidates.append(cand)

    # 4. 0건이면 즉시 ask로 fallback
    if not candidates:
        logger.warning(
            "refine: 0 candidates after exclusion. query=%r excluded=%s",
            search_query, exclude_ids,
        )
        return HandlerResult(
            intent="ask",
            answer="조건이 너무 까다로워서 맞는 메뉴를 찾지 못했어요. 조건을 조금 풀어주시면 다시 찾아볼게요.",
            recommendations=[],
            flags_override={"is_fallback": True},
            timings=timings,
        )

    # 5. LLM rerank (previously_recommended 전달, top_k=2)
    # slots.free_text가 비어 있으면 이번 턴 delta로 폴백 (recommend와 동일 패턴).
    # Spring이 누적해서 보내주면 slots.free_text 우선, 아니면 delta로라도 신호 살림.
    effective_free_text = slots.free_text or free_text_delta
    structured_inputs = {
        "meal_times": slots.meal_times,
        "purpose": slots.purpose,
        "free_text": effective_free_text,
    }
    previously_recommended = [lr.name for lr in last_recommendations]

    t4 = time.perf_counter()
    rerank_resp = await rerank(
        query=search_query,
        candidates=candidates,
        structured_inputs=structured_inputs,
        top_k=2,
        previously_recommended=previously_recommended,
    )
    t5 = time.perf_counter()
    timings["rerank_ms"] = int((t5 - t4) * 1000)

    # 6. Recommendation 매핑
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

        main_ing = recipe.get("main_ingredients", [])
        recommendations.append(
            Recommendation(
                recipe_id=str(recipe.get("rcp_seq", normalized)),
                name=recipe.get("name", ""),
                summary=recipe.get("summary", ""),
                main_ingredients=list(main_ing) if isinstance(main_ing, list) else [],
                cooking_time=_to_cooking_time(recipe.get("cooking_time")),
                reason=item.reason,
            )
        )
        recipe_details.append(recipe)
    t7 = time.perf_counter()
    timings["lookup_ms"] = int((t7 - t6) * 1000)

    if missing_ids:
        logger.warning(
            "refine: missing recipe_ids in store lookup: requested=%d missing=%s",
            len(rerank_resp.recommendations), missing_ids,
        )

    # 응답 불변식: refine은 recommendations=[] OR len=1~2 허용 (ChatResponse validator 참조).
    # 0개일 때만 ask 폴백, 1개는 정상 응답으로 통과시킨다.
    rec_count = len(recommendations)
    if rec_count == 0:
        logger.warning(
            "refine: no recommendations after lookup (expected 1~2)",
        )
        return HandlerResult(
            intent="ask",
            answer="조건에 맞는 메뉴를 충분히 못 찾았어요. 조건을 조금 풀어주시면 다시 찾아볼게요.",
            recommendations=[],
            flags_override={"is_fallback": True},
            timings=timings,
        )
    if rec_count == 1:
        # rerank가 1건만 반환한 빈도 관찰용. 데이터셋 한계 모니터링.
        logger.info("refine: single recommendation returned (got=1)")

    # 7. answer 본문 LLM 생성 (실패 시 정적 템플릿으로 silent fallback)
    # 추천 결과 자체는 정상이며 본문 텍스트 톤만 강등되는 케이스이므로,
    # recommendations / 응답 스키마 불변식(refine: 1~2개)은 영향 받지 않는다.
    # previously_recommended_names는 rerank에 전달했던 동일 리스트를 재사용한다.
    final_recommendations = recommendations[:2]
    final_recipe_details = recipe_details[:2]

    t8 = time.perf_counter()
    answer_is_fallback = False
    try:
        answer_resp = await generate_answer(
            intent="refine",
            user_query=message,
            slots=slots,
            free_text_delta=free_text_delta,
            recommendations=final_recommendations,
            recipe_details=final_recipe_details,
            previously_recommended_names=previously_recommended,
        )
        answer_text = answer_resp.answer
    except (AnswerTimeoutError, AnswerValidationError, AnswerGeneratorError) as e:
        logger.warning(
            "refine: answer_generator failed (%s: %s), using static fallback",
            type(e).__name__, e,
        )
        answer_text = _build_refine_fallback_answer(len(final_recommendations))
        answer_is_fallback = True
    t9 = time.perf_counter()
    timings["answer_ms"] = int((t9 - t8) * 1000)

    # 8. 정상 응답
    # is_fallback은 query rebuild / rerank / answer 생성 어느 한쪽이라도
    # fallback이면 True. answer fallback은 텍스트 톤만 강등된 케이스이지만
    # 클라이언트 호환을 위해 같은 flag로 합성.
    flags_override: dict[str, bool] = {}
    if is_fallback or rerank_resp.is_fallback or answer_is_fallback:
        flags_override["is_fallback"] = True

    return HandlerResult(
        intent="refine",
        answer=answer_text,
        recommendations=final_recommendations,
        flags_override=flags_override,
        timings=timings,
    )
