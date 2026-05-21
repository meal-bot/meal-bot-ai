"""ask 분기 핸들러 v0.3.

직전 추천(last_recommendations)에 있는 메뉴 N개를 모두 RecipeStore에서 lookup하여
rag.qa.answer()에 retrieved_docs로 전달한다. 단일 메뉴 단답이 아닌 다중 메뉴
컨텍스트 기반 답변을 지원한다.

last_recommendations가 비어 있는 경우는 orchestrator가 안내 메시지로 처리하므로
이 핸들러는 호출되지 않는다.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from api.handler_result import HandlerResult
from api.schemas import ChatMessage, LastRecommendation
from rag.qa import answer as qa_answer
from rag.recipe_store import RecipeStore


logger = logging.getLogger(__name__)


def _normalize_recipe_id(recipe_id: str) -> str:
    """recipe_42 / 42 / 'recipe_42 ' 모두 '42'로 정규화. main.py 로직 재사용."""
    return str(recipe_id).strip().removeprefix("recipe_").strip()


async def handle_ask(
    message: str,
    history: Sequence[ChatMessage],
    last_recommendations: Sequence[LastRecommendation],
    recipe_store: RecipeStore,
) -> HandlerResult:
    """ask 의도를 처리한다.

    Parameters
    ----------
    message : str
        사용자 발화 (질문)
    history : Sequence[ChatMessage]
        최근 메시지 (anaphora 해소 등에 사용)
    last_recommendations : Sequence[LastRecommendation]
        직전 턴 추천. 빈 리스트면 orchestrator가 다른 경로로 처리하므로
        이 핸들러 호출 시점에는 1개 이상이라고 가정.
    recipe_store : RecipeStore
        recipe_id로 원본 dict 조회

    Returns
    -------
    HandlerResult
        intent="ask", answer=QA 답변, recommendations=[], timings={qa_ms, lookup_ms}.
        QA가 실패하거나 refused이면 flags_override에 is_fallback=true 표시.
    """
    timings: dict[str, int] = {}

    # 1. last_recs 메뉴들 lookup
    t0 = time.perf_counter()
    retrieved_docs: list[dict] = []
    missing_ids: list[str] = []
    for lr in last_recommendations:
        normalized = _normalize_recipe_id(lr.recipe_id)
        recipe = recipe_store.get_recipe_by_id(normalized)
        if recipe is None:
            missing_ids.append(lr.recipe_id)
            continue
        retrieved_docs.append(recipe)
    t1 = time.perf_counter()
    timings["lookup_ms"] = int((t1 - t0) * 1000)

    if missing_ids:
        logger.warning(
            "qa_handler: missing recipe_ids in store lookup: %s",
            missing_ids,
        )

    if not retrieved_docs:
        # last_recs는 있었지만 모두 store에 없음 → fallback
        logger.warning("qa_handler: all last_recommendations missing in store")
        return HandlerResult(
            intent="ask",
            answer="죄송해요, 직전 추천한 메뉴 정보를 찾을 수 없어요. 다시 추천을 요청해 주시겠어요?",
            recommendations=[],
            flags_override={"is_fallback": True},
            timings=timings,
        )

    # 2. chat_history dict 변환 (rag.qa.answer 시그니처에 맞춤)
    chat_history_dicts = [
        {"role": m.role, "content": m.content} for m in history
    ]

    # 3. QA 호출
    t2 = time.perf_counter()
    qa_resp = await qa_answer(
        query=message,
        retrieved_docs=retrieved_docs,
        chat_history=chat_history_dicts,
    )
    t3 = time.perf_counter()
    timings["qa_ms"] = int((t3 - t2) * 1000)

    # 4. QA 응답 → HandlerResult 매핑
    # QA 모듈의 refused/out_of_scope/qa_failed/is_fallback 플래그를
    # v0.3 flags_override 정책에 맞춰 매핑한다.
    # - refused: QA가 정책상 거부 (예: 건강 질문). is_fallback=False 유지.
    # - qa_failed: QA 내부 실패. is_fallback=True.
    # - out_of_scope: QA가 판단한 범위 밖. orchestrator의 intent 분류 단계에서
    #   이미 out_of_scope를 분기해놓았으므로 여기서는 보조 신호로만 사용.
    #   여기에 도달했다는 건 intent 분류기가 ask로 본 케이스인데
    #   QA가 다시 out_of_scope로 판정 → is_fallback=True로 표시.
    flags_override: dict[str, bool] = {}
    if qa_resp.qa_failed or qa_resp.is_fallback:
        flags_override["is_fallback"] = True
    if qa_resp.out_of_scope:
        flags_override["is_fallback"] = True
        # 응답 스키마 정책상 ask 분기에서 out_of_scope를 다시 마킹하지 않는다.
        # orchestrator가 1차 분류 단계에서만 out_of_scope 플래그를 set.
    if qa_resp.refused:
        # 거부 응답은 답변 텍스트가 그대로 노출되므로 클라이언트가 식별할 수 있게
        # flags.refused로 전파. is_fallback / out_of_scope와는 독립.
        flags_override["refused"] = True

    return HandlerResult(
        intent="ask",
        answer=qa_resp.answer,
        recommendations=[],
        flags_override=flags_override,
        timings=timings,
    )