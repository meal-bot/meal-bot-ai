"""ChatOrchestrator v0.3.

POST /chat의 전체 흐름을 조정한다. docs/orchestrator-v0.3.md 명세 구현.

흐름 (의사코드):
1. 입력 정규화 + 타이머 시작
2. intent 분류 (LLM 호출 1)
3. out_of_scope 즉시 반환 (ask + flags.out_of_scope=true)
4. refine + last_recs=[] → recommend로 재분류
5. 필요 시 slot 추출 (LLM 호출 2). out_of_scope는 위에서 이미 반환됨.
   slot 추출 실패 → intent별로 분기 (ask는 진행, 나머지는 slot_fill 폴백)
6. merge_slots: delta를 기존 slots에 결정론적 병합
7. intent별 핸들러 분기
   - slot_fill: 슬롯 충족 검사 후 부족하면 질문, 충족이면 recommend 재분류
   - recommend: handle_recommend 호출
   - refine: handle_refine 호출
   - ask: last_recs 비어있으면 안내 메시지, 아니면 handle_ask 호출
8. HandlerResult를 ChatResponse로 조립
9. 응답 로깅 (단계별 latency 포함)
"""

from __future__ import annotations

import logging
import time

from api.errors import IntentClassifyError, SlotExtractError
from api.handler_result import HandlerResult
from api.intent import classify_intent
from api.qa_handler import handle_ask
from api.recommend import handle_recommend
from api.refine import handle_refine
from api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Flags,
    Slots,
)
from api.slot import SlotDelta, extract_slots
from api.slot_questions import (
    build_slot_question,
    detect_slot_question,
)
from rag.recipe_store import RecipeStore
from rag.retriever import HybridRetriever


logger = logging.getLogger(__name__)


# 안내 답변 상수 (슬롯 질문 외 정형 응답)
ANSWER_OUT_OF_SCOPE = "식사/레시피 관련 질문만 도와드릴 수 있어요."
ANSWER_INTENT_FALLBACK = "다시 말씀해 주시겠어요?"
ANSWER_ASK_NO_LAST_REC = (
    "먼저 추천받은 메뉴가 있어야 해당 메뉴에 대해 답변드릴 수 있어요. "
    "어떤 메뉴를 추천해 드릴까요?"
)

# free_text 슬롯 안전망용 상수
SAFETYNET_SIGNATURE = "더 알려주실 정보"

FREETEXT_SAFETYNET_QUESTION = (
    "마지막으로, 더 알려주실 정보가 있으세요? "
    "(없으면 '없음'이나 '패스'라고 답해주세요)"
)

# 안전망 응답에서 free_text를 null로 정규화할 부정 응답 집합
# (모두 strip + lower + 끝 문장부호 제거 후 비교)
NEGATIVE_ANSWERS: frozenset[str] = frozenset({
    # 명시적
    "없음", "없어", "없어요", "없습니다", "없는데",
    # 패스 류
    "패스", "pass", "스킵", "skip", "통과",
    # 단답
    "x", "-", "ㄴ",
    # 거부
    "괜찮아", "괜찮아요", "아니", "아니요", "no",
    # 위임
    "그냥", "그냥 추천", "알아서", "아무거나", "다 좋아",
    # 모름
    "몰라", "모르겠어", "모르겠어요",
    # 무의미 단답
    "응", "네", "ㅇㅇ", "ㅇ",
})


# ── free_text 안전망 헬퍼 ──────────────────────────────────────────────────

def _needs_freetext_safetynet(slots: Slots, free_text_delta: str | None) -> bool:
    """안전망 발동 여부. slots.free_text와 delta 모두 strip 후 3자 미만이면 True."""
    def _too_short(s: str | None) -> bool:
        return s is None or len(s.strip()) < 3

    return _too_short(slots.free_text) and _too_short(free_text_delta)


def _was_freetext_safetynet_asked(history: list[ChatMessage]) -> bool:
    """직전 assistant 메시지가 안전망 질문이었는지.
    시그니처 부분 문자열 매칭(문구 미세조정에 강건).
    """
    if not history:
        return False
    last = history[-1]
    if getattr(last, "role", None) != "assistant":
        return False
    return SAFETYNET_SIGNATURE in getattr(last, "content", "")


def _normalize_freetext_negative(
    slots: Slots,
    message: str,
    free_text_delta: str | None,
) -> tuple[Slots, str | None]:
    """안전망 응답이 부정형이면 free_text를 null로 정규화."""
    normalized = message.strip().lower().rstrip(".?!~。…")
    if normalized in NEGATIVE_ANSWERS:
        return (
            Slots(
                meal_times=slots.meal_times,
                purpose=slots.purpose,
                free_text=None,
            ),
            None,
        )
    return slots, free_text_delta


class ChatOrchestrator:
    """POST /chat 요청을 받아 흐름을 조정하고 ChatResponse를 반환한다.

    의존성은 생성자 주입. FastAPI lifespan에서 1회 초기화 후 app.state에 보관.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        recipe_store: RecipeStore,
    ) -> None:
        self.retriever = retriever
        self.recipe_store = recipe_store

    # ── public ──────────────────────────────────────────────────────────

    async def handle(self, request: ChatRequest) -> ChatResponse:
        t_start = time.perf_counter()
        timings: dict[str, int] = {}

        # 1. 입력 정규화
        slots = request.slots or Slots()
        last_recs = list(request.last_recommendations or [])
        history = list(request.history[-6:])
        previous_assistant_question = detect_slot_question(history)

        # 2. intent 분류 (LLM 1)
        t0 = time.perf_counter()
        try:
            intent_result = await classify_intent(
                message=request.message,
                history=history,
                slots=slots,
                has_last_recs=bool(last_recs),
                previous_assistant_question=previous_assistant_question,
            )
        except IntentClassifyError as e:
            logger.warning("orchestrator: intent classify failed: %s", e)
            timings["intent_ms"] = int((time.perf_counter() - t0) * 1000)
            return self._build_response(
                request=request,
                slots=slots,
                handler_result=HandlerResult(
                    intent="ask",
                    answer=ANSWER_INTENT_FALLBACK,
                    recommendations=[],
                    flags_override={"is_fallback": True},
                    timings=timings,
                ),
                t_start=t_start,
                initial_intent="<classify_error>",
            )
        timings["intent_ms"] = int((time.perf_counter() - t0) * 1000)

        initial_intent = intent_result.intent

        # 3. out_of_scope 즉시 반환 (slot 추출 안 함)
        if intent_result.intent == "out_of_scope":
            return self._build_response(
                request=request,
                slots=slots,
                handler_result=HandlerResult(
                    intent="ask",
                    answer=ANSWER_OUT_OF_SCOPE,
                    recommendations=[],
                    flags_override={"out_of_scope": True},
                    timings=timings,
                ),
                t_start=t_start,
                initial_intent=initial_intent,
            )

        # 4. refine + last_recs=[] → recommend로 재분류 (정상 처리)
        effective_intent: str = intent_result.intent
        if effective_intent == "refine" and not last_recs:
            logger.info("orchestrator: refine without last_recs → recommend")
            effective_intent = "recommend"

        # 5. slot 추출 (LLM 2). ask는 슬롯 정보 필요 없지만 일관성 위해 시도하되,
        # 실패 시 intent별로 분기. 실패해도 ask는 계속 진행.
        slot_extract_failed = False
        delta: SlotDelta | None = None
        t1 = time.perf_counter()
        try:
            delta = await extract_slots(message=request.message, history=history)
        except SlotExtractError as e:
            logger.warning("orchestrator: slot extract failed: %s", e)
            slot_extract_failed = True
        timings["slot_ms"] = int((time.perf_counter() - t1) * 1000)

        # 6. merge_slots (delta가 있을 때만)
        if delta is not None:
            slots = self._merge_slots(slots, delta)

        free_text_delta = delta.free_text_delta if delta is not None else None

        # slot 추출 실패 + 추천/refine/slot_fill 의도 → slot_fill 폴백
        if slot_extract_failed and effective_intent in {"recommend", "refine", "slot_fill"}:
            answer = build_slot_question(slots)
            return self._build_response(
                request=request,
                slots=slots,
                handler_result=HandlerResult(
                    intent="slot_fill",
                    answer=answer,
                    recommendations=[],
                    flags_override={"needs_more_slots": True, "is_fallback": True},
                    timings=timings,
                ),
                t_start=t_start,
                initial_intent=initial_intent,
            )

        # 7. intent별 핸들러 분기
        if effective_intent == "slot_fill":
            # 슬롯 충족 검사. 충족되면 recommend로 재분류 (fall-through).
            if self._is_slots_sufficient(slots):
                logger.info("orchestrator: slot_fill → sufficient → recommend")
                effective_intent = "recommend"
            else:
                answer = build_slot_question(slots)
                return self._build_response(
                    request=request,
                    slots=slots,
                    handler_result=HandlerResult(
                        intent="slot_fill",
                        answer=answer,
                        recommendations=[],
                        flags_override={"needs_more_slots": True},
                        timings=timings,
                    ),
                    t_start=t_start,
                    initial_intent=initial_intent,
                )

        if effective_intent == "recommend":
            # 슬롯 충족 검사. 미충족이면 slot_fill 폴백.
            if not self._is_slots_sufficient(slots):
                answer = build_slot_question(slots)
                return self._build_response(
                    request=request,
                    slots=slots,
                    handler_result=HandlerResult(
                        intent="slot_fill",
                        answer=answer,
                        recommendations=[],
                        flags_override={"needs_more_slots": True},
                        timings=timings,
                    ),
                    t_start=t_start,
                    initial_intent=initial_intent,
                )

            # ── free_text 안전망 ──
            # 직전에 안전망을 물어봤으면 이번 답변을 정규화하고 recommend 진행.
            # 안 물어봤고 free_text가 부족하면 안전망 질문을 한 번 던지고 종료.
            if _was_freetext_safetynet_asked(history):
                slots, free_text_delta = _normalize_freetext_negative(
                    slots, request.message, free_text_delta
                )
            elif _needs_freetext_safetynet(slots, free_text_delta):
                return self._build_response(
                    request=request,
                    slots=slots,
                    handler_result=HandlerResult(
                        intent="slot_fill",
                        answer=FREETEXT_SAFETYNET_QUESTION,
                        recommendations=[],
                        flags_override={"needs_more_slots": True},
                        timings=timings,
                    ),
                    t_start=t_start,
                    initial_intent=initial_intent,
                )

            result = await handle_recommend(
                slots=slots,
                free_text_delta=free_text_delta,
                retriever=self.retriever,
                recipe_store=self.recipe_store,
            )
            # handle_recommend는 0건/부족 시 intent="slot_fill"을 반환할 수 있음
            return self._build_response(
                request=request,
                slots=slots,
                handler_result=result,
                t_start=t_start,
                initial_intent=initial_intent,
                merge_timings=True,
                base_timings=timings,
            )

        if effective_intent == "refine":
            result = await handle_refine(
                slots=slots,
                free_text_delta=free_text_delta,
                message=request.message,
                last_recommendations=last_recs,
                retriever=self.retriever,
                recipe_store=self.recipe_store,
            )
            return self._build_response(
                request=request,
                slots=slots,
                handler_result=result,
                t_start=t_start,
                initial_intent=initial_intent,
                merge_timings=True,
                base_timings=timings,
            )

        if effective_intent == "ask":
            if not last_recs:
                return self._build_response(
                    request=request,
                    slots=slots,
                    handler_result=HandlerResult(
                        intent="ask",
                        answer=ANSWER_ASK_NO_LAST_REC,
                        recommendations=[],
                        flags_override={},
                        timings=timings,
                    ),
                    t_start=t_start,
                    initial_intent=initial_intent,
                )

            result = await handle_ask(
                message=request.message,
                history=history,
                last_recommendations=last_recs,
                recipe_store=self.recipe_store,
            )
            return self._build_response(
                request=request,
                slots=slots,
                handler_result=result,
                t_start=t_start,
                initial_intent=initial_intent,
                merge_timings=True,
                base_timings=timings,
            )

        # 도달하면 안 됨. 안전 fallback.
        logger.error("orchestrator: unreachable branch. intent=%r", effective_intent)
        return self._build_response(
            request=request,
            slots=slots,
            handler_result=HandlerResult(
                intent="ask",
                answer=ANSWER_INTENT_FALLBACK,
                recommendations=[],
                flags_override={"is_fallback": True},
                timings=timings,
            ),
            t_start=t_start,
            initial_intent=initial_intent,
        )

    # ── 내부 유틸 ────────────────────────────────────────────────────────

    @staticmethod
    def _merge_slots(slots: Slots, delta: SlotDelta) -> Slots:
        """delta를 slots에 결정론적으로 병합.

        - meal_times, purpose: delta 값이 있으면 덮어쓰기 (replace)
        - free_text: FastAPI는 누적하지 않음. Spring이 누적 관리.
          여기서는 slots.free_text를 그대로 유지.
        """
        new_meal_times = delta.meal_times if delta.meal_times is not None else slots.meal_times
        new_purpose = delta.purpose if delta.purpose is not None else slots.purpose
        return Slots(
            meal_times=new_meal_times,
            purpose=new_purpose,
            free_text=slots.free_text,
        )

    @staticmethod
    def _is_slots_sufficient(slots: Slots) -> bool:
        """meal_times와 purpose가 모두 채워져 있는지 검사. free_text는 옵션."""
        return bool(slots.meal_times) and bool(slots.purpose)

    def _build_response(
        self,
        request: ChatRequest,
        slots: Slots,
        handler_result: HandlerResult,
        t_start: float,
        initial_intent: str,
        merge_timings: bool = False,
        base_timings: dict[str, int] | None = None,
    ) -> ChatResponse:
        """HandlerResult + 공통 필드를 합쳐 ChatResponse를 조립하고 로깅."""
        # flags 기본값 + override
        flags_dict = {
            "needs_more_slots": False,
            "out_of_scope": False,
            "is_fallback": False,
        }
        flags_dict.update(handler_result.flags_override or {})
        flags = Flags(**flags_dict)

        # timings 합치기 (orchestrator base + handler)
        timings: dict[str, int] = {}
        if merge_timings and base_timings:
            timings.update(base_timings)
        timings.update(handler_result.timings or {})

        # ChatResponse 조립 (스키마 model_validator가 intent ↔ recommendations 검증)
        response = ChatResponse(
            turn_id=request.turn_id,
            intent=handler_result.intent,
            answer=handler_result.answer,
            slots_updated=slots,
            recommendations=handler_result.recommendations,
            flags=flags,
        )

        # 로깅
        total_ms = int((time.perf_counter() - t_start) * 1000)
        log_payload = {
            "event": "chat",
            "session_id": request.session_id,
            "turn_id": request.turn_id,
            "message": request.message,
            "initial_intent": initial_intent,
            "final_intent": handler_result.intent,
            "slots_before": request.slots.model_dump() if request.slots else None,
            "slots_after": slots.model_dump(),
            "last_recommendation_count": len(request.last_recommendations or []),
            "recommendation_count": len(handler_result.recommendations),
            "flags": flags.model_dump(),
            "total_ms": total_ms,
            **timings,
        }
        logger.info(str(log_payload))

        return response