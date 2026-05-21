"""POST /chat 엔드포인트용 Pydantic 요청/응답 스키마 v0.3.

v0.2까지 있던 /recommend, /ask는 폐기되었고 /chat 단일 엔드포인트로 통합되었다.
설계 문서: docs/orchestrator-v0.3.md
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    """GET /healthz 응답. 정상이면 status='ok'."""

    status: Literal["ok"]


class ChatMessage(BaseModel):
    """대화 히스토리의 한 메시지. orchestrator 흐름 v0.3 history 블록 요소."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class Slots(BaseModel):
    """현재 슬롯 스냅샷. null은 '해당 슬롯에 아직 값 없음'. meal_times는 복수 가능."""

    meal_times: list[Literal["아침", "점심", "저녁", "간식", "야식"]] | None = None
    purpose: Literal["light", "protein", "hearty", "tasty"] | None = None
    free_text: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _normalize_meal_times(self) -> "Slots":
        # 빈 배열은 "값 없음"으로 정규화. orchestrator/LLM 산출물 양쪽에서 안전망.
        if self.meal_times is not None and len(self.meal_times) == 0:
            self.meal_times = None
        return self


class LastRecommendation(BaseModel):
    """직전 턴 추천 요약. refine exclude / ask 컨텍스트용. id는 정규화 형태(예: '42')."""

    recipe_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class Recommendation(BaseModel):
    """추천 결과 단건. reason은 60~120자 권장, 스키마 제약은 10~200자 여유 둠.
    하한 10자는 rerank fallback 짧은 메시지("유사한 후보로 추천되었습니다.") 수용용."""

    recipe_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    summary: str
    main_ingredients: list[str] = Field(default_factory=list)
    cooking_time: int | None = None
    reason: str = Field(min_length=10, max_length=200)


class Flags(BaseModel):
    """응답 플래그. 결합 규칙은 orchestrator-v0.3.md 참조.

    refused는 기존 응답과의 호환을 위해 default=False 옵션 필드로 둔다.
    QA 거부 응답을 클라이언트가 식별할 수 있도록 P0 패치로 추가됨.
    """

    needs_more_slots: bool
    out_of_scope: bool
    is_fallback: bool
    refused: bool = False


class ChatRequest(BaseModel):
    """POST /chat 요청. Spring이 슬라이딩 윈도우(메시지 6개)로 history 전달.
    slots.free_text는 Spring 누적. last_recommendations는 직전 턴만."""

    session_id: str = Field(min_length=1, max_length=100)
    turn_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=500)
    history: list[ChatMessage] = Field(default_factory=list, max_length=6)
    slots: Slots = Field(default_factory=Slots)
    last_recommendations: list[LastRecommendation] = Field(
        default_factory=list, max_length=5
    )


class ChatResponse(BaseModel):
    """POST /chat 응답. intent는 응답 시점 최종 분류.
    recommendations는 0개 또는 2개. slots_updated는 항상 전체 스냅샷."""

    turn_id: str = Field(min_length=1, max_length=100)
    intent: Literal["recommend", "slot_fill", "refine", "ask"]
    answer: str = Field(min_length=1, max_length=2000)
    slots_updated: Slots
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=2)
    flags: Flags

    @model_validator(mode="after")
    def _check_intent_recommendations_consistency(self) -> "ChatResponse":
        # intent ↔ recommendations 길이 일관성. orchestrator 안전망(1차 방어선 아님).
        n = len(self.recommendations)
        if self.intent in ("slot_fill", "ask"):
            if n != 0:
                raise ValueError(
                    f"intent={self.intent!r}는 recommendations=[]여야 함 (got {n}개)"
                )
        else:  # recommend, refine
            if n != 2:
                raise ValueError(
                    f"intent={self.intent!r}는 recommendations 길이=2여야 함 (got {n}개)"
                )
        return self