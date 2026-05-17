"""FastAPI 엔드포인트용 Pydantic 요청/응답 스키마."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """대화 히스토리의 한 턴. QA 후속 질문에서 anaphora 해소용."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class Recommendation(BaseModel):
    """추천 결과 단건. reranker가 매긴 rank와 reason 포함."""

    rank: int = Field(ge=1, le=5)
    recipe_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    image_url: str | None = None
    summary: str
    kcal: float | None = None
    cooking_time: int | None = None
    reason: str
    matched_intents: list[str] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    """POST /recommend 요청 바디. Stateless — session_id/turn_id는 로깅용."""

    meal_times: list[Literal["아침", "점심", "저녁", "간식", "야식"]] = Field(min_length=1)
    purpose: Literal["light", "protein", "hearty", "tasty"]
    spicy_max: int = Field(ge=1, le=4)
    free_text: str | None = Field(default=None, max_length=200)
    session_id: str = Field(min_length=1, max_length=100)
    turn_id: str = Field(min_length=1, max_length=100)


class RecommendResponse(BaseModel):
    """POST /recommend 응답. recommendations 길이 0~5, insufficient_matches로 부족 신호."""

    turn_id: str
    recommendations: list[Recommendation]
    insufficient_matches: bool
    is_fallback: bool


class AskRequest(BaseModel):
    """POST /ask 요청 바디. 후속 질문 + 대상 recipe_id + chat_history."""

    question: str = Field(min_length=1, max_length=500)
    recipe_id: str = Field(min_length=1)
    chat_history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    session_id: str = Field(min_length=1, max_length=100)
    turn_id: str = Field(min_length=1, max_length=100)


class AskResponse(BaseModel):
    """POST /ask 응답. refused/qa_failed/is_fallback 플래그로 상태 구분."""

    turn_id: str
    answer: str
    used_fields: list[str] = Field(default_factory=list)
    refused: bool
    out_of_scope: bool
    qa_failed: bool
    is_fallback: bool


class HealthResponse(BaseModel):
    """GET /health 응답. 정상이면 status='ok'."""

    status: Literal["ok"]
