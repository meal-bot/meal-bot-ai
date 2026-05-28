"""POST /chat 엔드포인트용 Pydantic 요청/응답 스키마 v0.3.

v0.2까지 있던 /recommend, /ask는 폐기되었고 /chat 단일 엔드포인트로 통합되었다.
설계 문서: docs/orchestrator-v0.3.md
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from rag.config import HISTORY_MAX_MESSAGES


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
    """POST /chat 요청. Spring이 슬라이딩 윈도우로 history 전달.
    윈도우 크기는 rag.config.HISTORY_MAX_MESSAGES (기본 50).
    slots.free_text는 Spring 누적. last_recommendations는 직전 턴만."""

    session_id: str = Field(min_length=1, max_length=100)
    turn_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=500)
    history: list[ChatMessage] = Field(default_factory=list, max_length=HISTORY_MAX_MESSAGES)
    slots: Slots = Field(default_factory=Slots)
    last_recommendations: list[LastRecommendation] = Field(
        default_factory=list, max_length=5
    )


class ChatResponse(BaseModel):
    """POST /chat 응답. intent는 응답 시점 최종 분류.
    recommendations는 intent에 따라 개수가 정해진다.
    slot_fill/ask는 0개, recommend는 정확히 2개, refine은 1~2개.
    slots_updated는 항상 전체 슬롯 스냅샷.
    free_text_delta는 이번 턴 AI가 추출한 자유 텍스트이며 Spring 누적용이다.
    free_text_delta는 intent in {"recommend", "refine", "slot_fill"}일 때만 채워질 수 있고,
    ask/out_of_scope/refused 케이스에서는 항상 null이다 (orchestrator 가드).
    """

    turn_id: str = Field(min_length=1, max_length=100)
    intent: Literal["recommend", "slot_fill", "refine", "ask"]
    answer: str = Field(min_length=1, max_length=2000)
    slots_updated: Slots
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=2)
    flags: Flags
    free_text_delta: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _check_intent_recommendations_consistency(self) -> "ChatResponse":
        # intent ↔ recommendations 길이 일관성. orchestrator 안전망(1차 방어선 아님).
        # recommend는 첫 추천이라 정확히 2개를 보장한다.
        # refine은 조건이 좁아진 재추천이라 1~2개를 허용한다 (데이터셋 한계로 1개도 정상).
        n = len(self.recommendations)
        if self.intent in ("slot_fill", "ask"):
            if n != 0:
                raise ValueError(
                    f"intent={self.intent!r}는 recommendations=[]여야 함 (got {n}개)"
                )
        elif self.intent == "recommend":
            if n != 2:
                raise ValueError(
                    f"intent={self.intent!r}는 recommendations 길이=2여야 함 (got {n}개)"
                )
        else:  # refine
            if not (1 <= n <= 2):
                raise ValueError(
                    f"intent={self.intent!r}는 recommendations 길이=1~2여야 함 (got {n}개)"
                )
        return self


# === Recipe Detail (GET /recipes/{recipe_id}) ===
# 모달 표시용 단건 조회 응답 스키마. recipes_enriched_v2.json 28개 필드 중
# 모달이 실제로 쓰는 13개만 노출하고, 내부 메타 필드(enrichment_flags,
# question_1~3, ingredient_count_*, ingredients raw, hash_tag, cooking_way,
# ingredients_clean, meal_time, purpose)는 응답에서 제외한다.
# rcp_seq → recipe_id 키 변환은 라우트 핸들러에서 수행 (스키마에 validator 없음).


class RecipeManualStep(BaseModel):
    """조리법 단계. img는 1145/1146건에서 채워져 있지만 manuals 자체 결측 1건 보호용 Optional."""

    step: int
    desc: str
    img: str | None = None


class RecipeIngredient(BaseModel):
    """식재료 한 항목."""

    name: str
    amount: str
    note: str | None = None


class RecipeIngredientsStructured(BaseModel):
    """식재료를 main/sauce/garnish 3 카테고리로 구조화.

    1146건 모두 세 키가 존재하며, 비어있을 수 있어 sauce/garnish는 default 빈 배열.
    """

    main: list[RecipeIngredient]
    sauce: list[RecipeIngredient] = Field(default_factory=list)
    garnish: list[RecipeIngredient] = Field(default_factory=list)


class RecipeNutrition(BaseModel):
    """영양 정보. 1146건 전체에서 5개 키 모두 채워져 있어 non-optional."""

    energy_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    sodium_mg: float


class RecipeDetail(BaseModel):
    """GET /recipes/{recipe_id} 응답 스키마.

    모달 표시에 필요한 13개 필드만 노출. 내부 메타는 응답에 포함하지 않는다.
    """

    recipe_id: str = Field(..., description="recipes_enriched_v2.json의 rcp_seq")
    name: str
    summary: str
    category: str
    main_ingredients: list[str]
    taste_tags: list[str] = Field(default_factory=list)
    dish_type_tags: list[str] = Field(default_factory=list)
    cooking_time: int  # 분 단위
    difficulty: str    # 쉬움/보통/어려움
    spicy_level: int   # 1~4
    nutrition: RecipeNutrition
    ingredients_structured: RecipeIngredientsStructured
    manuals: list[RecipeManualStep] = Field(default_factory=list)
    img_main: str | None = None
    img_thumb: str | None = None