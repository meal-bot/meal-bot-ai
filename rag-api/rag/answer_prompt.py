"""recommend/refine answer 생성 프롬프트 & 스키마 정의.

이 파일은 프롬프트 문자열과 Structured Output용 Pydantic 모델, user prompt
빌더만 정의한다. OpenAI API 호출/연동(api/recommend.py, api/refine.py 호출부)은
별도 모듈(예: rag/answer_generator.py)에서 담당한다.

[역할]
- recommend / refine intent의 도입부 1~2문장(answer 본문)을 LLM이 생성하도록
  전환하기 위한 프롬프트.
- 추천 선정 자체는 이미 끝난 상태(recommendations 입력)이며, 본문 카드 옆에
  표시되는 도입부 멘트만 만든다.

설계 메모
- rag/rerank_prompt.py, rag/qa_prompt.py 와 동일한 모듈 패턴
  (SYSTEM_PROMPT 상수 + Pydantic 응답 스키마 + build_user_prompt 헬퍼).
- 다른 rag/ 모듈을 import하지 않는다 — 의존성은 api.schemas 두 클래스로 제한.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from api.schemas import Recommendation, Slots


# ── purpose 한글 라벨 ────────────────────────────────────────────────────────
# query_builder._PURPOSE_LABELS와 동일한 매핑을 자체적으로 보관한다.
# 의존성 최소화 원칙에 따라 rag/ 내부 모듈을 import하지 않는다.

_PURPOSE_LABELS: dict[str, str] = {
    "light":   "가볍게",
    "protein": "단백질 챙기기",
    "hearty":  "든든하게",
    "tasty":   "맛있게",
}


# ── 사용자 프롬프트에 노출할 recipe_details 필드 ──────────────────────────────
# 너무 많이 던지면 LLM이 전부 끼워넣으려 하므로 grounding에 도움되는 핵심만.
# RecipeStore lookup 결과의 원본 키 이름을 그대로 사용.
# 실제 recipes_enriched_v2.json 키 기준 확정 (2026-05-27).
# 모든 필드 1146개 레시피에 존재 검증 완료.
# spicy_level은 SYSTEM_PROMPT의 매운맛 환각 통제 룰을 LLM이 적용할 수 있도록 노출 필요.
# dish_type_tags는 도입부에서 메뉴 결("반찬", "국물요리" 등)을 짧게 짚을 때 grounding 근거.

_RECIPE_DETAIL_FIELDS: tuple[str, ...] = (
    "main_ingredients",
    "taste_tags",
    "meal_time",
    "purpose",
    "cooking_time",
    "spicy_level",
    "dish_type_tags",
)


# ── Structured Output 스키마 ──────────────────────────────────────────────────

class AnswerResponse(BaseModel):
    """recommend/refine answer 생성 결과.

    - answer: 사용자에게 노출할 도입부 본문. 30~150자 (한국어 1~2문장 분량).
      추천 카드의 reason과 중복되는 상세 사유는 작성하지 않는다.
    - used_recipe_ids: answer 본문에서 직접 언급한 메뉴의 recipe_id 목록.
      답변에 메뉴 이름이 등장한 만큼만 포함하며, recommendations에 없는 id는
      절대 포함하지 않는다 (호출부에서 추가 검증 예정).
    """

    answer: str = Field(
        min_length=30,
        max_length=150,
        description=(
            "사용자에게 노출할 도입부 본문 (30~150자, 한국어 1~2문장). "
            "추천 카드 reason과 중복되는 상세 사유는 작성하지 않는다."
        ),
    )
    used_recipe_ids: list[str] = Field(
        default_factory=list,
        description=(
            "answer 본문에서 직접 언급한 메뉴의 recipe_id 목록. "
            "recommendations에 존재하는 id만 포함한다."
        ),
    )


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 한국 음식 추천 챗봇 'Meal-bot'의 답변 작성자입니다.
추천 선정은 이미 끝났습니다. 당신의 역할은 추천 결과 카드 위에 표시될
도입부 한두 문장(answer 본문)을 작성하는 것입니다.

[역할 범위]
- 추천 메뉴를 새로 고르거나 순서를 바꾸지 않습니다. 입력으로 받은 recommendations
  순서를 그대로 받아들입니다.
- 각 추천에는 카드에 별도로 표시되는 reason이 이미 들어 있습니다. 같은 사유를
  answer 본문에 다시 풀어서 쓰지 않습니다 (중복 금지).
- answer 본문은 사용자의 요청을 받았다는 신호 + 어떤 결을 골랐는지 가볍게
  운만 띄우는 도입부입니다. 상세 설명은 카드가 담당합니다.

[답변 톤]
- 친근한 존댓말을 사용합니다. 반말 금지.
- 친근하고 생기 있게 씁니다. 문장 끝에 느낌표를 1개까지 자연스럽게 쓸 수 있고, "~딱이에요", "~즐겨보세요" 같은 부드러운 권유형 어미를 사용합니다.
- 단, 이모티콘과 과장된 평가 수식어("최고의", "완벽한", "정말 맛있는")는 쓰지 않습니다. "와!", "오!" 같은 단독 감탄사도 쓰지 않습니다. (느낌표로 생기를 주되 과장 어휘는 피한다는 의미)
- 사용자 발화(user_query)와 슬롯의 핵심 키워드("야식", "든든하게", "가볍게",
  "매콤한" 등)를 자연스럽게 미러링합니다. 단, 그대로 따옴표로 감싸 인용하지
  않고 문장에 녹입니다.
- 한국어 30~150자 사이로 작성합니다. 한 문장 또는 두 문장.

[intent별 차이]
- intent=recommend: 첫 추천 상황입니다.
  - "{시간대} {목적}에 딱 맞춰 {이름1}과(와) {이름2}로 골라봤어요!" 같은 톤.
  - 사용자 요청 결을 한 번 짚어주고 메뉴 이름을 자연스럽게 소개합니다.
- intent=refine: 재추천 상황입니다.
  - previously_recommended_names가 함께 전달됩니다. 직전 추천과 결이 어떻게
    달라졌는지 짧게 짚어주는 도입부가 좋습니다 (예: "이번엔 좀 더 매콤한
    쪽으로 {이름1}과 {이름2}로 가져왔어요!").
  - 사용자 발화를 그대로 echo하지 않습니다. 예를 들어 사용자가 "다른거 추천해줘"
    라고 했다고 "다른거 추천해줘 반영해서…"라고 쓰지 마세요. 자연스러운 문장으로
    풀어 씁니다.
  - free_text_delta가 의미 있는 조건이면 그 결을 본문에 반영합니다.

[환각 통제 — 핵심]
- 사실 환각 금지:
  - 입력으로 제공되지 않은 수치(칼로리, 단백질량, 가격 등)는 절대 언급하지 않습니다.
  - recipe_details에 없는 재료를 새로 추가하지 않습니다.
  - 조리법 동사(볶다, 끓이다, 굽다, 찌다, 튀기다 등)는 입력에 명시되지 않은
    한 사용하지 않습니다. 도입부 본문에서는 조리 단계를 묘사할 필요가 없습니다.
- 분위기/감성 형용사는 허용:
  - 입력 필드 값과 모순되지 않는 한 일반 형용사("든든한", "매콤한", "간단한",
    "담백한" 등)는 자연스러움을 위해 사용할 수 있습니다.
  - 단, 슬롯/recipe_details와 정면으로 충돌하는 형용사는 금지합니다
    (예: spicy_level=1인데 "얼큰한"이라고 쓰지 마세요).
- 메뉴 이름은 recommendations[].name에 있는 표기 그대로 사용합니다.
  축약, 변형, 별명을 만들지 않습니다.

[used_recipe_ids 작성 규칙]
- answer 본문에서 직접 이름으로 언급한 메뉴의 recipe_id만 포함합니다.
- 본문에 메뉴 이름이 1개만 등장하면 1개, 2개 다 등장하면 2개를 포함합니다.
- recommendations 입력에 없는 recipe_id는 절대 포함하지 않습니다.
- 본문에서 메뉴 이름을 하나도 언급하지 않은 경우 빈 리스트로 둡니다.

[예시]

[좋은 예시 1 — recommend, 슬롯 미러링]
- intent: recommend / 사용자: "야식으로 든든하게 먹을거 추천해줘"
- 슬롯: 시간대=야식, 목적=든든하게
- 추천: 부대찌개, 김치볶음밥
- answer: "출출한 밤 제대로 든든하게 채우고 싶으시죠? 부대찌개와 김치볶음밥으로 골라봤으니 마음에 드는 걸로 즐겨보세요!"

[좋은 예시 2 — refine, 직전 추천 대비]
- intent: refine / 사용자: "더 매콤한 걸로" / free_text_delta: "더 매콤한"
- 직전 추천: 부대찌개, 김치볶음밥 / 이번 추천: 닭발, 마라샹궈
- answer: "이번엔 매콤함을 한 단계 끌어올려 닭발과 마라샹궈로 가져왔어요!"

[좋은 예시 3 — recommend, 자유 요청 반영]
- intent: recommend / 사용자: "저녁에 가볍게 먹을건데 매운 건 빼고"
- 슬롯: 시간대=저녁, 목적=가볍게, 자유 요청=매운 건 빼고
- 추천: 두부샐러드, 닭가슴살구이
- answer: "저녁엔 맵지 않고 산뜻하게 가는 게 좋죠. 두부샐러드와 닭가슴살구이로 가볍게 골라봤어요!"

→ 공통 패턴: 슬롯/자유 요청 키워드를 인용 없이 미러링, 메뉴 이름은 정확 표기 그대로, 30~150자 준수, 카드 reason과 중복되는 사유는 본문에 쓰지 않음.

[나쁜 예시 1 — 사용자 발화 그대로 echo]
- intent: refine / 사용자: "다른거 추천해줘"
- answer: "다른거 추천해줘 반영해서 다시 골라봤어요." (X)
- 이유: 사용자 발화를 그대로 따와서 비문 발생. "이번엔 결이 다른 …을 가져왔어요." 같은 자연스러운 문장으로 풀어 써야 합니다.

[나쁜 예시 2 — 사실 환각]
- 추천: 닭갈비볶음면 (cooking_time=20, main_ingredients=["닭다리살","면","고추장"])
- answer: "단백질 25g이 듬뿍 든 닭갈비볶음면과 김치찌개를 추천드려요." (X)
- 이유: 입력에 단백질 수치도, 김치찌개도 없습니다. 수치/영양 환각 및 입력 외 메뉴 추가 금지.

[출력]
- Structured Output(JSON)으로 answer와 used_recipe_ids 두 필드만 채워 반환합니다.
- answer 길이 제약(30~150자)을 반드시 지킵니다. 너무 짧거나 너무 길게 쓰지 마세요.
"""


# ── 헬퍼: 입력 포맷터 ─────────────────────────────────────────────────────────

def _format_slots_block(slots: Slots) -> str:
    """Slots를 사람·LLM이 읽기 좋은 라인 모음으로 변환."""
    lines: list[str] = []

    if slots.meal_times:
        lines.append(f"- 시간대: {', '.join(slots.meal_times)}")
    else:
        lines.append("- 시간대: (없음)")

    if slots.purpose:
        label = _PURPOSE_LABELS.get(slots.purpose, slots.purpose)
        lines.append(f"- 목적: {label}")
    else:
        lines.append("- 목적: (없음)")

    if slots.free_text and slots.free_text.strip():
        lines.append(f"- 자유 요청(누적): {slots.free_text.strip()}")
    else:
        lines.append("- 자유 요청(누적): (없음)")

    return "\n".join(lines)


def _format_recipe_detail_value(value) -> str:
    """recipe_details 필드 값을 짧은 텍스트로 직렬화."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value if v is not None and v != "")
    return str(value)


def _format_recommendation_block(
    index: int,
    rec: Recommendation,
    detail: dict,
) -> str:
    """단일 추천(카드 정보 + 원본 상세 일부)을 텍스트 블록으로 변환."""
    lines: list[str] = [
        f"## 추천 {index}",
        f"- name: {rec.name}",
        f"- recipe_id: {rec.recipe_id}",
        f"- reason(카드에 표시됨, 본문에 중복 작성 금지): {rec.reason}",
    ]
    for field in _RECIPE_DETAIL_FIELDS:
        if field not in detail:
            continue
        formatted = _format_recipe_detail_value(detail[field])
        if not formatted:
            continue
        lines.append(f"- {field}: {formatted}")
    return "\n".join(lines)


# ── 유저 프롬프트 빌더 ────────────────────────────────────────────────────────

def build_user_prompt(
    intent: Literal["recommend", "refine"],
    user_query: str,
    slots: Slots,
    free_text_delta: str | None,
    recommendations: list[Recommendation],
    recipe_details: list[dict],
    previously_recommended_names: list[str] | None,
) -> str:
    """answer 생성용 LLM user message 텍스트를 조립한다.

    Args:
        intent: "recommend" 또는 "refine".
        user_query: 이번 턴 사용자 원문 발화.
        slots: 누적 슬롯 스냅샷 (meal_times, purpose, free_text).
        free_text_delta: 이번 턴 새로 추출된 자유 텍스트. 없으면 None.
        recommendations: 카드로 노출될 추천 결과. 1~2개.
        recipe_details: recommendations와 같은 순서·같은 길이의 RecipeStore
            lookup 결과 원본 dict 리스트. taste_tags, meal_time, purpose,
            main_ingredients, cooking_time 등 grounding용 필드 포함.
        previously_recommended_names: refine intent일 때 직전 턴 추천 메뉴명
            목록. recommend intent면 None을 전달.

    Returns:
        구조화된 user prompt 문자열.

    Raises:
        ValueError: recommendations와 recipe_details 길이가 다른 경우.
    """
    if len(recommendations) != len(recipe_details):
        raise ValueError(
            "recommendations와 recipe_details 길이가 일치해야 합니다. "
            f"recommendations={len(recommendations)}, recipe_details={len(recipe_details)}"
        )

    safe_query = user_query if user_query is not None else ""

    blocks: list[str] = []

    blocks.append(f"# intent\n{intent}")
    blocks.append(f"# 사용자 발화\n{safe_query}")
    blocks.append(f"# 누적 슬롯\n{_format_slots_block(slots)}")

    if free_text_delta and free_text_delta.strip():
        blocks.append(f"# 이번 턴 free_text_delta\n{free_text_delta.strip()}")
    else:
        blocks.append("# 이번 턴 free_text_delta\n(없음)")

    if intent == "refine":
        if previously_recommended_names:
            prev = ", ".join(previously_recommended_names)
        else:
            prev = "(없음)"
        blocks.append(f"# 직전 턴 추천 메뉴\n{prev}")

    if not recommendations:
        rec_block = "(추천 없음 — 호출 자체가 잘못된 케이스)"
    else:
        rec_blocks = [
            _format_recommendation_block(i, rec, detail)
            for i, (rec, detail) in enumerate(
                zip(recommendations, recipe_details), start=1,
            )
        ]
        rec_block = "\n\n".join(rec_blocks)
    blocks.append(f"# 이번 턴 추천 {len(recommendations)}건\n{rec_block}")

    blocks.append(
        "# 작업\n"
        "위 정보를 바탕으로 사용자에게 노출할 answer 도입부 본문을 작성하시오.\n"
        "- 30~150자, 한국어 1~2문장.\n"
        "- 메뉴 이름은 recommendations[].name 표기를 그대로 사용한다.\n"
        "- 카드 reason과 같은 사유를 본문에 반복하지 않는다.\n"
        "- 입력에 없는 수치/재료/조리법 동사는 절대 만들지 않는다.\n"
        "- 본문에서 언급한 메뉴의 recipe_id만 used_recipe_ids에 담는다."
    )

    return "\n\n".join(blocks)


__all__ = ["SYSTEM_PROMPT", "AnswerResponse", "build_user_prompt"]
