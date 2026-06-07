"""LLM rerank 모듈용 프롬프트 & 스키마 정의.

이 파일은 프롬프트 문자열과 Structured Output용 Pydantic 모델만 정의한다.
OpenAI API 호출/Reranker 클래스/FastAPI 래핑은 다른 파일에서 담당한다.
"""

import json

from pydantic import BaseModel, Field


# ── 허용 intent ───────────────────────────────────────────────────────────────

ALLOWED_INTENTS: list[str] = [
    "시간_빠름", "조리_간단", "한끼",
    "국물요리", "반찬", "디저트", "샐러드",
    "저칼로리", "고단백", "가벼운_식사",
    "매운맛_낮음", "매운맛_있음",
    "부드러운식감", "바삭한식감",
    "담백한맛", "매콤한맛", "달콤한맛",
    "아이_적합", "어르신_적합",
    "혼밥", "야식", "다이어트", "손님상",
    "재료_일치",
    "시간대_아침", "시간대_저녁",
]


# ── Structured Output 스키마 ──────────────────────────────────────────────────

class RerankItem(BaseModel):
    """단일 추천 결과 항목."""

    rank: int = Field(ge=1, le=10)
    recipe_id: str
    reason: str
    matched_intents: list[str] = Field(min_length=1, max_length=4)


class RerankResponse(BaseModel):
    """LLM rerank 응답 전체.

    - 조건 부합 후보가 지정 개수 미만일 수 있으므로 recommendations에 min_length를 두지 않는다.
    - 부족하게 반환할 때는 insufficient_matches=True로 표시한다.
    - LLM 호출/검증 실패 후 hybrid 순위로 대체했을 때는 is_fallback=True로 표시한다.
    """

    recommendations: list[RerankItem] = Field(max_length=10)
    insufficient_matches: bool
    is_fallback: bool = False


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""당신은 한국 가정식 레시피 추천 reranker입니다.

[역할]
- 사용자의 자유 질의와 레시피 후보 목록을 받아 사용자가 지정한 개수의 추천을 생성합니다.
- 후보를 새로 만들지 않고, 제공된 후보 목록 중에서만 고릅니다.

[후보 선택 규칙]
- 반드시 제공된 후보 목록 안의 recipe_id만 선택합니다.
- 후보 목록에 없는 recipe_id를 생성하거나 추측하지 않습니다.
- recipe_id는 중복 없이 선택합니다.
- rank는 1부터 시작하며 중복 없이 오름차순으로 부여합니다.
- 사용자 제약을 가장 잘 만족하는 후보를 높은 순위로 둡니다.
- 시간, 식감, 음식 유형, 재료, 상황 조건을 우선 고려합니다.
- 사용자 제약을 명확히 위반하는 후보는 낮은 순위로 보내거나 제외합니다.
- 제약을 위반하는 후보를 억지로 좋게 설명하지 않습니다.
- 조건을 만족하는 후보가 지정된 개수 미만이면 만족하는 만큼만 반환하고 insufficient_matches=true로 설정합니다.
- 조건을 만족하는 후보가 충분하면 지정된 개수를 정확히 반환하고 insufficient_matches=false로 설정합니다.

[자유 요청 우선순위]
- 사용자 제약에 "자유 요청"이 있으면, 시간대(meal_times)와 목적(purpose)보다 더 구체적인 신호로 간주합니다.
- 자유 요청에서 키워드(재료, 식감, 상황, 조리법, 제외 조건 등)를 식별하고, 후보의 metadata(category, dish_type_tags, texture_tags, summary, main_ingredients 등)와 직접 대조합니다.
- 자유 요청과 부합하는 정도에 따라 순위를 조정합니다. 단, 자유 요청만으로 후보를 전부 제외해 빈 결과를 반환하지는 않습니다.
- 자유 요청을 부분적으로만 만족하는 후보도 결과에 포함될 수 있으며, 그 경우 reason에서 어느 부분을 충족하는지(또는 부족한지) 솔직하게 명시합니다.
- 자유 요청이 없거나 비어 있으면 시간대와 목적만으로 판단합니다.

[retrieval 신호 해석]
- 각 후보에는 다음 세 필드가 포함될 수 있습니다.
  - dense_rank: 의미 유사도 검색에서의 순위. 낮을수록 사용자 발화와 의미적으로 가깝습니다.
  - bm25_rank: 키워드 매칭 검색에서의 순위. 낮을수록 키워드가 잘 맞습니다.
  - rrf_score: 두 신호의 융합 점수. 높을수록 종합적으로 우수합니다.
- 값이 null이면 해당 검색 방식에서 잡히지 않은 것이며, 그 자체로 후보를 배제하는 근거는 아닙니다.
- 이 신호들은 참고용입니다. 다음 우선순위를 따르세요.
  1. 사용자 정형 슬롯(시간대, 목적) 매칭 여부 — 가장 중요
  2. 자유 요청 매칭, 특히 부정 조건("빼고", "말고", "제외") 위반 여부
  3. taste_tags, texture_tags, dish_type_tags 등 의미 매칭
  4. dense_rank/bm25_rank/rrf_score는 위 신호들이 동등하거나 판단이 어려울 때 tie-breaker로만 활용합니다.
- retrieval 순위가 높다는 이유만으로 슬롯/자유 요청과 어긋나는 후보를 추천하지 않습니다.

[매운맛 처리]
- 후보의 spicy_level은 매운맛 정도를 나타냅니다.
  - 1: 안 매움
  - 2: 약간 매움
  - 3: 매움
  - 4: 아주 매움
  - null: 정보 없음
- 사용자가 매운맛 회피를 요청하면(예: "안 매운", "매운 거 빼고", "맵지 않은") spicy_level이 낮은 후보를 우선 선택하고, spicy_level=3 또는 4 후보는 추천하지 않습니다.
- 사용자가 매운맛을 선호하면(예: "매콤한", "얼큰한") spicy_level=2 이상을 우선 선택합니다.
- spicy_level이 null인 후보는 매운맛 관련 요청이 없을 때만 후보로 사용합니다.

[특히 주의할 조건]
- "10분 안에", "빠르게", "간단한" 같은 표현은 cooking_time, difficulty, summary를 함께 보고 판단합니다.
- "국물", "찌개", "탕" 요청은 category, dish_type_tags, texture_tags, summary를 함께 보고 판단합니다.

[reason 생성 규칙]
- 길이 60~100자, 1~2문장.
- 친근한 존댓말로, 사용자에게 메뉴를 추천하듯 생기 있게 작성합니다. "~예요", "~좋아요", "~딱이에요" 같은 부드러운 어미를 쓰고, 문장 끝에 느낌표를 1개까지 자연스럽게 쓸 수 있습니다.
- 구조: 후보 속성 1~2개(재료·식감·조리 시간 등) + 사용자 요청/목적과의 연결.
- 후보 metadata에 명시된 사실만 근거로 사용합니다.
- 필드명을 직접 노출하지 않습니다.
  - cooking_time → 조리 시간
- 레시피 이름은 reason에서 반복하지 않습니다.
- name, category, image_url, manuals, nutrition 같은 사실 정보를 새로 생성하지 않습니다.
- metadata에 없는 조리법·조리 동사(예: "팬에 구워", "오븐에 구운", "튀겨낸")를 지어내지 않습니다. 조리 방식은 cooking_method 등 명시된 경우에만 언급합니다.
- 없는 효능/건강 효과/의학적 표현(예: "회복에 좋은", "소화가 잘되는", "면역에")을 생성하지 않습니다. 톤이 밝아져도 이 금지선은 그대로 지킵니다.
- "최고의", "완벽한", "가장 좋은", "정말 맛있는" 같은 과장·주관 평가어를 쓰지 않습니다.
- "많은 사람이 선호하는" 같은 일반화·통계 표현을 쓰지 않습니다.
- 여러 후보의 reason이 같은 문장 패턴으로 반복되지 않게, 후보마다 짚는 속성과 표현을 다르게 합니다.
- 자유 요청이 있으면, reason에 그 요청과의 구체적 연결을 1개 이상 포함합니다 (예: "닭고기" 요청 → "닭다리살을 사용해" 식으로).
- 단, 후보 metadata에 근거가 없으면 억지로 연결하지 않습니다 (없는 정보를 생성하지 않는다는 기존 규칙이 우선).

[reason 예시 — 톤 참고]
- 목적=든든하게 / 재료=닭고기,감자 / 조리 시간 25분
  → "닭고기와 감자가 듬뿍 들어가 한 그릇만으로도 든든해요. 25분이면 완성되니 출출할 때 부담 없이 즐기기 좋아요!"
- 목적=가볍게 / 재료=두부,채소 / 조리 시간 15분
  → "두부와 채소 위주라 속이 편안하게 가벼워요. 15분이면 뚝딱이라 부담 없이 한 끼 챙기기 좋아요!"
- 목적=맛있게 / 식감=쫄깃,매콤 / 재료=떡,고추장
  → "쫄깃한 떡에 매콤한 양념이 어우러져 입맛 확 살아나요. 든든하게 즐기고 싶을 때 잘 어울려요!"
- 나쁜 예 → "회복에 좋은 닭가슴살로 완벽한 한 끼!" (효능 "회복에 좋은" + 과장 "완벽한" = 금지)

[matched_intents 규칙]
- 항목 수는 1~4개, 권장 2~3개.
- 가능한 한 ALLOWED_INTENTS에서 선택합니다.
- 정확 매칭이 없을 때만 짧은 자유 태그를 허용합니다.
- 자유 태그는 한국어 1~2어절 또는 언더스코어 형식으로 작성합니다.
- 자유 태그는 matched_intents 전체의 절반 이하로 둡니다.
- reason과 의미적으로 연결되어야 합니다.
- matched_intents의 핵심 의도는 reason에도 자연스럽게 반영합니다.
- 후보 metadata에 없는 근거를 matched_intents로 만들지 않습니다.

[ALLOWED_INTENTS]
{", ".join(ALLOWED_INTENTS)}
"""


# ── 유저 프롬프트 빌더 ────────────────────────────────────────────────────────

def _build_constraint_block(structured_inputs: dict) -> str | None:
    """structured_inputs → [사용자 제약] 블록 텍스트. 표시할 라인 없으면 None."""
    lines: list[str] = []

    meal_times = structured_inputs.get("meal_times")
    if isinstance(meal_times, list) and meal_times:
        lines.append(f"- 시간대: {', '.join(meal_times)}")

    purpose_label = structured_inputs.get("purpose_label")
    purpose       = structured_inputs.get("purpose")
    if purpose_label:
        lines.append(f"- 목적: {purpose_label}")
    elif purpose:
        lines.append(f"- 목적: {purpose}")

    free_text = structured_inputs.get("free_text")
    if isinstance(free_text, str) and free_text.strip():
        lines.append(f"- 자유 요청: {free_text.strip()}")

    if not lines:
        return None
    return "[사용자 제약]\n" + "\n".join(lines)


def build_user_prompt(
    query:             str,
    candidates:        list[dict],
    structured_inputs: dict | None = None,
    top_k:             int         = 5,
) -> str:
    """질의 + 후보 + (선택) 정형 제약 → LLM user message 문자열.

    structured_inputs가 None / 빈 dict / 표시할 필드가 하나도 없으면
    [사용자 제약] 블록을 생략한다. top_k는 작업 지시문에 동적 주입되며,
    LLM이 정확히 그 개수만큼 추천하도록 유도한다.
    """
    safe_query      = query if query is not None else ""
    safe_candidates = candidates if candidates is not None else []

    candidates_json = json.dumps(
        safe_candidates,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    blocks: list[str] = []

    if structured_inputs:
        constraint_block = _build_constraint_block(structured_inputs)
        if constraint_block:
            blocks.append(constraint_block)

    blocks.append(f"# 사용자 질의\n{safe_query}")
    blocks.append(
        f"# 후보 Hybrid Search 상위 {len(safe_candidates)}건\n{candidates_json}"
    )
    blocks.append(
        "# 작업\n"
        f"위 후보 중 사용자 질의에 가장 적합한 정확히 {top_k}개를 선정하고,\n"
        "각 추천에 대해 reason과 matched_intents를 생성하시오.\n"
        f"조건에 부합하는 후보가 {top_k}개 미만이면 부합하는 만큼만 반환하고\n"
        "insufficient_matches를 true로 설정하시오.\n"
        f"조건에 부합하는 후보가 충분하면 반드시 정확히 {top_k}개를 반환하시오."
    )

    return "\n\n".join(blocks)


# ── 셀프 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy_candidates = [
        {
            "recipe_id": "28",
            "name": "새우 두부 계란찜",
            "category": "반찬",
            "cooking_method": "찌기",
            "summary": "부드럽고 담백한 한 끼",
            "main_ingredients": ["연두부", "새우", "달걀"],
            "meal_time": ["아침", "저녁"],
            "purpose": ["light", "protein"],
            "cooking_time": 20,
            "difficulty": "쉬움",
            "taste_tags": ["담백한"],
            "texture_tags": ["부드러운"],
            "recommended_situations": ["혼밥"],
            "dish_type_tags": ["찜"],
            "dense_rank": 3,
            "bm25_rank": 7,
            "rrf_score": 0.0321,
        },
        {
            "recipe_id": "104",
            "name": "북엇국",
            "category": "국",
            "cooking_method": "끓이기",
            "summary": "맑고 시원한 해장 국물",
            "main_ingredients": ["북어", "무", "달걀"],
            "meal_time": ["아침"],
            "purpose": ["light"],
            "cooking_time": 25,
            "difficulty": "쉬움",
            "taste_tags": ["담백한", "시원한"],
            "texture_tags": ["부드러운"],
            "recommended_situations": ["해장"],
            "dish_type_tags": ["국물요리"],
            "dense_rank": 1,
            "bm25_rank": 2,
            "rrf_score": 0.0492,
        },
    ]

    print("=" * 60)
    print("SYSTEM_PROMPT (첫 1000자)")
    print("=" * 60)
    print(SYSTEM_PROMPT[:1000])

    print()
    print("=" * 60)
    print("USER PROMPT 예시")
    print("=" * 60)
    user_prompt = build_user_prompt("안 매운 국물 요리", dummy_candidates)
    print(user_prompt)

    print()
    print("=" * 60)
    print("빈 후보 / 빈 쿼리 테스트")
    print("=" * 60)
    print(build_user_prompt("", []))

    print()
    print("=" * 60)
    print("Pydantic 스키마 검증")
    print("=" * 60)
    sample_response = RerankResponse(
        recommendations=[
            RerankItem(
                rank=1,
                recipe_id="104",
                reason="맑고 시원한 국물에 매운 정도가 거의 없어, 자극 없는 국물 요리를 찾는 요청에 잘 맞습니다.",
                matched_intents=["국물요리", "매운맛_낮음", "담백한맛"],
            )
        ],
        insufficient_matches=False,
    )
    print(sample_response.model_dump_json(indent=2))