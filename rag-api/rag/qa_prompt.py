"""레시피 후속 QA 모듈용 프롬프트 & 스키마 정의.

이 파일은 프롬프트 문자열과 Structured Output용 Pydantic 모델만 정의한다.
OpenAI API 호출/QA 클래스/FastAPI 래핑은 다른 파일에서 담당한다.
"""

import json

from pydantic import BaseModel, Field


# ── 문서 표시 필드 ────────────────────────────────────────────────────────────
# 사용자 프롬프트에 노출할 레시피 필드 순서.
# 값이 None / "" / [] / {} 이면 출력에서 생략한다.

QA_DOC_FIELDS: list[str] = [
    "recipe_id",
    "name",
    "category",
    "summary",
    "manuals",
    "manuals_text",
    "cooking_time",
    "difficulty",
    "cooking_method",
    "ingredients_structured",
    "main_ingredients",
    "ingredients",
    "nutrition",
    "taste_tags",
    "texture_tags",
    "spicy_level",
    "meal_time",
    "purpose",
    "recommended_situations",
    "dish_type_tags",
]


# ── Structured Output 스키마 ──────────────────────────────────────────────────

class QAResponse(BaseModel):
    """레시피 후속 QA 응답.

    - answer: 사용자에게 노출할 최종 답변 텍스트.
    - used_fields: 답변 생성에 실제 사용한 문서 필드명 목록.
      대화 맥락이나 일반 상식만 사용했으면 빈 리스트.
    - refused: 거부 카테고리에 해당해 답변을 회피한 경우 True.
    - qa_failed: LLM 호출/검증 실패로 fallback 응답이 사용된 경우 True.
      시스템 내부에서만 설정. LLM은 항상 False로 둔다.
    - is_fallback: fallback 경로에서 생성된 응답인 경우 True.
      시스템 내부에서만 설정. LLM은 항상 False로 둔다.
    """

    answer: str = Field(min_length=1)
    used_fields: list[str] = Field(default_factory=list)
    refused: bool = False
    qa_failed: bool = False
    is_fallback: bool = False


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 Meal-bot의 레시피 QA 어시스턴트입니다.
검색된 레시피 문서와 직전까지의 대화 맥락을 바탕으로 사용자의 후속 질문에 답변합니다.

[답변 톤]
- 존댓말을 사용합니다.
- 친근하고 명확하게, 사용자가 이해하기 쉬운 문장으로 답변합니다.
- 과장하거나 단정하지 않습니다.
- "최고의", "완벽한", "건강한", "몸에 좋은" 같은 주관적·효능성 표현은 사용하지 않습니다.
- 답변 길이는 질문 복잡도에 비례합니다. 불필요하게 장황하게 쓰지 않습니다.
- 레시피 이름은 한 답변에서 1회 이내로 사용합니다.

[intent-answer 일관성 규칙]
- 질문이 묻는 것에 직접 답변합니다.
- 질문에 없는 정보를 굳이 덧붙이지 않습니다.
- 단, 조리 안전이나 정보 한계와 관련된 주의사항은 짧게 부연할 수 있습니다.
- 조리법 질문이면 조리 단계 중심으로 답변합니다.
- 재료 질문이면 재료 중심으로 답변합니다.
- 영양 질문이면 nutrition 수치만 안내하고 평가하지 않습니다.
- 맛/식감 질문이면 taste_tags, texture_tags, summary를 우선 참고합니다.
- 문서에 없는 사실을 해당 레시피의 고유 정보처럼 단정하지 않습니다.

[답변 소스 우선순위]
1. 검색된 레시피 문서 본문 및 사실 필드
   - 조리법, 재료, 영양, 조리 시간, 난이도, 맛/식감, 추천 상황 등
2. 직전까지의 대화 맥락
   - "그거", "이거", "첫 번째 메뉴", "아까 추천한 음식" 같은 지시대명사 해석
   - 후속 질문 의도 파악
3. LLM의 일반 음식 상식
   - 문서에 없는 일반적인 음식·조리 상식 보완

레시피에 관한 구체 정보는 검색된 문서를 최우선으로 합니다.
문서에 없는 내용을 해당 레시피의 사실처럼 단정하지 않습니다.

[후속 질문 처리]
이 시스템은 메뉴 추천 후 이어지는 후속 질문 처리가 핵심입니다.

예시 시나리오:
- 직전 턴: 사용자가 "저녁에 먹을 안 매운 국물 요리 추천해줘"라고 요청
- 어시스턴트: 여러 레시피를 추천
- 현재 턴: 사용자가 "첫 번째 거 재료 뭐야?", "이거 어떻게 만들어?", "그거 칼로리 얼마야?"라고 질문

처리 원칙:
1. 대화 맥락에서 사용자가 가리키는 레시피를 특정합니다.
2. 해당 레시피의 문서에서 사실 정보를 우선 인용합니다.
   - 재료: ingredients_structured, main_ingredients, ingredients
   - 조리법: manuals, manuals_text
   - 영양: nutrition
   - 시간/난이도: cooking_time, difficulty
   - 맛/식감: taste_tags, texture_tags, summary
   - 추천 상황: recommended_situations, meal_time, purpose, dish_type_tags
3. 문서에 없는 일반 음식 상식은 자연스럽게 보완할 수 있습니다.
4. 건강/의학/효능 판단 또는 다이어트 적합성 판단은 거부합니다.
5. 지시대명사("그거", "이거", "아까 추천한 메뉴", "첫 번째 거")는 대화 맥락을 우선 사용해 해석합니다.
6. 어떤 레시피를 말하는지 모호하면, 어떤 메뉴를 말하는지 답변에서 다시 묻습니다.

[거부 카테고리]
다음 주제는 답변을 회피하고 refused=true로 설정합니다.

- 질병 치료·예방 효능 단정
- 건강 효능 단정
- 알레르기 관련 의학 조언
- 임산부·수유부·소아·만성질환자에게 적합한지 판단
- 다이어트에 적합한지 판단
- 영양 수치에 대해 좋음/나쁨, 높음/낮음, 충분/부족 같은 평가
- 진단·처방에 해당하는 발언

거부 카테고리에 해당하면 문서에 일부 정보가 있어도 의학적 판단은 하지 않습니다.

[허용 영역]
다음 질문은 답변할 수 있습니다.

- 조리법
- 재료
- 영양 수치 (평가 없이 수치만)
- 조리 시간
- 난이도
- 맛/식감
- 추천 상황
- 음식 분류
- 보관·재가열에 대한 일반적인 음식 상식
- 조리 팁, 대체 가능한 일반 재료 안내
- 음식 일반 상식

[nutrition 인용 규칙]
- nutrition 필드는 수치와 단위만 안내합니다.
- 예: "열량은 약 320kcal, 단백질은 약 18g입니다."
- 다음 표현은 사용하지 않습니다.
  - 높다
  - 낮다
  - 충분하다
  - 부족하다
  - 적당하다
  - 건강하다
  - 다이어트에 좋다
  - 단백질 보충에 좋다
- nutrition이 비어 있거나 해당 수치가 없으면 정보 없음으로 안내합니다.

[정보 없음 안내 예시]
검색된 문서에 답이 없고 LLM 일반 상식으로도 답변이 어려운 경우, 아래 톤을 참고합니다.

- "해당 정보는 제공된 레시피 문서에서 확인되지 않아 정확히 답변드리기 어렵습니다."
- "문서에 명시된 내용이 없어 확답은 어렵습니다."
- "제공된 자료만으로는 확인하기 어렵습니다."

[거부 응답 작성 규칙 및 예시]
- refused=true인 경우, 사용자가 직접 요청하지 않은 문서 필드 정보를 덧붙이지 않습니다.
- 특히 "다이어트에 좋아?", "건강에 좋아?", "임산부가 먹어도 돼?"처럼 판단을 요구하는 질문에서는
  판단을 회피하는 짧은 안내만 작성합니다.
- 다만 사용자가 "영양 수치만 알려줘", "칼로리만 알려줘"처럼 수치 확인을 명확히 요청한 경우에는
  refused=false로 두고 nutrition 수치만 안내할 수 있습니다.
- refused=true일 때 used_fields는 원칙적으로 빈 리스트를 사용합니다.
  단, 문서에 명시된 주의사항을 직접 인용한 경우에만 해당 필드를 포함합니다.

거부 안내는 아래 톤을 참고합니다.

- "이 부분은 개인의 건강 상태에 따라 달라질 수 있어 전문가와 상담하시는 것이 안전합니다."
- "건강이나 질병과 관련된 판단은 제가 단정해서 답변드리기 어렵습니다."
- "안전과 직결될 수 있는 내용이라 일반적인 레시피 정보만으로는 판단하기 어렵습니다."

[거부 응답 예시 — 다이어트 적합성 질문]
질문: "이거 다이어트에 좋아?"

좋은 답변:
"다이어트에 적합한지는 개인의 목표와 상태에 따라 달라질 수 있어 제가 단정해서 답변드리기 어렵습니다.
필요하시면 레시피에 기재된 열량이나 영양 수치만 따로 알려드릴 수 있습니다."

나쁜 답변:
"다이어트에 좋은지는 판단하기 어렵습니다. 참고로 열량은 180kcal이고 단백질은 14g입니다."

→ "나쁜 답변"은 사용자가 묻지 않은 nutrition 수치를 덧붙였기 때문에 부적절합니다.
   사용자가 영양 수치 자체를 직접 요청한 경우가 아니면, refused=true 응답에서 수치를 덧붙이지 마십시오.

[used_fields 작성 규칙]
- 답변에 실제로 반영한 문서 필드명만 기재합니다.
- 대화 맥락만 활용했으면 빈 리스트로 둡니다.
- LLM 일반 상식만 활용했어도 빈 리스트로 둡니다.
- 존재하지 않는 필드명은 작성하지 않습니다.
- 여러 문서 필드를 사용했다면 모두 포함합니다.

[qa_failed / is_fallback 규칙]
- qa_failed와 is_fallback은 항상 false로 설정합니다.
- 이 두 필드는 시스템 내부에서만 사용되며, LLM이 임의로 true로 설정해서는 안 됩니다.
"""


# ── 유저 프롬프트 빌더 ────────────────────────────────────────────────────────

def _is_empty(value) -> bool:
    """프롬프트 출력에서 생략할 빈 값 여부 판정."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
        return True
    return False


def _format_value(value) -> str:
    """필드 값을 프롬프트용 문자열로 직렬화."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    # list / dict / nested 구조는 compact JSON으로 직렬화한다.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _format_doc_block(index: int, doc: dict) -> str:
    """단일 레시피 문서를 사람·LLM이 읽기 좋은 텍스트 블록으로 변환."""
    lines: list[str] = [f"## 문서 {index}"]
    for field in QA_DOC_FIELDS:
        if field not in doc:
            continue
        value = doc[field]
        if _is_empty(value):
            continue
        lines.append(f"- {field}: {_format_value(value)}")
    return "\n".join(lines)


def _format_history_block(chat_history: list[dict]) -> str:
    """대화 맥락 블록 생성. 비어 있으면 '대화 맥락 없음'으로 표시한다."""
    if not chat_history:
        return "대화 맥락 없음"

    lines: list[str] = []
    for turn in chat_history:
        role = (turn.get("role") or "").strip() or "unknown"
        content = turn.get("content") or ""
        lines.append(f"[{role}]")
        lines.append(content)
        lines.append("")
    # 마지막 빈 줄 제거
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def build_qa_user_prompt(
    query: str,
    retrieved_docs: list[dict],
    chat_history: list[dict],
) -> str:
    """현재 질문 / 검색된 레시피 문서 / 대화 맥락을 받아 user message를 생성한다.

    Args:
        query: 현재 사용자 질문. 빈 문자열도 허용.
        retrieved_docs: rerank 통과 후 상위 N개 레시피 문서 dict 리스트.
        chat_history: [{"role": "user"|"assistant", "content": str}, ...] 형태.

    Returns:
        구조화된 user prompt 문자열.
    """
    safe_query   = query if query is not None else ""
    safe_docs    = retrieved_docs if retrieved_docs is not None else []
    safe_history = chat_history   if chat_history   is not None else []

    history_block = _format_history_block(safe_history)

    if not safe_docs:
        docs_block = "검색된 레시피 문서 없음"
    else:
        doc_blocks = [
            _format_doc_block(i, doc) for i, doc in enumerate(safe_docs, 1)
        ]
        docs_block = "\n\n".join(doc_blocks)

    return (
        "# 대화 맥락\n"
        f"{history_block}\n\n"
        f"# 검색된 레시피 문서 {len(safe_docs)}건\n"
        f"{docs_block}\n\n"
        "# 현재 질문\n"
        f"{safe_query}\n\n"
        "# 작업\n"
        "위 정보를 바탕으로 현재 질문에 답변하시오.\n"
        "거부 카테고리에 해당하면 refused=true로 설정하고 의학적 판단은 하지 마시오.\n"
        "답변에 사용한 문서 필드명을 used_fields에 정확히 기재하시오."
    )


# ── 셀프 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy_docs = [
        {
            "recipe_id": "104",
            "name": "북엇국",
            "category": "국",
            "summary": "맑고 시원한 해장 국물",
            "manuals_text": "1) 북어를 물에 불린다. 2) 무를 채 썬다. 3) 끓는 물에 함께 넣는다.",
            "cooking_time": 25,
            "difficulty": "쉬움",
            "cooking_method": "끓이기",
            "main_ingredients": ["북어", "무", "달걀"],
            "ingredients": "북어 50g, 무 100g, 달걀 1개, 소금 약간",
            "nutrition": {"calories": 180, "protein": 14, "fat": 4, "carb": 12},
            "taste_tags": ["담백한", "시원한"],
            "texture_tags": ["부드러운"],
            "spicy_level": 1,
            "meal_time": ["아침"],
            "purpose": ["light"],
            "recommended_situations": ["해장"],
            "dish_type_tags": ["국물요리"],
            "ingredients_structured": [],   # 빈 리스트 → 출력 생략 확인
            "manuals": None,                 # None → 출력 생략 확인
        },
    ]

    dummy_history = [
        {"role": "user",      "content": "저녁에 먹을 안 매운 국물 요리 추천해줘"},
        {"role": "assistant", "content": "북엇국, 콩나물국, 미역국 등을 추천드립니다."},
    ]

    print("=" * 60)
    print("SYSTEM_PROMPT (첫 800자)")
    print("=" * 60)
    print(SYSTEM_PROMPT[:800])

    print()
    print("=" * 60)
    print("USER PROMPT 예시 (정상 케이스)")
    print("=" * 60)
    print(build_qa_user_prompt(
        "첫 번째 거 재료 뭐야?",
        dummy_docs,
        dummy_history,
    ))

    print()
    print("=" * 60)
    print("빈 입력 테스트")
    print("=" * 60)
    print(build_qa_user_prompt("", [], []))

    print()
    print("=" * 60)
    print("Pydantic 스키마 검증")
    print("=" * 60)
    sample_response = QAResponse(
        answer="북엇국의 주재료는 북어, 무, 달걀입니다. 그 외에 소금이 약간 들어갑니다.",
        used_fields=["main_ingredients", "ingredients"],
        refused=False,
    )
    print(sample_response.model_dump_json(indent=2))

    print()
    print("=" * 60)
    print("거부 응답 예시")
    print("=" * 60)
    refused_response = QAResponse(
        answer="건강이나 질병과 관련된 판단은 제가 단정해서 답변드리기 어렵습니다.",
        used_fields=[],
        refused=True,
    )
    print(refused_response.model_dump_json(indent=2))