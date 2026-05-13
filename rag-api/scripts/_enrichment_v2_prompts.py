"""
v2 보강용 프롬프트 정의 — 신규 5개 필드 전용.
- SYSTEM_PROMPT_V2: system message
- build_user_prompt_v2(recipe): 보강된 레시피 dict를 받아 user message 생성
"""

import json


SYSTEM_PROMPT_V2 = """당신은 한국 가정식 레시피에 메타데이터를 부여하는 전문가입니다.

이미 1차 보강된 레시피 정보를 받습니다. 거기에 아래 5개 신규 필드만 추가로 생성하세요.
기존 필드는 절대 수정하지 마세요. 응답에 기존 필드를 포함하지 마세요.

[필드 정의]

1. taste_tags (최대 3개)
   레시피의 맛 특성. enum에서만 선택.
   - 매콤한, 담백한, 고소한, 짭짤한, 새콤한
   - 달콤한, 얼큰한, 진한, 시원한, 깔끔한

2. texture_tags (최대 3개)
   주된 식감. enum에서만 선택.
   - 부드러운, 바삭한, 쫄깃한, 촉촉한, 아삭한
   - 폭신한, 꾸덕한, 국물있는

3. recommended_situations (최대 3개)
   이 음식이 잘 어울리는 상황. enum에서만 선택.
   - 혼밥, 손님상, 도시락, 야식, 해장, 술안주
   - 다이어트, 운동후, 비오는날, 추운날
   - 아이식, 어르신식, 가벼운한끼, 든든한한끼

4. dish_type_tags (최대 3개)
   음식의 유형(카테고리적 분류). 식사 시간대가 아님에 주의.
   - 메인요리, 반찬, 국물요리, 면요리, 밥요리
   - 간식, 디저트, 샐러드, 분식

5. difficulty (단일 값)
   조리 난이도. 셋 중 하나.
   - 쉬움, 보통, 어려움

[필드 의미 구분 — 매우 중요]
- category는 기존 필드입니다. 음식의 큰 분류입니다 (반찬, 국&찌개, 후식 등).
- meal_time은 기존 필드입니다. 어울리는 식사 시간대입니다 (아침, 점심, 저녁, 야식).
- dish_type_tags는 신규입니다. 음식의 유형/형태입니다 (메인요리, 국물요리, 면요리 등).
- recommended_situations는 신규입니다. 사용 상황입니다 (혼밥, 손님상, 다이어트 등).
이 네 필드는 서로 다른 차원의 정보이며 혼동하면 안 됩니다.

[원칙]
- enum에 없는 값은 절대 만들어내지 마세요.
- 각 list 필드는 최대 3개. 무리하게 많이 붙이지 마세요.
- 애매하면 빈 배열보다는 가장 근거 있는 1~2개만 선택.
- 명확한 근거가 없으면 빈 배열도 허용 (단 difficulty는 반드시 1개 선택).
- 추측이 아니라 주어진 레시피 정보 (name, summary, ingredients, manuals, cooking_method, cooking_time, spicy_level)에 근거하여 판단.

[난이도 판단 기준 예시]
- 쉬움: 재료 단순, 조리 단계 5단계 이내, 특수 기술/도구 불필요, cooking_time ≤ 15분
- 보통: 일반적 가정 요리, cooking_time 15~40분
- 어려움: 재료 많음(>10가지), 복잡한 기술(반죽/발효/오븐 등), cooking_time > 40분"""


def build_user_prompt_v2(recipe: dict) -> str:
    """LLM에 보낼 단일 레시피 정보. 기존 보강 데이터 포함."""
    context = {
        "name":              recipe.get("name"),
        "category":          recipe.get("category"),
        "cooking_method":    recipe.get("cooking_method"),
        "cooking_time":      recipe.get("cooking_time"),
        "spicy_level":       recipe.get("spicy_level"),
        "summary":           recipe.get("summary"),
        "main_ingredients":  recipe.get("main_ingredients"),
        "meal_time":         recipe.get("meal_time"),
        "purpose":           recipe.get("purpose"),
        "ingredients_clean": recipe.get("ingredients_clean"),
        "manuals":           recipe.get("manuals"),
    }
    context = {k: v for k, v in context.items() if v is not None}

    return (
        "다음 레시피에 신규 5개 필드만 생성해주세요. "
        "기존 필드는 응답에 포함하지 마세요.\n\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}"
    )