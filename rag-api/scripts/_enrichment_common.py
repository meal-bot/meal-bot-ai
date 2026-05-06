"""enrichment 공통 모듈: 프롬프트, 스키마, OpenAI 호출, 후처리, 검증, 샘플 선정."""

from __future__ import annotations

import json
import time

from openai import OpenAI


PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-mini":  {"input": 0.25, "output": 2.00},
}

ALLOWED_TASTE = {"담백한", "매콤한", "짭짤한", "새콤한", "단맛", "고소한", "진한", "깔끔한"}
ALLOWED_TEXTURE = {"부드러운", "바삭한", "쫄깃한", "촉촉한", "국물있는"}
ALLOWED_MEAL_TYPE = {"아침", "점심", "저녁", "야식"}
ALLOWED_SITUATIONS = {
    "야식", "안주", "손님상", "도시락", "든든한한끼", "가벼운한끼",
    "운동후", "다이어트", "저녁가족식사",
    "건강식", "속편한", "어르신식", "아이식",
}


SYSTEM_PROMPT = """당신은 한식 레시피 메타데이터를 추출하는 전문가입니다. 주어진 레시피 정보(이름, 카테고리, 조리법, 재료, 조리 단계)를 분석하여 RAG 검색 시스템에서 사용할 정형 필드와 자연어 태그를 생성합니다.

# 작업 원칙
1. 주어진 재료, 조리 단계, 영양정보를 최우선 근거로 판단합니다.
2. 단, 태그 선택과 난이도/매운맛/추천 상황 판단에는 일반적인 한식 조리 상식을 보조적으로 사용할 수 있습니다.
3. 근거가 부족하면 억지로 채우지 말고 null 또는 빈 배열을 반환합니다.
4. 모든 폐쇄형 태그는 제공된 후보 리스트 안에서만 선택합니다. 후보에 없는 값은 절대 만들지 마세요.
5. 태그 개수 제한을 엄격히 준수합니다. 애매하면 적게 선택하세요. 최대 개수를 반드시 채울 필요는 없습니다.

# 필드별 추론 가이드

## category (카테고리 보강)
- 입력 category가 null인 경우에만 추론
- 후보: "밥", "반찬", "국&찌개", "후식", "일품"

## cooking_method (조리법 보강)
- 입력 cooking_method가 "기타"이거나 null인 경우에만 추론
- 후보: "끓이기", "볶기", "굽기", "찌기", "튀기기", "삶기", "무치기", "조리기", "데치기", "부치기"

## cooking_time (조리 시간, 분 단위 정수)
- manuals 단계 수 + 단계 복잡도 기반 추정
- 5단계 이하 → 15~20분 / 6~10단계 → 25~40분 / 11단계+ → 45~60분+
- 절임/숙성/우려내기 등 대기 시간은 제외

## difficulty (난이도, 1~3) — 단계 복잡도와 실패 가능성 기준
- 1: 초보. 재료 적고 손질 단순, 한 가지 조리법
- 2: 보통. 양념장+재료 손질+가열 조합. 일반적 한식 대부분
- 3: 고급. 육수/속채우기/튀김옷/말기 등 실패 가능성 큼

## spicy_level (매운맛, 1~5)
- 1: 안 매움. 고추류 전혀 없음
- 2: 거의 안 매움. 후추, 풋고추 등 향/장식 수준
- 3: 약간 매움. 고춧가루/고추장/청양고추 들어가지만 보조적
- 4: 매움. 고추장/고춧가루/청양고추가 양념의 핵심
- 5: 매우 매움. 매운맛 자체가 핵심 ("불", "매운", "화끈한" 등 강한 표현)

## meal_type_tags (최대 2개)
- 후보: ["아침", "점심", "저녁", "야식"]
- 메인요리/일품/국&찌개: ["점심", "저녁"] 가능
- 반찬류: ["저녁"] 또는 ["점심", "저녁"]
- 죽/계란찜/부드러운 메뉴: ["아침"] 포함 가능
- 안주성/간식성: ["야식"] 포함 가능
- 확신 낮으면 1개만

## summary
- 40~120자 권장. 메인 재료+조리법+특징을 담은 1~2문장 자연어 요약

## taste_tags (최대 4개)
- 후보: ["담백한", "매콤한", "짭짤한", "새콤한", "단맛", "고소한", "진한", "깔끔한"]

## texture_tags (최대 3개)
- 후보: ["부드러운", "바삭한", "쫄깃한", "촉촉한", "국물있는"]

## recommended_situations (최대 4개)
- 후보: ["야식", "안주", "손님상", "도시락", "든든한한끼", "가벼운한끼", "운동후", "다이어트", "저녁가족식사", "건강식", "속편한", "어르신식", "아이식"]
- "운동후": 단백질 충분 + 자극적/기름지지 않은 메뉴만
- "다이어트": 칼로리 낮고 기름기/탄수화물 적음
- "어르신식"/"아이식": 자극적이지 않고 부드러운 메뉴만
- 1~4개. 적절하면 3개만 둬도 OK

## main_ingredients (최대 7개)
- 양념(소금, 간장, 고춧가루, 식초, 설탕, 다진마늘, 다진파, 후추 등) 제외
- 정상 레시피 3~7개. 재료 결손 시 빈 배열

# 출력 형식
JSON 객체로 반환합니다. 모든 필드를 포함해야 합니다."""


USER_PROMPT_TEMPLATE = """다음 레시피의 메타데이터를 추출하세요.

# 레시피 정보
- 이름: {name}
- 카테고리: {category_display}
- 조리법: {cooking_method_display}
- 영양정보: 칼로리 {calories}kcal, 단백질 {protein}g, 지방 {fat}g, 나트륨 {sodium}mg, 탄수화물 {carbs}g
- 재료: {ingredients_clean}
- 조리 단계:
{manuals_text}

# 보강 필요 여부
- category 보강 필요: {needs_category}
- cooking_method 보강 필요: {needs_cooking_method}"""


ENRICH_SCHEMA = {
    "name": "enrich_recipe",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "category": {
                "type": ["string", "null"],
                "enum": ["밥", "반찬", "국&찌개", "후식", "일품", None],
            },
            "cooking_method": {
                "type": ["string", "null"],
                "enum": ["끓이기", "볶기", "굽기", "찌기", "튀기기", "삶기",
                         "무치기", "조리기", "데치기", "부치기", None],
            },
            "cooking_time": {"type": ["integer", "null"]},
            "difficulty": {"type": "integer", "enum": [1, 2, 3]},
            "spicy_level": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "meal_type_tags": {
                "type": "array",
                "items": {"type": "string", "enum": ["아침", "점심", "저녁", "야식"]},
            },
            "summary": {"type": "string"},
            "taste_tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["담백한", "매콤한", "짭짤한", "새콤한",
                            "단맛", "고소한", "진한", "깔끔한"],
                },
            },
            "texture_tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["부드러운", "바삭한", "쫄깃한", "촉촉한", "국물있는"],
                },
            },
            "recommended_situations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["야식", "안주", "손님상", "도시락",
                             "든든한한끼", "가벼운한끼", "운동후", "다이어트",
                             "저녁가족식사", "건강식", "속편한", "어르신식", "아이식"],
                },
            },
            "main_ingredients": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "category", "cooking_method", "cooking_time", "difficulty",
            "spicy_level", "meal_type_tags", "summary", "taste_tags",
            "texture_tags", "recommended_situations", "main_ingredients",
        ],
    },
}


def select_base_samples(recipes: list[dict], target: int = 5) -> list[dict]:
    """우선순위 기반 기본 샘플 선정 (compare/enrich 양쪽 공통)."""
    selected: list[dict] = []
    selected_seqs: set = set()

    def add_if_new(recipe):
        if recipe and recipe["rcp_seq"] not in selected_seqs:
            selected.append(recipe)
            selected_seqs.add(recipe["rcp_seq"])
            return True
        return False

    # 1. rcp_seq=28
    target_28 = next((r for r in recipes if r.get("rcp_seq") == "28"), None)
    add_if_new(target_28)

    # 2. 매운 메뉴
    SPICY_KEYWORDS = ["고추장", "고춧가루", "청양고추", "매운", "매콤"]
    for r in recipes:
        if r["rcp_seq"] in selected_seqs:
            continue
        haystack = (r.get("name") or "") + " " + (r.get("ingredients_clean") or "")
        if any(kw in haystack for kw in SPICY_KEYWORDS):
            if add_if_new(r):
                break

    # 3. 국&찌개
    soup = next((r for r in recipes
                 if r.get("category") == "국&찌개" and r["rcp_seq"] not in selected_seqs), None)
    add_if_new(soup)

    # 4. category=None
    no_category = next((r for r in recipes
                        if r.get("category") is None and r["rcp_seq"] not in selected_seqs), None)
    add_if_new(no_category)

    # 5. cooking_way="기타" or None  (데이터 키는 cooking_way)
    other_method = next((r for r in recipes
                         if r.get("cooking_way") in ("기타", None)
                         and r["rcp_seq"] not in selected_seqs), None)
    add_if_new(other_method)

    # 부족분 보충: category 다양성
    if len(selected) < target:
        seen_categories = {r.get("category") for r in selected}
        for r in recipes:
            if len(selected) >= target:
                break
            if r["rcp_seq"] in selected_seqs:
                continue
            if r.get("category") not in seen_categories:
                add_if_new(r)
                seen_categories.add(r.get("category"))

    # 그래도 부족하면 앞에서 채움
    if len(selected) < target:
        for r in recipes:
            if len(selected) >= target:
                break
            add_if_new(r)

    return selected[:target]


def build_manuals_text(manuals: list) -> str:
    if not manuals:
        return ""
    sorted_steps = sorted(manuals, key=lambda x: x.get("step", 0))
    return "\n".join(
        f"[{s['step']}] {s.get('desc', '').strip()}"
        for s in sorted_steps
        if s.get("desc")
    )


def build_user_prompt(recipe: dict) -> tuple[str, bool, bool]:
    """user prompt 생성 + 보강 필요 플래그 반환. (데이터 키: cooking_way)"""
    original_category = recipe.get("category")
    original_cooking_method = recipe.get("cooking_way")  # read는 cooking_way

    needs_category = original_category is None
    needs_cooking_method = original_cooking_method in (None, "기타")

    if original_category is None:
        category_display = "(없음, 보강 필요)"
    else:
        category_display = original_category

    if original_cooking_method is None:
        cooking_method_display = "(없음, 보강 필요)"
    elif original_cooking_method == "기타":
        cooking_method_display = "기타 (보강 필요)"
    else:
        cooking_method_display = original_cooking_method

    nutrition = recipe.get("nutrition", {}) or {}
    calories = nutrition.get("energy_kcal", recipe.get("calories"))
    protein = nutrition.get("protein_g", recipe.get("protein"))
    fat = nutrition.get("fat_g", recipe.get("fat"))
    sodium = nutrition.get("sodium_mg", recipe.get("sodium"))
    carbs = nutrition.get("carbs_g", recipe.get("carbs"))

    manuals_text = build_manuals_text(recipe.get("manuals", []))

    prompt = USER_PROMPT_TEMPLATE.format(
        name=recipe.get("name", ""),
        category_display=category_display,
        cooking_method_display=cooking_method_display,
        calories=calories,
        protein=protein,
        fat=fat,
        sodium=sodium,
        carbs=carbs,
        ingredients_clean=recipe.get("ingredients_clean", ""),
        manuals_text=manuals_text,
        needs_category=needs_category,
        needs_cooking_method=needs_cooking_method,
    )
    return prompt, needs_category, needs_cooking_method


def enforce_limits(enriched: dict) -> dict:
    """maxItems 강제 + enum 외 값 제거. 원본 우선 정책은 별도."""
    enriched["meal_type_tags"] = (enriched.get("meal_type_tags") or [])[:2]
    enriched["taste_tags"] = (enriched.get("taste_tags") or [])[:4]
    enriched["texture_tags"] = (enriched.get("texture_tags") or [])[:3]
    enriched["recommended_situations"] = (enriched.get("recommended_situations") or [])[:4]
    enriched["main_ingredients"] = (enriched.get("main_ingredients") or [])[:7]

    enriched["meal_type_tags"] = [v for v in enriched["meal_type_tags"] if v in ALLOWED_MEAL_TYPE]
    enriched["taste_tags"] = [v for v in enriched["taste_tags"] if v in ALLOWED_TASTE]
    enriched["texture_tags"] = [v for v in enriched["texture_tags"] if v in ALLOWED_TEXTURE]
    enriched["recommended_situations"] = [
        v for v in enriched["recommended_situations"] if v in ALLOWED_SITUATIONS
    ]
    return enriched


def post_process(enriched: dict, recipe: dict,
                 needs_category: bool, needs_cooking_method: bool,
                 model: str) -> dict:
    """후처리: enforce_limits + 원본 우선 + flags 부착."""
    enriched = enforce_limits(enriched)

    original_category = recipe.get("category")
    original_cooking_method = recipe.get("cooking_way")  # read는 cooking_way

    if not needs_category:
        enriched["category"] = original_category
    if not needs_cooking_method:
        enriched["cooking_method"] = original_cooking_method  # write는 cooking_method

    enriched["_flags"] = {
        "category_ai_filled": needs_category and enriched.get("category") is not None,
        "cooking_method_ai_filled": needs_cooking_method and enriched.get("cooking_method") is not None,
        "enrichment_completed": True,
        "enrichment_failed": False,
        "enrichment_model": model,
    }
    return enriched


def validate_enriched(enriched: dict) -> list[str]:
    """enriched 결과 검증. 위반 사항 리스트 반환 (빈 리스트면 통과)."""
    issues = []

    # 1. enum 외 값 검증
    bad = [v for v in enriched.get("meal_type_tags", []) if v not in ALLOWED_MEAL_TYPE]
    if bad:
        issues.append(f"meal_type_tags enum 외 값: {bad}")
    bad = [v for v in enriched.get("taste_tags", []) if v not in ALLOWED_TASTE]
    if bad:
        issues.append(f"taste_tags enum 외 값: {bad}")
    bad = [v for v in enriched.get("texture_tags", []) if v not in ALLOWED_TEXTURE]
    if bad:
        issues.append(f"texture_tags enum 외 값: {bad}")
    bad = [v for v in enriched.get("recommended_situations", []) if v not in ALLOWED_SITUATIONS]
    if bad:
        issues.append(f"recommended_situations enum 외 값: {bad}")

    # 2. maxItems 위반 검증
    if len(enriched.get("meal_type_tags", [])) > 2:
        issues.append(f"meal_type_tags 초과: {len(enriched['meal_type_tags'])}개")
    if len(enriched.get("taste_tags", [])) > 4:
        issues.append(f"taste_tags 초과: {len(enriched['taste_tags'])}개")
    if len(enriched.get("texture_tags", [])) > 3:
        issues.append(f"texture_tags 초과: {len(enriched['texture_tags'])}개")
    if len(enriched.get("recommended_situations", [])) > 4:
        issues.append(f"recommended_situations 초과: {len(enriched['recommended_situations'])}개")
    if len(enriched.get("main_ingredients", [])) > 7:
        issues.append(f"main_ingredients 초과: {len(enriched['main_ingredients'])}개")

    # 3. summary 길이 검증 (20자 미만 또는 150자 초과)
    summary_len = len(enriched.get("summary") or "")
    if summary_len < 20 or summary_len > 150:
        issues.append(f"summary 길이 위반: {summary_len}자")

    # 4. cooking_time 범위 검증 (null 허용, 값 있으면 5~180)
    ct = enriched.get("cooking_time")
    if ct is not None and (ct < 5 or ct > 180):
        issues.append(f"cooking_time 범위 위반: {ct}분")

    # 5. difficulty 검증
    diff = enriched.get("difficulty")
    if diff not in (1, 2, 3):
        issues.append(f"difficulty 값 위반: {diff}")

    # 6. spicy_level 검증
    spicy = enriched.get("spicy_level")
    if spicy not in (1, 2, 3, 4, 5):
        issues.append(f"spicy_level 값 위반: {spicy}")

    return issues


def enrich_one(client: OpenAI, recipe: dict, model: str) -> dict:
    """단일 레시피 enrichment. 호출 + 파싱 + post_process + _meta 부착."""
    user_prompt, needs_category, needs_cooking_method = build_user_prompt(recipe)

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": ENRICH_SCHEMA,
        },
    )
    elapsed = time.time() - start

    content = response.choices[0].message.content
    enriched = json.loads(content)
    enriched = post_process(enriched, recipe, needs_category, needs_cooking_method, model)

    enriched["_meta"] = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "elapsed_sec": round(elapsed, 2),
    }
    return enriched
