"""
LLM 보강을 위한 JSON Schema 정의.
OpenAI Structured Output (strict mode)에서 사용.
"""

RECIPE_ENRICHMENT_SCHEMA = {
    "name": "recipe_enrichment",
    "strict": True,
    "schema": {
        "type": "object",
        "required": [
            "category",
            "cooking_method",
            "meal_time",
            "purpose",
            "spicy_level",
            "summary",
            "question_1",
            "question_2",
            "question_3",
            "main_ingredients",
            "cooking_time",
        ],
        "additionalProperties": False,
        "properties": {
            # 음식 카테고리 (반찬/국&찌개/일품/후식/기타)
            "category": {
                "type": "string",
                "enum": ["반찬", "국&찌개", "일품", "후식", "기타"],
            },
            # 주요 조리 방법 1가지
            "cooking_method": {
                "type": "string",
                "enum": ["끓이기", "굽기", "볶기", "찌기", "튀기기", "부치기", "무치기", "조림", "삶기", "기타"],
            },
            # 적합한 식사 시간대 (1~3개)
            "meal_time": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["아침", "점심", "저녁", "간식", "야식"],
                },
                "minItems": 1,
                "maxItems": 3,
            },
            # 식사 목적 태그 (1~3개, 영문 소문자)
            "purpose": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["light", "protein", "hearty", "tasty"],
                },
                "minItems": 1,
                "maxItems": 3,
            },
            # 매운맛 단계 (1=안 매움, 4=매움)
            "spicy_level": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4,
            },
            # 메뉴 특징 요약 (10~100자 자연어)
            "summary": {
                "type": "string",
                "minLength": 10,
                "maxLength": 100,
            },
            # 사용자 검색 표현 1번
            "question_1": {
                "type": "string",
                "minLength": 10,
                "maxLength": 50,
            },
            # 사용자 검색 표현 2번
            "question_2": {
                "type": "string",
                "minLength": 10,
                "maxLength": 50,
            },
            # 사용자 검색 표현 3번
            "question_3": {
                "type": "string",
                "minLength": 10,
                "maxLength": 50,
            },
            # 주재료 목록 (양념류 제외, 1~5개)
            "main_ingredients": {
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 20,
                },
                "minItems": 1,
                "maxItems": 5,
            },
            # 조리 시간 (분 단위, 대기·숙성 시간 제외)
            "cooking_time": {
                "type": "integer",
                "minimum": 1,
                "maximum": 180,
            },
        },
    },
}
