"""
v2 보강용 JSON Schema 정의 — 신규 5개 필드 전용.
OpenAI Structured Output (strict mode)에서 사용.
"""

TASTE_TAGS = [
    "매콤한", "담백한", "고소한", "짭짤한", "새콤한",
    "달콤한", "얼큰한", "진한", "시원한", "깔끔한",
]

TEXTURE_TAGS = [
    "부드러운", "바삭한", "쫄깃한", "촉촉한", "아삭한",
    "폭신한", "꾸덕한", "국물있는",
]

SITUATIONS = [
    "혼밥", "손님상", "도시락", "야식", "해장", "술안주",
    "다이어트", "운동후", "비오는날", "추운날",
    "아이식", "어르신식", "가벼운한끼", "든든한한끼",
]

DISH_TYPES = [
    "메인요리", "반찬", "국물요리", "면요리", "밥요리",
    "간식", "디저트", "샐러드", "분식",
]

DIFFICULTIES = ["쉬움", "보통", "어려움"]


SCHEMA_V2 = {
    "name": "recipe_enrichment_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "taste_tags",
            "texture_tags",
            "recommended_situations",
            "dish_type_tags",
            "difficulty",
        ],
        "properties": {
            "taste_tags": {
                "type": "array",
                "minItems": 0,
                "maxItems": 3,
                "items": {"type": "string", "enum": TASTE_TAGS},
            },
            "texture_tags": {
                "type": "array",
                "minItems": 0,
                "maxItems": 3,
                "items": {"type": "string", "enum": TEXTURE_TAGS},
            },
            "recommended_situations": {
                "type": "array",
                "minItems": 0,
                "maxItems": 3,
                "items": {"type": "string", "enum": SITUATIONS},
            },
            "dish_type_tags": {
                "type": "array",
                "minItems": 0,
                "maxItems": 3,
                "items": {"type": "string", "enum": DISH_TYPES},
            },
            "difficulty": {
                "type": "string",
                "enum": DIFFICULTIES,
            },
        },
    },
}