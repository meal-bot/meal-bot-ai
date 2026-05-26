"""rag/reranker.py의 _candidate_to_prompt_dict 방어 동작 단위 테스트.

retriever 계층에서 이미 list로 정규화하지만, 직접 호출 경로(테스트/디버그)에서
JSON string이나 비정상 값이 들어와도 추천 흐름이 깨지지 않도록 한 번 더 막는다.
"""

from rag.reranker import _candidate_to_prompt_dict


def _base_candidate(**overrides) -> dict:
    """공통 필드를 채운 후보 dict. overrides로 list 필드만 덮어쓰기."""
    base = {
        "recipe_id": "28",
        "name": "새우 두부 계란찜",
        "category": "반찬",
        "cooking_method": "찌기",
        "summary": "담백한 찜 요리",
        "difficulty": "보통",
        "main_ingredients": [],
        "meal_time": [],
        "purpose": [],
        "taste_tags": [],
        "texture_tags": [],
        "recommended_situations": [],
        "dish_type_tags": [],
        "cooking_time": 20,
    }
    base.update(overrides)
    return base


def test_prompt_dict_preserves_list_from_json_string():
    """JSON string list로 들어와도 prompt dict에서 list로 보존된다."""
    candidate = _base_candidate(
        taste_tags='["담백한", "고소한"]',
        main_ingredients='["연두부", "새우"]',
    )

    out = _candidate_to_prompt_dict(candidate)

    assert out["taste_tags"] == ["담백한", "고소한"]
    assert out["main_ingredients"] == ["연두부", "새우"]


def test_prompt_dict_preserves_native_list():
    """native list로 들어오면 그대로 보존된다."""
    candidate = _base_candidate(
        taste_tags=["담백한", "고소한"],
        texture_tags=["부드러운"],
    )

    out = _candidate_to_prompt_dict(candidate)

    assert out["taste_tags"] == ["담백한", "고소한"]
    assert out["texture_tags"] == ["부드러운"]


def test_prompt_dict_handles_malformed_json_string_without_crash():
    """malformed JSON string이 들어와도 []로 처리되고 함수가 정상 반환된다."""
    candidate = _base_candidate(
        taste_tags="[abc",            # parse 실패
        main_ingredients="담백한",     # 단일 str
        purpose=42,                    # 타입 자체가 비정상
    )

    # 흐름 실패 없이 정상 반환되어야 함
    out = _candidate_to_prompt_dict(candidate)

    assert out["taste_tags"] == []
    assert out["main_ingredients"] == []
    assert out["purpose"] == []
    # recipe_id 등 다른 필드는 정상
    assert out["recipe_id"] == "28"
    assert out["name"] == "새우 두부 계란찜"
