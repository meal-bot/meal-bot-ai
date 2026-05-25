"""rag/retriever.py의 LIST_METADATA_FIELDS / _normalize_metadata 단위 테스트.

ChromaDB는 list 성격 메타데이터를 JSON string으로 반환하는 경우가 있어,
Hit.metadata에 담기 전 한 곳에서 list로 정규화한다. 이 테스트는 그 정규화
계층의 입력 케이스 매트릭스를 고정한다.
"""

import pytest

from rag.retriever import (
    LIST_METADATA_FIELDS,
    _normalize_list_metadata_value,
    _normalize_metadata,
)


# ── 단일 값 정규화 케이스 ────────────────────────────────────────────────────

def test_json_string_list_parsed_to_list():
    """JSON string list가 list[str]로 변환된다."""
    result = _normalize_list_metadata_value('["담백한", "고소한"]')
    assert result == ["담백한", "고소한"]


def test_native_list_is_idempotent():
    """native list는 그대로 유지된다."""
    original = ["담백한", "고소한"]
    result = _normalize_list_metadata_value(original)
    assert result == ["담백한", "고소한"]


def test_empty_json_array_string_becomes_empty_list():
    """'[]' (빈 JSON 배열)은 []가 된다."""
    assert _normalize_list_metadata_value("[]") == []


def test_none_becomes_empty_list():
    """None은 []가 된다."""
    assert _normalize_list_metadata_value(None) == []


def test_malformed_json_string_becomes_empty_list():
    """JSON 파싱 실패는 [] (정보 손실보다 일관성 우선)."""
    assert _normalize_list_metadata_value("[abc") == []
    assert _normalize_list_metadata_value("[1, 2,") == []


def test_non_string_items_in_list_are_filtered():
    """JSON list 내부 비-string 값은 제거된다."""
    assert _normalize_list_metadata_value(["담백한", 1, None, "고소한"]) == [
        "담백한", "고소한"
    ]
    # JSON string 경로에서도 동일
    assert _normalize_list_metadata_value('["담백한", 1, null]') == ["담백한"]


def test_single_non_json_string_becomes_empty_list():
    """JSON list 형태가 아닌 단일 문자열은 []. 정보 손실 감수."""
    assert _normalize_list_metadata_value("담백한") == []


# ── 모듈 상수 ────────────────────────────────────────────────────────────────

def test_list_metadata_fields_exact_set():
    """LIST_METADATA_FIELDS는 정확히 7개. 합의된 화이트리스트가 흔들리지 않게 고정."""
    expected = {
        "main_ingredients", "meal_time", "purpose",
        "taste_tags", "texture_tags",
        "recommended_situations", "dish_type_tags",
    }
    assert set(LIST_METADATA_FIELDS) == expected
    assert len(LIST_METADATA_FIELDS) == 7


# ── dict 단위 정규화 ─────────────────────────────────────────────────────────

def test_normalize_metadata_does_not_mutate_original():
    """원본 dict는 mutate되지 않는다 (caller가 raw chroma 결과를 그대로 보관할 수 있도록)."""
    original = {
        "recipe_id": "28",
        "name": "새우 두부 계란찜",
        "taste_tags": '["담백한"]',
        "main_ingredients": '["연두부", "새우"]',
    }
    snapshot = dict(original)

    _ = _normalize_metadata(original)

    assert original == snapshot, "원본 dict가 변경되었음"


def test_non_whitelisted_fields_passed_through():
    """LIST_METADATA_FIELDS 외 필드는 정규화하지 않고 그대로 유지."""
    original = {
        "recipe_id": "28",
        "name": "새우 두부 계란찜",
        "summary": "담백한 찜 요리",
        "cooking_time": 20,
        "spicy_level": 1,
        "taste_tags": '["담백한"]',  # 화이트리스트 → 정규화 됨
    }
    result = _normalize_metadata(original)

    # 화이트리스트 외 필드는 원본 그대로
    assert result["recipe_id"] == "28"
    assert result["name"] == "새우 두부 계란찜"
    assert result["summary"] == "담백한 찜 요리"
    assert result["cooking_time"] == 20
    assert result["spicy_level"] == 1
    # 화이트리스트 필드는 list로 정규화
    assert result["taste_tags"] == ["담백한"]
