"""rag.recipe_store.RecipeStore 단위 테스트."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from rag.recipe_store import RecipeStore


# ── fixtures ────────────────────────────────────────────────────────────────


DUMMY_RECIPES = [
    {"rcp_seq": "1", "name": "김치찌개", "summary": "매콤한 국물 요리"},
    {"rcp_seq": "2", "name": "된장찌개", "summary": "구수한 국물 요리"},
    {"rcp_seq": "42", "name": "비빔밥", "summary": "나물과 고추장의 조화"},
]


@pytest.fixture
def dummy_json(tmp_path: Path) -> Path:
    """더미 레시피 3개를 담은 임시 JSON 파일."""
    path = tmp_path / "recipes.json"
    path.write_text(json.dumps(DUMMY_RECIPES, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def store(dummy_json: Path) -> RecipeStore:
    return RecipeStore(dummy_json)


# ── 로드 ────────────────────────────────────────────────────────────────────


def test_load_success(store: RecipeStore):
    assert len(store) == 3


def test_load_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        RecipeStore(tmp_path / "nope.json")


def test_load_invalid_json(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        RecipeStore(bad)


def test_missing_rcp_seq_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    path = tmp_path / "partial.json"
    data = [
        {"rcp_seq": "1", "name": "김치찌개"},
        {"name": "rcp_seq 없는 항목"},
        {"rcp_seq": "2", "name": "된장찌개"},
    ]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="rag.recipe_store"):
        store = RecipeStore(path)

    assert len(store) == 2
    assert store.has_recipe("1")
    assert store.has_recipe("2")
    assert any("without rcp_seq" in rec.message for rec in caplog.records)


def test_duplicate_rcp_seq_overwrites(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    path = tmp_path / "dup.json"
    data = [
        {"rcp_seq": "1", "name": "원본"},
        {"rcp_seq": "1", "name": "덮어쓰기"},
    ]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="rag.recipe_store"):
        store = RecipeStore(path)

    assert len(store) == 1
    assert store.get_recipe_by_id("1")["name"] == "덮어쓰기"
    assert any("Duplicate rcp_seq" in rec.message for rec in caplog.records)


# ── 단건 조회 ───────────────────────────────────────────────────────────────


def test_get_recipe_by_id_raw(store: RecipeStore):
    hit = store.get_recipe_by_id("1")
    assert hit is not None
    assert hit["name"] == "김치찌개"


def test_get_recipe_by_id_with_prefix(store: RecipeStore):
    hit = store.get_recipe_by_id("recipe_1")
    assert hit is not None
    assert hit["name"] == "김치찌개"


def test_get_recipe_by_id_not_found(store: RecipeStore):
    assert store.get_recipe_by_id("999") is None


# ── 배치 조회 ───────────────────────────────────────────────────────────────


def test_get_recipes_by_ids_order_preserved(store: RecipeStore):
    result = store.get_recipes_by_ids(["recipe_42", "1", "2"])
    assert [r["name"] for r in result] == ["비빔밥", "김치찌개", "된장찌개"]


def test_get_recipes_by_ids_with_missing(store: RecipeStore, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="rag.recipe_store"):
        result = store.get_recipes_by_ids(["1", "999", "2"])

    assert len(result) == 2
    assert [r["name"] for r in result] == ["김치찌개", "된장찌개"]
    assert any("Missing recipe_ids" in rec.message for rec in caplog.records)


# ── 존재 확인 / 정규화 ──────────────────────────────────────────────────────


def test_has_recipe_true(store: RecipeStore):
    assert store.has_recipe("recipe_42") is True


def test_has_recipe_false(store: RecipeStore):
    assert store.has_recipe("999") is False


def test_normalize_recipe_id(store: RecipeStore):
    assert store.normalize_recipe_id("recipe_42") == "42"
    assert store.normalize_recipe_id("42") == "42"
