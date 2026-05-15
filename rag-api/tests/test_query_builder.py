"""rag.query_builder.build_retrieval_query 단위 테스트."""

import pytest

from rag.query_builder import build_retrieval_query


# ── 정상 케이스 ──────────────────────────────────────────────────────────────

def test_single_meal_time():
    """단일 meal_time + free_text 없음."""
    result = build_retrieval_query(
        meal_times=["저녁"],
        purpose="light",
        spicy_max=1,
    )
    assert result == "시간대: 저녁. 목적: 가볍게. 매운맛: 안 매운."


def test_multiple_meal_times():
    """복수 meal_time은 공백으로 join."""
    result = build_retrieval_query(
        meal_times=["아침", "점심"],
        purpose="protein",
        spicy_max=2,
    )
    assert result == (
        "시간대: 아침 점심. 목적: 단백질 챙기기. 매운맛: 약간 매운 정도까지."
    )


def test_with_free_text():
    """free_text가 있으면 마지막 절로 붙음."""
    result = build_retrieval_query(
        meal_times=["저녁"],
        purpose="hearty",
        spicy_max=3,
        free_text="국물 요리 추천해줘",
    )
    assert result == (
        "시간대: 저녁. 목적: 든든하게. 매운맛: 보통 매운 정도까지. "
        "국물 요리 추천해줘"
    )


# ── free_text 엣지 케이스 ────────────────────────────────────────────────────

def test_free_text_none_omitted():
    """free_text=None이면 마지막 절(마침표 포함) 생략."""
    result = build_retrieval_query(
        meal_times=["점심"],
        purpose="tasty",
        spicy_max=4,
        free_text=None,
    )
    assert result == "시간대: 점심. 목적: 맛있게. 매운맛: 매운맛 가능."
    assert not result.endswith(" ")


def test_free_text_empty_string_omitted():
    """free_text=''(빈 문자열)도 생략."""
    result = build_retrieval_query(
        meal_times=["점심"],
        purpose="tasty",
        spicy_max=4,
        free_text="",
    )
    assert result == "시간대: 점심. 목적: 맛있게. 매운맛: 매운맛 가능."


def test_free_text_whitespace_only_omitted():
    """free_text가 공백/탭/개행만 있어도 생략."""
    result = build_retrieval_query(
        meal_times=["점심"],
        purpose="tasty",
        spicy_max=4,
        free_text="   \t\n  ",
    )
    assert result == "시간대: 점심. 목적: 맛있게. 매운맛: 매운맛 가능."


def test_free_text_strips_surrounding_whitespace():
    """free_text 앞뒤 공백은 strip해서 붙임."""
    result = build_retrieval_query(
        meal_times=["저녁"],
        purpose="light",
        spicy_max=1,
        free_text="  국물요리  ",
    )
    assert result.endswith("매운맛: 안 매운. 국물요리")


# ── 검증 실패 케이스 ────────────────────────────────────────────────────────

def test_empty_meal_times_raises():
    """meal_times가 빈 리스트면 ValueError."""
    with pytest.raises(ValueError, match="meal_times"):
        build_retrieval_query(
            meal_times=[],
            purpose="light",
            spicy_max=1,
        )


def test_invalid_purpose_raises():
    """정의되지 않은 purpose는 ValueError + 잘못된 값 명시."""
    with pytest.raises(ValueError, match="purpose"):
        build_retrieval_query(
            meal_times=["저녁"],
            purpose="diet",
            spicy_max=1,
        )


def test_invalid_spicy_max_raises():
    """spicy_max가 범위 밖(예: 0, 5)이면 ValueError."""
    with pytest.raises(ValueError, match="spicy_max"):
        build_retrieval_query(
            meal_times=["저녁"],
            purpose="light",
            spicy_max=5,
        )

    with pytest.raises(ValueError, match="spicy_max"):
        build_retrieval_query(
            meal_times=["저녁"],
            purpose="light",
            spicy_max=0,
        )