"""api.schemas Pydantic 스키마 단위 테스트."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import (
    AskRequest,
    AskResponse,
    ChatMessage,
    HealthResponse,
    Recommendation,
    RecommendRequest,
    RecommendResponse,
)


# ── Valid 케이스 ────────────────────────────────────────────────────────────


def test_recommend_request_valid():
    req = RecommendRequest(
        meal_times=["점심"],
        purpose="light",
        spicy_max=2,
        free_text="매콤한 거",
        session_id="s1",
        turn_id="t1",
    )
    assert req.meal_times == ["점심"]
    assert req.free_text == "매콤한 거"


def test_recommend_request_no_free_text():
    req = RecommendRequest(
        meal_times=["저녁"],
        purpose="protein",
        spicy_max=1,
        session_id="s1",
        turn_id="t1",
    )
    assert req.free_text is None


def test_ask_request_valid():
    req = AskRequest(
        question="이거 매워?",
        recipe_id="recipe_28",
        chat_history=[ChatMessage(role="user", content="안녕")],
        session_id="s1",
        turn_id="t2",
    )
    assert len(req.chat_history) == 1
    assert req.chat_history[0].role == "user"


def test_ask_request_empty_history():
    req = AskRequest(
        question="이거 매워?",
        recipe_id="recipe_28",
        session_id="s1",
        turn_id="t2",
    )
    assert req.chat_history == []


def test_recommendation_valid():
    rec = Recommendation(
        rank=1,
        recipe_id="recipe_28",
        name="새우 두부 계란찜",
        image_url="http://example.com/x.png",
        summary="담백한 찜",
        kcal=220.0,
        cooking_time=20,
        reason="단백질 챙기기 좋음",
        matched_intents=["protein", "light"],
    )
    assert rec.rank == 1
    assert rec.matched_intents == ["protein", "light"]


def test_recommend_response_empty():
    resp = RecommendResponse(
        turn_id="t1",
        recommendations=[],
        insufficient_matches=True,
        is_fallback=False,
    )
    assert resp.recommendations == []
    assert resp.insufficient_matches is True


def test_ask_response_fallback():
    resp = AskResponse(
        turn_id="t1",
        answer="잠시 후 다시 시도해 주세요.",
        used_fields=[],
        refused=False,
        qa_failed=True,
        is_fallback=True,
        out_of_scope=False,
    )
    assert resp.is_fallback is True
    assert resp.qa_failed is True


def test_ask_response_out_of_scope_required():
    """out_of_scope 미지정 시 ValidationError 발생 (required 필드)."""
    with pytest.raises(ValidationError):
        AskResponse(
            turn_id="t1",
            answer="재료는 연두부, 새우, 달걀입니다.",
            used_fields=[],
            refused=False,
            qa_failed=False,
            is_fallback=False,
        )


def test_health_response():
    resp = HealthResponse(status="ok")
    assert resp.status == "ok"


# ── Invalid 케이스 ──────────────────────────────────────────────────────────


def test_recommend_request_empty_meal_times():
    with pytest.raises(ValidationError):
        RecommendRequest(
            meal_times=[],
            purpose="light",
            spicy_max=2,
            session_id="s1",
            turn_id="t1",
        )


def test_recommend_request_invalid_meal_time():
    with pytest.raises(ValidationError):
        RecommendRequest(
            meal_times=["lunch"],
            purpose="light",
            spicy_max=2,
            session_id="s1",
            turn_id="t1",
        )


def test_recommend_request_invalid_purpose():
    with pytest.raises(ValidationError):
        RecommendRequest(
            meal_times=["점심"],
            purpose="quick",
            spicy_max=2,
            session_id="s1",
            turn_id="t1",
        )


def test_recommend_request_spicy_out_of_range():
    with pytest.raises(ValidationError):
        RecommendRequest(
            meal_times=["점심"],
            purpose="light",
            spicy_max=5,
            session_id="s1",
            turn_id="t1",
        )


def test_recommend_request_spicy_too_low():
    with pytest.raises(ValidationError):
        RecommendRequest(
            meal_times=["점심"],
            purpose="light",
            spicy_max=0,
            session_id="s1",
            turn_id="t1",
        )


def test_recommend_request_free_text_too_long():
    with pytest.raises(ValidationError):
        RecommendRequest(
            meal_times=["점심"],
            purpose="light",
            spicy_max=2,
            free_text="가" * 201,
            session_id="s1",
            turn_id="t1",
        )


def test_recommend_request_missing_session_id():
    with pytest.raises(ValidationError):
        RecommendRequest(
            meal_times=["점심"],
            purpose="light",
            spicy_max=2,
            turn_id="t1",
        )


def test_recommendation_rank_out_of_range():
    with pytest.raises(ValidationError):
        Recommendation(
            rank=6,
            recipe_id="recipe_28",
            name="x",
            summary="x",
            reason="x",
        )


def test_recommendation_rank_zero():
    with pytest.raises(ValidationError):
        Recommendation(
            rank=0,
            recipe_id="recipe_28",
            name="x",
            summary="x",
            reason="x",
        )


def test_ask_request_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(
            question="",
            recipe_id="recipe_28",
            session_id="s1",
            turn_id="t1",
        )


def test_ask_request_question_too_long():
    with pytest.raises(ValidationError):
        AskRequest(
            question="가" * 501,
            recipe_id="recipe_28",
            session_id="s1",
            turn_id="t1",
        )


def test_ask_request_chat_history_too_long():
    history = [ChatMessage(role="user", content="x") for _ in range(21)]
    with pytest.raises(ValidationError):
        AskRequest(
            question="이거 매워?",
            recipe_id="recipe_28",
            chat_history=history,
            session_id="s1",
            turn_id="t1",
        )


def test_chat_message_invalid_role():
    with pytest.raises(ValidationError):
        ChatMessage(role="system", content="x")


def test_chat_message_empty_content():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")


# ── Edge 케이스 ─────────────────────────────────────────────────────────────


def test_recommend_request_all_meal_times():
    req = RecommendRequest(
        meal_times=["아침", "점심", "저녁", "간식", "야식"],
        purpose="light",
        spicy_max=2,
        session_id="s1",
        turn_id="t1",
    )
    assert len(req.meal_times) == 5


def test_recommendation_optional_fields_null():
    rec = Recommendation(
        rank=1,
        recipe_id="recipe_28",
        name="x",
        image_url=None,
        summary="x",
        kcal=None,
        cooking_time=None,
        reason="x",
    )
    assert rec.image_url is None
    assert rec.kcal is None
    assert rec.cooking_time is None


def test_recommendation_kcal_float():
    rec = Recommendation(
        rank=1,
        recipe_id="recipe_28",
        name="x",
        summary="x",
        kcal=46.89,
        reason="x",
    )
    assert rec.kcal == 46.89
