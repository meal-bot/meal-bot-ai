"""rag.reranker.rerank — is_fallback 플래그 동작 검증.

LLM 호출은 _call_llm을 mocking하여 결정론적으로 테스트.
JSONL 로깅은 부수 효과로 발생하지만 테스트 본질에 영향 없음.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from rag.rerank_prompt import RerankItem, RerankResponse
from rag.reranker import rerank


# ── fixtures ────────────────────────────────────────────────────────────────


def _make_candidate(rid: str, name: str = "더미") -> dict:
    """Hybrid 후보 dict (rerank가 요구하는 최소 필드만)."""
    return {
        "recipe_id": rid,
        "name": name,
        "category": "반찬",
        "cooking_method": "찌기",
        "summary": "테스트용 더미 요약",
        "main_ingredients": ["재료A", "재료B"],
        "meal_time": ["저녁"],
        "purpose": ["light"],
        "taste_tags": ["담백한"],
        "texture_tags": ["부드러운"],
        "recommended_situations": ["혼밥"],
        "dish_type_tags": ["반찬"],
        "cooking_time": 20,
        "spicy_level": 1,
        "difficulty": "쉬움",
        "dense_rank": 1,
        "bm25_rank": 2,
        "rrf_score": 0.0321,
    }


@pytest.fixture
def six_candidates() -> list[dict]:
    """LLM 호출 임계(RERANK_MIN_CANDIDATES=5) 이상의 후보 6개."""
    return [_make_candidate(str(i), f"레시피{i}") for i in range(1, 7)]


def _good_llm_response(candidate_ids: list[str], top_k: int = 5) -> RerankResponse:
    """검증을 통과하는 정상 LLM 응답."""
    items = [
        RerankItem(
            rank=i + 1,
            recipe_id=candidate_ids[i],
            reason="사용자 요청과 잘 맞는 후보입니다. 담백하고 부드러운 식감이 특징입니다.",
            matched_intents=["담백한맛", "부드러운식감"],
        )
        for i in range(top_k)
    ]
    return RerankResponse(
        recommendations=items,
        insufficient_matches=False,
    )


# ── 빈 후보 / 부족 후보 분기 ────────────────────────────────────────────────


def test_rerank_empty_candidates_not_fallback():
    """빈 후보는 'LLM 못 부른 게 아니라 그냥 결과 없음' → is_fallback=False."""
    resp = asyncio.run(rerank(query="안 매운 국물", candidates=[]))
    assert resp.recommendations == []
    assert resp.insufficient_matches is True
    assert resp.is_fallback is False


def test_rerank_insufficient_candidates_is_fallback_true():
    """후보가 RERANK_MIN_CANDIDATES 미만이면 LLM 스킵 → hybrid fallback."""
    candidates = [_make_candidate(str(i)) for i in range(1, 4)]  # 3개 < 5
    resp = asyncio.run(rerank(query="국물 요리", candidates=candidates))
    assert resp.is_fallback is True
    assert len(resp.recommendations) == 3
    assert all(it.reason for it in resp.recommendations)


# ── 정상 / 실패 / 검증실패 ──────────────────────────────────────────────────


def test_rerank_success_is_fallback_false(six_candidates):
    """LLM이 유효한 응답 반환 → is_fallback=False."""
    ids = [c["recipe_id"] for c in six_candidates]
    mock_resp = _good_llm_response(ids, top_k=5)

    with patch("rag.reranker._call_llm", new=AsyncMock(return_value=mock_resp)):
        resp = asyncio.run(rerank(query="담백한 반찬", candidates=six_candidates))

    assert resp.is_fallback is False
    assert resp.insufficient_matches is False
    assert len(resp.recommendations) == 5
    assert [it.recipe_id for it in resp.recommendations] == ids[:5]


def test_rerank_llm_failure_is_fallback_true(six_candidates):
    """LLM 호출이 매번 예외 → 재시도 한도까지 실패 후 hybrid fallback."""
    with patch(
        "rag.reranker._call_llm",
        new=AsyncMock(side_effect=RuntimeError("OpenAI down")),
    ):
        resp = asyncio.run(rerank(query="담백한 반찬", candidates=six_candidates))

    assert resp.is_fallback is True
    assert len(resp.recommendations) == 5
    # hybrid 순서 그대로 들어가야 함
    expected_ids = [c["recipe_id"] for c in six_candidates[:5]]
    assert [it.recipe_id for it in resp.recommendations] == expected_ids


def test_rerank_validation_failure_is_fallback_true(six_candidates):
    """LLM이 후보 밖 recipe_id 반환 → validation 실패 → fallback."""
    # 후보 밖 ID "999"를 응답에 섞어서 환각 케이스 흉내
    bad_resp = RerankResponse(
        recommendations=[
            RerankItem(
                rank=1,
                recipe_id="999",  # 후보 밖
                reason="환각된 응답입니다.",
                matched_intents=["담백한맛"],
            )
        ],
        insufficient_matches=False,
    )

    with patch("rag.reranker._call_llm", new=AsyncMock(return_value=bad_resp)):
        resp = asyncio.run(rerank(query="담백한 반찬", candidates=six_candidates))

    assert resp.is_fallback is True
    assert len(resp.recommendations) == 5
    expected_ids = [c["recipe_id"] for c in six_candidates[:5]]
    assert [it.recipe_id for it in resp.recommendations] == expected_ids
