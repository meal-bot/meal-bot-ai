"""api.main FastAPI 엔드포인트 테스트.

lifespan은 startup에서 실제 retriever/store를 만들어 ChromaDB까지 로딩하므로,
TestClient를 with 블록 없이 사용해 lifespan을 건너뛰고
fixture에서 app.state에 mock을 직접 주입한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from rag.qa_prompt import QAResponse
from rag.rerank_prompt import RerankItem, RerankResponse


# ── helpers ─────────────────────────────────────────────────────────────────


@dataclass
class FakeHit:
    """retriever.search 반환값으로 쓸 더미 Hit."""

    recipe_id: str
    metadata: dict


def make_hit(rid: str) -> FakeHit:
    return FakeHit(recipe_id=rid, metadata={"recipe_id": rid, "name": f"레시피{rid}"})


def make_recipe(rid: str, name: str | None = None) -> dict:
    """RecipeStore.get_recipe_by_id가 돌려줄 더미 JSON 원본."""
    return {
        "rcp_seq": rid,
        "name": name or f"레시피{rid}",
        "summary": "맛있는 한식 요리",
        "img_main": "http://example.com/main.png",
        "img_thumb": "http://example.com/thumb.png",
        "cooking_time": 20,
        "nutrition": {"energy_kcal": 220.0, "protein_g": 14.0},
    }


def _rerank_resp(ids: list[str], is_fallback: bool = False) -> RerankResponse:
    items = [
        RerankItem(
            rank=i + 1,
            recipe_id=rid,
            reason="사용자 요청과 잘 맞는 후보입니다. 담백한 풍미가 특징입니다.",
            matched_intents=["담백한맛"],
        )
        for i, rid in enumerate(ids)
    ]
    return RerankResponse(
        recommendations=items,
        insufficient_matches=len(items) < 5,
        is_fallback=is_fallback,
    )


def _qa_resp(
    answer: str = "재료는 연두부, 새우, 달걀입니다.",
    refused: bool = False,
    out_of_scope: bool = False,
    qa_failed: bool = False,
    is_fallback: bool = False,
) -> QAResponse:
    return QAResponse(
        answer=answer,
        used_fields=["main_ingredients"],
        refused=refused,
        out_of_scope=out_of_scope,
        qa_failed=qa_failed,
        is_fallback=is_fallback,
    )


VALID_RECOMMEND_BODY = {
    "meal_times": ["저녁"],
    "purpose": "light",
    "spicy_max": 2,
    "free_text": "담백한 거",
    "session_id": "s1",
    "turn_id": "t1",
}


VALID_ASK_BODY = {
    "question": "재료가 뭐야?",
    "recipe_id": "recipe_28",
    "chat_history": [],
    "session_id": "s1",
    "turn_id": "t2",
}


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """lifespan 우회용. with 블록 없이 TestClient만 생성."""
    app.state.retriever = MagicMock()
    app.state.recipe_store = MagicMock()
    return TestClient(app)


@pytest.fixture
def client_no_raise():
    """500 응답 검증용. server exception을 클라이언트에 전파."""
    app.state.retriever = MagicMock()
    app.state.recipe_store = MagicMock()
    return TestClient(app, raise_server_exceptions=False)


# ── /healthz ────────────────────────────────────────────────────────────────


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── /recommend ──────────────────────────────────────────────────────────────


def test_recommend_success(client):
    app.state.retriever.search.return_value = [make_hit(str(i)) for i in range(1, 6)]
    app.state.recipe_store.get_recipe_by_id.side_effect = lambda rid: make_recipe(rid)

    mock_resp = _rerank_resp([str(i) for i in range(1, 6)], is_fallback=False)
    mock_resp.insufficient_matches = False  # 5개 모두 채움
    with patch("api.main.rerank", new=AsyncMock(return_value=mock_resp)):
        r = client.post("/recommend", json=VALID_RECOMMEND_BODY)

    assert r.status_code == 200
    body = r.json()
    assert body["turn_id"] == "t1"
    assert len(body["recommendations"]) == 5
    assert body["insufficient_matches"] is False
    assert body["is_fallback"] is False
    first = body["recommendations"][0]
    assert first["rank"] == 1
    assert first["recipe_id"] == "1"
    assert first["image_url"] == "http://example.com/main.png"
    assert first["kcal"] == 220.0
    assert first["cooking_time"] == 20


def test_recommend_empty_hits(client):
    app.state.retriever.search.return_value = []
    mock_resp = RerankResponse(
        recommendations=[], insufficient_matches=True, is_fallback=False,
    )
    with patch("api.main.rerank", new=AsyncMock(return_value=mock_resp)):
        r = client.post("/recommend", json=VALID_RECOMMEND_BODY)

    assert r.status_code == 200
    body = r.json()
    assert body["recommendations"] == []
    assert body["insufficient_matches"] is True
    assert body["is_fallback"] is False


def test_recommend_fallback(client):
    app.state.retriever.search.return_value = [make_hit(str(i)) for i in range(1, 6)]
    app.state.recipe_store.get_recipe_by_id.side_effect = lambda rid: make_recipe(rid)

    mock_resp = _rerank_resp([str(i) for i in range(1, 6)], is_fallback=True)
    mock_resp.insufficient_matches = False
    with patch("api.main.rerank", new=AsyncMock(return_value=mock_resp)):
        r = client.post("/recommend", json=VALID_RECOMMEND_BODY)

    assert r.status_code == 200
    assert r.json()["is_fallback"] is True


def test_recommend_partial_missing(client):
    app.state.retriever.search.return_value = [make_hit(str(i)) for i in range(1, 4)]

    # 첫 번째는 None, 나머지 2개는 정상
    def lookup(rid: str):
        if rid == "1":
            return None
        return make_recipe(rid)

    app.state.recipe_store.get_recipe_by_id.side_effect = lookup

    mock_resp = _rerank_resp(["1", "2", "3"], is_fallback=False)
    with patch("api.main.rerank", new=AsyncMock(return_value=mock_resp)):
        r = client.post("/recommend", json=VALID_RECOMMEND_BODY)

    assert r.status_code == 200
    body = r.json()
    assert len(body["recommendations"]) == 2
    assert [rec["recipe_id"] for rec in body["recommendations"]] == ["2", "3"]


def test_recommend_all_missing_insufficient(client):
    app.state.retriever.search.return_value = [make_hit("1"), make_hit("2")]
    app.state.recipe_store.get_recipe_by_id.return_value = None

    mock_resp = _rerank_resp(["1", "2"], is_fallback=False)
    mock_resp.insufficient_matches = False  # rerank는 충분하다고 했지만 store에서 다 빠짐
    with patch("api.main.rerank", new=AsyncMock(return_value=mock_resp)):
        r = client.post("/recommend", json=VALID_RECOMMEND_BODY)

    assert r.status_code == 200
    body = r.json()
    assert body["recommendations"] == []
    assert body["insufficient_matches"] is True


def test_recommend_validation_error(client):
    bad = {**VALID_RECOMMEND_BODY, "meal_times": []}
    r = client.post("/recommend", json=bad)
    assert r.status_code == 422


def test_recommend_invalid_purpose(client):
    bad = {**VALID_RECOMMEND_BODY, "purpose": "invalid"}
    r = client.post("/recommend", json=bad)
    assert r.status_code == 422


# ── /ask ────────────────────────────────────────────────────────────────────


def test_ask_success(client):
    app.state.recipe_store.get_recipe_by_id.return_value = make_recipe("28")
    mock_qa = _qa_resp(answer="재료는 연두부와 새우, 달걀입니다.")
    with patch("api.main.qa_answer", new=AsyncMock(return_value=mock_qa)):
        r = client.post("/ask", json=VALID_ASK_BODY)

    assert r.status_code == 200
    body = r.json()
    assert body["turn_id"] == "t2"
    assert body["answer"] == "재료는 연두부와 새우, 달걀입니다."
    assert body["refused"] is False
    assert body["out_of_scope"] is False
    assert body["qa_failed"] is False
    assert body["is_fallback"] is False


def test_ask_recipe_not_found(client):
    app.state.recipe_store.get_recipe_by_id.return_value = None
    # qa_answer가 호출되면 안 됨
    with patch("api.main.qa_answer", new=AsyncMock()) as mock_qa:
        r = client.post("/ask", json=VALID_ASK_BODY)
        mock_qa.assert_not_called()

    assert r.status_code == 404
    assert r.json()["detail"] == "Recipe not found"


def test_ask_refused(client):
    app.state.recipe_store.get_recipe_by_id.return_value = make_recipe("28")
    mock_qa = _qa_resp(answer="해당 질문엔 답변드리기 어렵습니다.", refused=True)
    with patch("api.main.qa_answer", new=AsyncMock(return_value=mock_qa)):
        r = client.post("/ask", json=VALID_ASK_BODY)

    assert r.status_code == 200
    assert r.json()["refused"] is True


def test_ask_qa_failed(client):
    app.state.recipe_store.get_recipe_by_id.return_value = make_recipe("28")
    mock_qa = _qa_resp(answer="일시적 오류", qa_failed=True, is_fallback=True)
    with patch("api.main.qa_answer", new=AsyncMock(return_value=mock_qa)):
        r = client.post("/ask", json=VALID_ASK_BODY)

    assert r.status_code == 200
    body = r.json()
    assert body["qa_failed"] is True
    assert body["is_fallback"] is True


def test_ask_validation_error(client):
    bad = {**VALID_ASK_BODY, "question": ""}
    r = client.post("/ask", json=bad)
    assert r.status_code == 422


# ── 글로벌 예외 핸들러 ──────────────────────────────────────────────────────


def test_unhandled_exception(client_no_raise):
    app.state.retriever.search.side_effect = RuntimeError("ChromaDB down")
    r = client_no_raise.post("/recommend", json=VALID_RECOMMEND_BODY)
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}
