"""FastAPI 메인 앱. /healthz, /recommend, /ask 엔드포인트와 lifespan/예외 핸들러."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    Recommendation,
    RecommendRequest,
    RecommendResponse,
)
from rag.qa import answer as qa_answer
from rag.query_builder import build_retrieval_query
from rag.recipe_store import RecipeStore
from rag.reranker import rerank
from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── 헬퍼 ────────────────────────────────────────────────────────────────────


def normalize_recipe_id(recipe_id: str) -> str:
    """recipe_42 / 42 / 'recipe_42 ' 모두 '42'로 정규화."""
    return str(recipe_id).strip().removeprefix("recipe_").strip()


def get_image_url(recipe: dict) -> str | None:
    """JSON 원본에서 이미지 URL을 방어적으로 추출. 후보 키 순서대로 확인."""
    candidates = ["image_url", "img_main", "att_file_no_main", "thumbnail_url", "img_thumb"]
    for key in candidates:
        value = recipe.get(key)
        if value and isinstance(value, str) and value.strip():
            return value.strip()
    return None


# ── lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: initializing retrievers and recipe_store")

    dense = DenseRetriever()
    bm25 = BM25Retriever()
    app.state.retriever = HybridRetriever(dense, bm25)

    recipe_json_path = os.getenv("RECIPE_JSON_PATH", "data/recipes_enriched_v2.json")
    app.state.recipe_store = RecipeStore(recipe_json_path)

    logger.info(
        "Startup complete: recipe_store loaded %d recipes from %s",
        len(app.state.recipe_store),
        recipe_json_path,
    )

    yield

    logger.info("Shutdown")


# ── app ─────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="Meal-bot RAG API",
    description="레시피 추천 + 대화형 follow-up Q&A",
    version="0.1.0",
    lifespan=lifespan,
)


# ── 예외 핸들러 ─────────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """HTTPException은 FastAPI 기본 처리에 위임, 나머지는 500으로 마스킹."""
    if isinstance(exc, HTTPException):
        raise exc

    logger.exception(
        "Unhandled exception: path=%s method=%s error=%s",
        request.url.path,
        request.method,
        exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── 엔드포인트 ──────────────────────────────────────────────────────────────


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    return HealthResponse(status="ok")


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, request: Request):
    retriever = request.app.state.retriever
    recipe_store = request.app.state.recipe_store

    t0 = time.perf_counter()

    query = build_retrieval_query(
        meal_times=req.meal_times,
        purpose=req.purpose,
        spicy_max=req.spicy_max,
        free_text=req.free_text,
    )
    t1 = time.perf_counter()

    hits = retriever.search(query)
    t2 = time.perf_counter()

    candidates = [h.metadata for h in hits if h.metadata]
    structured_inputs = {
        "meal_times": req.meal_times,
        "purpose": req.purpose,
        "spicy_max": req.spicy_max,
        "free_text": req.free_text,
    }
    rerank_resp = await rerank(
        query=query,
        candidates=candidates,
        structured_inputs=structured_inputs,
        top_k=5,
    )
    t3 = time.perf_counter()

    recommendations: list[Recommendation] = []
    missing_ids: list[str] = []

    for item in rerank_resp.recommendations:
        normalized_id = normalize_recipe_id(item.recipe_id)
        recipe = recipe_store.get_recipe_by_id(normalized_id)

        if recipe is None:
            missing_ids.append(item.recipe_id)
            continue

        # kcal: nutrition dict 안에 있음
        kcal_value: float | None = None
        nutrition = recipe.get("nutrition")
        if isinstance(nutrition, dict):
            kcal_raw = nutrition.get("kcal")
            if kcal_raw is not None:
                try:
                    kcal_value = float(kcal_raw)
                except (TypeError, ValueError):
                    kcal_value = None

        cooking_time_value: int | None = None
        ct_raw = recipe.get("cooking_time")
        if ct_raw is not None:
            try:
                cooking_time_value = int(ct_raw)
            except (TypeError, ValueError):
                cooking_time_value = None

        recommendations.append(
            Recommendation(
                rank=item.rank,
                recipe_id=str(recipe.get("rcp_seq", normalized_id)),
                name=recipe.get("name", ""),
                image_url=get_image_url(recipe),
                summary=recipe.get("summary", ""),
                kcal=kcal_value,
                cooking_time=cooking_time_value,
                reason=item.reason,
                matched_intents=item.matched_intents,
            )
        )

    t4 = time.perf_counter()

    if missing_ids:
        logger.warning(
            "recommend: missing recipe_ids in store lookup: requested=%d, missing=%s",
            len(rerank_resp.recommendations),
            missing_ids,
        )

    insufficient = True if not recommendations else rerank_resp.insufficient_matches

    logger.info(str({
        "event": "recommend",
        "session_id": req.session_id,
        "turn_id": req.turn_id,
        "query_build_ms": int((t1 - t0) * 1000),
        "retrieval_ms": int((t2 - t1) * 1000),
        "rerank_ms": int((t3 - t2) * 1000),
        "lookup_ms": int((t4 - t3) * 1000),
        "total_ms": int((t4 - t0) * 1000),
        "hits_count": len(hits),
        "candidates_count": len(candidates),
        "recommendations_count": len(recommendations),
        "insufficient_matches": insufficient,
        "is_fallback": rerank_resp.is_fallback,
    }))

    return RecommendResponse(
        turn_id=req.turn_id,
        recommendations=recommendations,
        insufficient_matches=insufficient,
        is_fallback=rerank_resp.is_fallback,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    recipe_store = request.app.state.recipe_store

    t0 = time.perf_counter()

    normalized_id = normalize_recipe_id(req.recipe_id)
    recipe = recipe_store.get_recipe_by_id(normalized_id)
    t1 = time.perf_counter()

    if recipe is None:
        logger.warning(
            "ask: recipe not found: recipe_id=%s (normalized=%s)",
            req.recipe_id,
            normalized_id,
        )
        raise HTTPException(status_code=404, detail="Recipe not found")

    chat_history_dicts = [
        {"role": m.role, "content": m.content} for m in req.chat_history
    ]

    qa_resp = await qa_answer(
        query=req.question,
        retrieved_docs=[recipe],
        chat_history=chat_history_dicts,
    )
    t2 = time.perf_counter()

    logger.info(str({
        "event": "ask",
        "session_id": req.session_id,
        "turn_id": req.turn_id,
        "lookup_ms": int((t1 - t0) * 1000),
        "qa_ms": int((t2 - t1) * 1000),
        "total_ms": int((t2 - t0) * 1000),
        "refused": qa_resp.refused,
        "out_of_scope": qa_resp.out_of_scope,
        "qa_failed": qa_resp.qa_failed,
        "is_fallback": qa_resp.is_fallback,
    }))

    return AskResponse(
        turn_id=req.turn_id,
        answer=qa_resp.answer,
        used_fields=qa_resp.used_fields,
        refused=qa_resp.refused,
        out_of_scope=qa_resp.out_of_scope,
        qa_failed=qa_resp.qa_failed,
        is_fallback=qa_resp.is_fallback,
    )
