"""FastAPI 메인 앱 v0.3. /healthz, /chat 엔드포인트.

v0.2까지 있던 /recommend, /ask는 폐기되었고 /chat 단일 엔드포인트로 통합되었다.
ChatOrchestrator가 흐름 전체를 조정하며, main.py는 진입점 + DI만 담당.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.chat_orchestrator import ChatOrchestrator
from api.schemas import ChatRequest, ChatResponse, HealthResponse
from rag.recipe_store import RecipeStore
from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: initializing retrievers, recipe_store, chat_orchestrator")

    dense = DenseRetriever()
    bm25 = BM25Retriever()
    retriever = HybridRetriever(dense, bm25)

    recipe_json_path = os.getenv("RECIPE_JSON_PATH", "data/recipes_enriched_v2.json")
    recipe_store = RecipeStore(recipe_json_path)

    # ChatOrchestrator 1회 초기화. 향후 더 많은 의존성이 추가되어도 여기서 주입.
    chat_orchestrator = ChatOrchestrator(
        retriever=retriever,
        recipe_store=recipe_store,
    )

    app.state.retriever = retriever
    app.state.recipe_store = recipe_store
    app.state.chat_orchestrator = chat_orchestrator

    logger.info(
        "Startup complete: recipe_store loaded %d recipes from %s",
        len(recipe_store),
        recipe_json_path,
    )

    yield

    logger.info("Shutdown")


# ── app ─────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="Meal-bot RAG API",
    description="자유 채팅 기반 레시피 추천 + 대화형 follow-up Q&A (v0.3)",
    version="0.3.0",
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


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """v0.3 단일 엔드포인트. 흐름 처리는 ChatOrchestrator에 위임."""
    orchestrator: ChatOrchestrator = request.app.state.chat_orchestrator
    return await orchestrator.handle(req)