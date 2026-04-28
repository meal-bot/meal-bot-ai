"""
FastAPI entrypoint for the Meal-bot RAG API.

Run:
uvicorn app.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.answer_generator import generate_answer
from core.image_resolver import get_recipe_images
from core.reranker import rerank
from core.retrieval import Hit, search

SUPPORTED_MODES = {"v1", "v4-lite"}
DEFAULT_TOP_K = 5
V4_LITE_BASE_TOP_K = 10

app = FastAPI(title="Meal-bot RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    mode: str = "v4-lite"


def _hit_to_result(rank: int, hit: Hit, images_by_id: dict[int, dict]) -> dict:
    images = images_by_id.get(hit.recipe_id, {})
    return {
        "rank": rank,
        "recipe_id": hit.recipe_id,
        "name": hit.name,
        "category": hit.metadata.get("category") or "",
        "cooking_way": hit.metadata.get("cooking_way") or "",
        "score": hit.score,
        "image_url": images.get("image_url"),
        "thumbnail_url": images.get("thumbnail_url"),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend")
def recommend(request: RecommendRequest) -> dict:
    query = request.query.strip()
    mode = request.mode
    top_k = request.top_k

    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    if mode not in SUPPORTED_MODES:
        raise HTTPException(status_code=400, detail="unsupported mode")
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0")

    try:
        if mode == "v1":
            hits = search(query, top_k=top_k)
            answer = generate_answer(query, hits, top_k=top_k)
        else:
            base_hits = search(query, top_k=V4_LITE_BASE_TOP_K)
            reranked_hits = rerank(query, base_hits)
            hits = reranked_hits[:top_k]
            answer = generate_answer(query, reranked_hits, top_k=top_k)

        recipe_ids = [hit.recipe_id for hit in hits]
        images_by_id = get_recipe_images(recipe_ids)

        return {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "answer": answer,
            "results": [
                _hit_to_result(rank, hit, images_by_id)
                for rank, hit in enumerate(hits, start=1)
            ],
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="internal server error")
