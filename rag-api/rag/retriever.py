"""Dense + BM25 retriever."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import (
    CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL, NORMALIZE_EMBEDDINGS,
    BM25_INDEX_PATH,
    DENSE_TOP_K, BM25_TOP_K, RRF_TOP_K, RRF_K,
)
from rag.tokenizer import KiwiTokenizer


@dataclass
class Hit:
    recipe_id:   str
    rank:        int
    score:       float
    source:      str
    name:        str | None          = None
    metadata:    dict | None         = None
    dense_rank:  int | None          = None
    bm25_rank:   int | None          = None
    dense_score: float | None        = None
    bm25_score:  float | None        = None


class DenseRetriever:
    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_collection(name=COLLECTION_NAME)
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        query_emb = self.model.encode(
            query,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        hits: list[Hit] = []
        for i, (doc_id, dist, meta) in enumerate(zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        )):
            rank = i + 1
            # Chroma cosine distance. 낮을수록 가까움 (0~2 범위).
            score = dist
            hits.append(Hit(
                recipe_id   = meta["recipe_id"],
                rank        = rank,
                score       = score,
                source      = "dense",
                name        = meta.get("name"),
                metadata    = meta,
                dense_rank  = rank,
                dense_score = score,
            ))

        return hits


class BM25Retriever:
    """BM25 키워드 기반 검색."""

    def __init__(
        self,
        index_path: str = BM25_INDEX_PATH,
        data_path: str = "data/recipes_enriched_v2.json",
    ) -> None:
        index_dir = Path(index_path)

        with open(index_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(index_dir / "recipe_ids.pkl", "rb") as f:
            self.recipe_ids: list[str] = pickle.load(f)
        with open(index_dir / "tokenized_corpus.pkl", "rb") as f:
            self.tokenized_corpus = pickle.load(f)

        self.tokenizer = KiwiTokenizer()

        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        self.recipe_metadata: dict[str, dict] = {
            str(r["rcp_seq"]): r for r in data
        }

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        if not query or not query.strip():
            return []

        tokenized_query = self.tokenizer.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_k = min(top_k, len(scores))
        top_indices = scores.argsort()[::-1][:top_k]

        hits: list[Hit] = []
        for i, idx in enumerate(top_indices):
            recipe_id = self.recipe_ids[idx]
            # BM25 raw score: 높을수록 관련성 높음. Dense distance와 스케일/방향 모두 다름.
            score = float(scores[idx])
            meta = self.recipe_metadata.get(recipe_id, {})
            rank = i + 1

            hits.append(Hit(
                recipe_id   = recipe_id,
                rank        = rank,
                score       = score,
                source      = "bm25",
                name        = meta.get("name"),
                metadata    = meta,
                dense_rank  = None,
                dense_score = None,
                bm25_rank   = rank,
                bm25_score  = score,
            ))

        return hits


class HybridRetriever:
    """Dense + BM25를 RRF로 병합하는 Hybrid 검색기."""

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever) -> None:
        self.dense = dense
        self.bm25 = bm25

    def search(
        self,
        query: str,
        top_k: int | None = None,
        exclude_ids: list[str] | None = None,
    ) -> list[Hit]:
        if not query or not query.strip():
            return []

        if top_k is None:
            top_k = RRF_TOP_K

        dense_hits = self.dense.search(query, top_k=DENSE_TOP_K)
        bm25_hits  = self.bm25.search(query, top_k=BM25_TOP_K)

        hits = self._rrf_merge(dense_hits, bm25_hits, top_k)

        if not exclude_ids:
            return hits

        # 정규화: "recipe_42" / "42" / " 42 " 모두 "42"로 비교.
        exclude_set = {
            str(rid).strip().removeprefix("recipe_").strip()
            for rid in exclude_ids
        }

        filtered: list[Hit] = []
        for hit in hits:
            rid_norm = str(hit.recipe_id).strip().removeprefix("recipe_").strip()
            if rid_norm in exclude_set:
                continue
            hit.rank = len(filtered) + 1
            filtered.append(hit)

        return filtered

    def _rrf_merge(
        self,
        dense_hits: list[Hit],
        bm25_hits:  list[Hit],
        top_k:      int,
    ) -> list[Hit]:
        rrf_scores: dict[str, float] = {}

        dense_rank_map:  dict[str, int]   = {}
        dense_score_map: dict[str, float] = {}
        bm25_rank_map:   dict[str, int]   = {}
        bm25_score_map:  dict[str, float] = {}

        name_map:     dict[str, str | None] = {}
        metadata_map: dict[str, dict]       = {}

        for hit in dense_hits:
            rid = hit.recipe_id
            rrf_scores[rid] = rrf_scores.get(rid, 0.0) + 1.0 / (RRF_K + hit.rank)
            dense_rank_map[rid]  = hit.rank
            dense_score_map[rid] = hit.score
            name_map[rid]     = hit.name
            metadata_map[rid] = hit.metadata or {}

        for hit in bm25_hits:
            rid = hit.recipe_id
            rrf_scores[rid] = rrf_scores.get(rid, 0.0) + 1.0 / (RRF_K + hit.rank)
            bm25_rank_map[rid]  = hit.rank
            bm25_score_map[rid] = hit.score
            if rid not in name_map:
                name_map[rid] = hit.name
            if rid not in metadata_map or not metadata_map[rid]:
                metadata_map[rid] = hit.metadata or {}

        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda rid: rrf_scores[rid],
            reverse=True,
        )[:top_k]

        hits: list[Hit] = []
        for i, rid in enumerate(sorted_ids):
            hits.append(Hit(
                recipe_id   = rid,
                rank        = i + 1,
                # RRF 누적 점수: 높을수록 좋음. Dense distance / BM25 raw score와 스케일/방향 모두 다름.
                score       = rrf_scores[rid],
                source      = "hybrid",
                name        = name_map.get(rid),
                metadata    = metadata_map.get(rid, {}),
                dense_rank  = dense_rank_map.get(rid),
                dense_score = dense_score_map.get(rid),
                bm25_rank   = bm25_rank_map.get(rid),
                bm25_score  = bm25_score_map.get(rid),
            ))

        return hits