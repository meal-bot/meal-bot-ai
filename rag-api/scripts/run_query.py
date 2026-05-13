"""Dense / BM25 / Hybrid 3-way 검색 비교 스크립트."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import DenseRetriever, BM25Retriever, HybridRetriever

QUERIES = [
    "매콤한 국물 요리 추천",
    "다이어트에 좋은 저칼로리 메뉴",
    "아이가 먹기 좋은 부드러운 요리",
    "10분 안에 만드는 간단한 한 끼",
    "고단백 닭가슴살 요리",
]

SEP = "━" * 60


def fmt_rank(value):
    return value if value is not None else "-"


def main() -> None:
    dense  = DenseRetriever()
    bm25   = BM25Retriever()
    hybrid = HybridRetriever(dense, bm25)

    for q in QUERIES:
        print(SEP)
        print(f"Query: {q}")
        print(SEP)

        d_hits = dense.search(q, top_k=5)
        b_hits = bm25.search(q, top_k=5)
        h_hits = hybrid.search(q, top_k=5)

        print("\n[Dense]  ※ Chroma distance, 낮을수록 좋음")
        for h in d_hits:
            print(f"  {h.rank}. {h.name} (d={h.score:.3f})")

        print("\n[BM25]   ※ raw score, 높을수록 좋음")
        for h in b_hits:
            print(f"  {h.rank}. {h.name} (s={h.score:.2f})")

        print("\n[Hybrid] ※ RRF score, 높을수록 좋음")
        for h in h_hits:
            d = fmt_rank(h.dense_rank)
            b = fmt_rank(h.bm25_rank)
            print(f"  {h.rank}. {h.name} (rrf={h.score:.4f}) [D={d}, B={b}]")

        print()


if __name__ == "__main__":
    main()