"""
manual_search_check.py

페르소나 기반 10개 쿼리로 ChromaDB top-K 결과를 출력. 사람이 정성 점검용.
P@5/MRR 같은 정량 평가 아님.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "chroma" / "recipes_v1"
COLLECTION_NAME = "meal_bot_recipes_v1"
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"

QUERIES = [
    ("P1", "퇴근하고 너무 피곤한데 간단하게 먹을 저녁"),
    ("P2", "다이어트 중인데 배부르게 먹을 수 있는 저녁"),
    ("P3", "운동 끝나고 빨리 먹을 수 있는 단백질 많은 거"),
    ("P4", "아이도 먹을 수 있게 맵지 않은 저녁 반찬"),
    ("P5", "요리 처음 해보는데 실패 확률 낮은 메뉴"),
    ("P6", "부모님이 드시기 좋은 자극적이지 않은 한식"),
    ("P7", "맥주랑 같이 먹을 간단한 안주"),
    ("P8", "손님 왔을 때 있어 보이는 한식 메인요리"),
    ("P3-건강", "고단백 도시락"),
    ("P6-건강", "저염 한식"),
]


def extract_summary(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.splitlines():
        if line.startswith("[요약] "):
            return line[len("[요약] "):].strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--save", type=str, default=None,
                    help="결과 텍스트 저장 경로 (예: artifacts/manual_search_results.txt)")
    args = ap.parse_args()

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    out_lines: list[str] = []

    def emit(s: str = ""):
        print(s)
        out_lines.append(s)

    emit(f"# manual_search_check (top-{args.top_k})")
    emit(f"collection: {COLLECTION_NAME}  count={collection.count()}")
    emit("")

    for tag, q in QUERIES:
        emit(f"=== {tag}: {q} ===")
        res = collection.query(query_texts=[q], n_results=args.top_k)
        ids = res["ids"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        docs = res["documents"][0]
        for i, (rid, meta, dist, doc) in enumerate(zip(ids, metas, dists, docs), 1):
            name = meta.get("name", "?")
            cat = meta.get("category", "?")
            emit(f"{i}. [{cat}] {name} ({rid}, dist={dist:.4f})")
            summary = extract_summary(doc)
            if summary:
                emit(f"   요약: {summary[:100]}")
        emit("")

    if args.save:
        out_path = Path(args.save)
        if not out_path.is_absolute():
            out_path = BASE_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
        print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
