"""
jhgan/ko-sbert-nli 모델 스모크 테스트.
모델 다운로드 + 임베딩 정상 생성 확인.

실행:
    cd rag-api
    source venv/bin/activate
    python scripts/test_embedding_model.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import EMBEDDING_MODEL_NAME

SAMPLE_TEXTS = [
    "김치찌개. 국 찌개 요리. 끓이기 방식. 주재료 김치 돼지고기 두부 대파.",
    "초코 케이크. 후식 요리. 굽기 방식. 주재료 초콜릿 밀가루 설탕 버터 달걀.",
    "시금치 무침. 반찬 요리. 주재료 시금치 마늘 참기름 깨 소금.",
]


def main():
    print(f"모델 로드 중: {EMBEDDING_MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"  완료 ({time.time() - t0:.1f}초)")

    print(f"\n임베딩 생성 중: {len(SAMPLE_TEXTS)}건")
    t0 = time.time()
    embeddings = model.encode(SAMPLE_TEXTS, show_progress_bar=False)
    print(f"  완료 ({time.time() - t0:.2f}초)")

    print(f"\n결과:")
    print(f"  shape: {embeddings.shape}")
    print(f"  dtype: {embeddings.dtype}")

    for text, vec in zip(SAMPLE_TEXTS, embeddings):
        print(f"\n  텍스트: {text[:40]}...")
        print(f"  벡터[:5]: {vec[:5]}")
        print(f"  norm: {(vec ** 2).sum() ** 0.5:.4f}")

    # 의미 유사도 sanity check
    from sentence_transformers.util import cos_sim
    print(f"\n[의미 유사도 검증]")
    sim_01 = cos_sim(embeddings[0], embeddings[1]).item()  # 찌개 vs 케이크
    sim_02 = cos_sim(embeddings[0], embeddings[2]).item()  # 찌개 vs 반찬
    sim_12 = cos_sim(embeddings[1], embeddings[2]).item()  # 케이크 vs 반찬
    print(f"  김치찌개 ↔ 초코케이크: {sim_01:.4f}")
    print(f"  김치찌개 ↔ 시금치무침: {sim_02:.4f}")
    print(f"  초코케이크 ↔ 시금치무침: {sim_12:.4f}")


if __name__ == "__main__":
    main()
