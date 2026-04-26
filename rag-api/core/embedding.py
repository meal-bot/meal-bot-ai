"""
임베딩 모델 래퍼.
ko-sbert-nli 모델을 프로세스당 1회만 로드하여 재사용.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    임베딩 모델 싱글톤 반환.
    최초 호출 시 로드 (약 2~3초), 이후 캐시 재사용.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def encode(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    텍스트 리스트 → 임베딩 벡터 배열.

    Args:
        texts: 인코딩할 문자열 리스트
        batch_size: 배치 크기 (기본 32)

    Returns:
        shape=(len(texts), EMBEDDING_DIM) ndarray
    """
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
