"""
프로젝트 전역 설정.
모든 상수/환경변수는 여기에서만 관리.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# .env 로드 (프로젝트 루트 기준)
PROJECT_ROOT = Path(__file__).parent.parent  # rag-api/
load_dotenv(PROJECT_ROOT / ".env")

# ─── MySQL ───────────────────────────────────────
MYSQL_CONFIG = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "port":     int(os.getenv("MYSQL_PORT", 3308)),
    "user":     os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE", "meal_bot"),
    "charset":  "utf8mb4",
}

# ─── Chroma ──────────────────────────────────────
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "recipes_v1"

# ─── Embedding ───────────────────────────────────
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli"
EMBEDDING_DIM = 768

# ─── Retrieval ───────────────────────────────────
DEFAULT_TOP_K = 5

# ─── OpenAI (v2+ 사용 예정) ─────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
