import os

from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH          = "chroma/recipes_v2"
COLLECTION_NAME      = "meal_bot_recipes_v2"
EMBEDDING_MODEL      = "BAAI/bge-m3"
NORMALIZE_EMBEDDINGS = True

# BM25
BM25_INDEX_PATH = "bm25/recipes_v2"
BM25_TOP_K      = 50

# Dense
DENSE_TOP_K = 50

# Hybrid
RRF_TOP_K = 30
RRF_K     = 60

# LLM Rerank
OPENAI_API_KEY               = os.getenv("OPENAI_API_KEY")
RERANK_MODEL                 = "gpt-5-mini"
RERANK_REASONING_EFFORT      = "low"
RERANK_MAX_COMPLETION_TOKENS = 3000
RERANK_TOP_K_OUTPUT          = 5
RERANK_MIN_CANDIDATES        = 5
RERANK_RETRY_LIMIT           = 1

# LLM QA (후속 질문 답변)
# NOTE: 기존 RERANK_* prefix와 통일성을 위해 향후 QA_* / RERANK_* 로 prefix 정리 고려.
RAG_QA_MODEL            = RERANK_MODEL
RAG_QA_MAX_TOKENS       = 4000
RAG_QA_REASONING_EFFORT = RERANK_REASONING_EFFORT
RAG_QA_RETRY_LIMIT      = 1
QA_MAX_DOCS             = 3

# Logging
LOG_DIR                  = "logs"
RERANK_LOG_FILE_PATTERN  = "rerank_{date}.jsonl"
QA_LOG_FILE_PATTERN      = "qa_{date}.jsonl"