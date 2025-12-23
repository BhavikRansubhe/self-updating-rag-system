import os
from pathlib import Path

# Load .env from project root (../.env) if present
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
except Exception:
    pass

DATA_DIR = os.getenv("DATA_DIR", "../data")
DOCS_DIR = os.getenv("DOCS_DIR", "../docs")

# Providers:
# - EMBED_PROVIDER: "sbert" (default) or "openai"
# - LLM_PROVIDER:   "local" (default) or "openai"
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "sbert").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
if LLM_PROVIDER == "fallback":
    LLM_PROVIDER = "local"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Chunking
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "250"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "8"))
MAX_CONTEXTS = int(os.getenv("MAX_CONTEXTS", "4"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))
SCORE_WINDOW = float(os.getenv("SCORE_WINDOW", "0.08"))
