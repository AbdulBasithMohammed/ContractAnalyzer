from pathlib import Path

from pydantic_settings import BaseSettings

# Pin .env to backend/.env regardless of where uvicorn/streamlit is launched
# from. Without this, pydantic-settings resolves the path against CWD and
# misses the file when run from the repo root.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_ENV_FILE = _BACKEND_DIR / ".env"


class Settings(BaseSettings):
    anthropic_api_key: str
    voyage_api_key: str
    vision_model: str = "claude-haiku-4-5-20251001"
    analysis_model: str = "claude-sonnet-4-6"
    embedding_model: str = "voyage-law-2"
    embedding_dim: int = 1024
    embedding_batch_size: int = 8
    embedding_batch_sleep_sec: float = 21.0
    embedding_max_retries: int = 5
    # `:memory:` runs an embedded Qdrant inside the uvicorn process and
    # dies with it. A URL (e.g. http://localhost:6333) points at a server
    # whose collections survive restarts — the production path.
    qdrant_location: str = "http://localhost:6333"
    session_db_path: str = str(_BACKEND_DIR / "sessions.db")
    max_chunk_tokens: int = 1000
    chunk_overlap_tokens: int = 200
    retrieval_top_k: int = 5
    retrieval_context_tokens: int = 8000
    analysis_max_tokens: int = 1024
    analysis_max_retries: int = 3
    chat_top_k: int = 6
    chat_max_tokens: int = 1024
    min_text_length: int = 100
    significant_image_min_dim: int = 200
    upload_dir: str = "uploads"

    model_config = {"env_file": str(_ENV_FILE)}


settings = Settings()
