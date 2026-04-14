from pydantic_settings import BaseSettings


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
    qdrant_location: str = ":memory:"
    max_chunk_tokens: int = 1000
    chunk_overlap_tokens: int = 200
    retrieval_top_k: int = 5
    retrieval_context_tokens: int = 8000
    min_text_length: int = 100
    significant_image_min_dim: int = 200
    upload_dir: str = "uploads"

    model_config = {"env_file": ".env"}


settings = Settings()
