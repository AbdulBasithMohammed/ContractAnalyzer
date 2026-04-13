from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    vision_model: str = "claude-haiku-4-5-20251001"
    analysis_model: str = "claude-sonnet-4-6"
    max_chunk_tokens: int = 1000
    chunk_overlap_tokens: int = 200
    retrieval_top_k: int = 5
    min_text_length: int = 100
    significant_image_min_dim: int = 200
    upload_dir: str = "uploads"

    model_config = {"env_file": ".env"}


settings = Settings()
