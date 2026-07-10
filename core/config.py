"""Central configuration. Every tunable lives here, sourced from env / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- secrets / auth ---
    gemini_api_key: str = ""
    jwt_secret: str = "change-me-in-env"
    jwt_ttl_minutes: int = 60
    demo_username: str = "demo"
    demo_password: str = "demo123"

    # --- services ---
    api_port: int = 8000
    ui_port: int = 7860
    redis_url: str = "redis://localhost:6379/0"

    # --- models ---
    model_main: str = "gemini/gemini-3.5-flash"  # ⚠ VERIFY against LiteLLM Gemini provider docs
    model_cheap: str = "gemini/gemini-3.1-flash-lite"  # ⚠ VERIFY against LiteLLM Gemini provider docs
    rpm_main: int = 10
    rpm_cheap: int = 15
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_device: str = "cpu"  # cuda for local dev only; Docker submission is CPU-only
    embed_batch_size: int = 8
    eager_load_models: bool = True  # disabled in unit tests to avoid model downloads

    # --- optional local Ollama fallback (never used in the Docker submission) ---
    ollama_fallback_enabled: bool = False
    ollama_model: str = "qwen2.5:14b-instruct-q4_k_m"
    ollama_base_url: str = "http://localhost:11434"

    # --- retrieval / chat tuning ---
    history_window_turns: int = 6
    history_token_budget: int = 2000
    min_rerank_score: float = 0.25
    dense_top_k: int = 12
    sparse_top_k: int = 12
    rerank_candidates: int = 10
    final_top_k: int = 4
    rrf_k: int = 60

    # --- chunking ---
    chunk_target_tokens: int = 450
    chunk_overlap_ratio: float = 0.15
    heading_font_ratio: float = 1.2

    # --- storage paths ---
    chroma_dir: str = "storage/chroma"
    bm25_path: str = "storage/chroma/bm25.pkl"
    data_dir: str = "data"
    prebuilt_index_dir: str = "prebuilt_index"

    # --- rate limits (per authenticated user) ---
    rate_limit_chat: str = "30/minute"
    rate_limit_ingest: str = "5/minute"

    log_level: str = "INFO"


settings = Settings()
