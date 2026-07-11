"""Central configuration. Every tunable lives here, sourced from env / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- secrets / auth ---
    gemini_api_key: str = ""
    # Optional comma-separated key rotation (first = primary). Each free-tier
    # key is its own project with its own quotas, so the LLM wrapper exhausts
    # a model across ALL keys before falling back to the next model.
    gemini_api_keys: str = ""
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
    # Fallback chain tried in order when a model's budget is spent or it 429s
    # (best model first). All ids probe-verified against AI Studio.
    model_fallbacks: str = (
        "gemini/gemini-3-flash-preview,gemini/gemini-2.5-flash,"
        "gemini/gemini-3.1-flash-lite,gemini/gemini-2.5-flash-lite"
    )
    # LLM-as-judge model — deliberately NOT the answer model, so the eval isn't
    # a model grading its own output.
    model_judge: str = "gemini/gemini-2.5-flash"
    # Free-tier budgets from the AI Studio rate-limit dashboard (2026-07-11):
    # 3.5-flash = 5 RPM / 20 RPD, 3.1-flash-lite = 15 RPM / 500 RPD.
    rpm_main: int = 5
    rpd_main: int = 20
    rpm_cheap: int = 15
    rpd_cheap: int = 500
    embedding_model: str = "BAAI/bge-m3"
    # Benchmarked on this corpus: identical hit rate to BAAI/bge-reranker-v2-m3
    # at ~25x lower CPU latency (150-250ms vs ~5s per query in-container).
    # Swap back to the bge reranker for GPU dev via RERANKER_MODEL.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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

    # --- vector store ---
    # When CHROMA_HOST is set the api talks to a dedicated Chroma service
    # (client/server mode — production shape, used in docker-compose).
    # When empty, Chroma runs embedded in-process (local dev and unit tests).
    chroma_host: str = ""
    chroma_port: int = 8000

    # --- storage paths ---
    # In embedded mode this dir holds the Chroma data itself; in server mode it
    # holds only the api's derived state (docs.json registry cache + BM25 pickle).
    chroma_dir: str = "storage/chroma"
    bm25_path: str = "storage/chroma/bm25.pkl"
    data_dir: str = "data"
    prebuilt_index_dir: str = "prebuilt_index"

    # --- rate limits (per authenticated user) ---
    rate_limit_chat: str = "30/minute"
    rate_limit_ingest: str = "5/minute"

    log_level: str = "INFO"


settings = Settings()
