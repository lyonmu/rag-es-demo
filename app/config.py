"""Application configuration loaded from .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ES
    es_host: str = "localhost"
    es_port: int = 9200
    es_user: str = "elastic"
    es_password: str = "Cass@123456"

    @property
    def es_url(self) -> str:
        return f"http://{self.es_host}:{self.es_port}"

    # Embedding
    embedding_model_path: str = "data/models/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_batch_size: int = 32

    # LLM (OpenAI 标准协议)
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7
    llm_timeout: int = 30

    # RRF
    rrf_k: int = 60
    retriever_top_k: int = 5
    retriever_pre_multiplier: int = 2

    # Chunking
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150
    chunk_min_chars: int = 80

    # Query
    query_embedding_cache_size: int = 256

    # Ingest
    ingest_refresh_policy: str = "false"

    # BM25
    bm25_heading_boost: float = 2.0
    bm25_content_boost: float = 1.0
    bm25_phrase_boost: float = 1.5

    # Chat
    chat_context_max_chars: int = 6000

    # Vector candidate
    vector_candidate_mode: bool = False
    vector_candidate_top_k: int = 200

    # Service
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    # Upload
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Global singleton
settings = Settings()
