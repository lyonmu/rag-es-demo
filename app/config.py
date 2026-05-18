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
