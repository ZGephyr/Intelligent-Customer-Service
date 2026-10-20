"""应用配置：统一由环境变量 / .env 注入，启动即校验。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LLM_BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"
DEFAULT_EMBEDDING_PATH = r"D:\python\eduRAG\rag_qa\models\bge-m3"
DEFAULT_RERANK_PATH = r"D:\python\eduRAG\rag_qa\models\bge-reranker-v2-m3"


class Settings(BaseSettings):
    """全局配置。敏感项只从环境变量 / .env 读取，代码中不出现明文。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        case_sensitive=False,
    )

    app_env: str = "dev"
    log_level: str = "INFO"
    log_dir: Path = Path("logs")

    llm_api_key: str = Field(validation_alias=AliasChoices("LLM_API_KEY", "DASHSCOPE_API_KEY"))
    llm_base_url: str = Field(
        default=DEFAULT_LLM_BASE_URL,
        validation_alias=AliasChoices("LLM_BASE_URL", "DASHSCOPE_BASE_URL"),
    )
    llm_model: str = "qwen3.8-flash"
    llm_timeout_seconds: float = 60
    rewrite_timeout_seconds: float = 3
    llm_temperature: float = 0.2

    embedding_model_path: str = DEFAULT_EMBEDDING_PATH
    embedding_device: str = "auto"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32

    rerank_enabled: bool = True
    rerank_model_path: str = DEFAULT_RERANK_PATH

    milvus_uri: str = "http://192.168.34.97:19530"
    milvus_token: str | None = None
    milvus_database: str = "rag_hzg"
    milvus_collection: str = "rag_project"

    mysql_url: str
    redis_url: str
    admin_token: str
    admin_auth_enabled: bool = False

    candidate_k: int = 20
    rerank_candidates: int = 10
    top_k: int = 5
    score_threshold: float = 0.35
    rerank_score_threshold: float = 0.3
    rrf_k: int = 60
    max_context_chars: int = 6000

    history_max_messages: int = 20
    history_ttl_seconds: int = 3600

    upload_dir: Path = Path("data/uploads")
    max_upload_mb: int = 20
    ws_allowed_origins: str = "http://127.0.0.1:8000"
    shop_name: str = "本店"
    handoff_hint: str = "建议您联系人工客服"

    @property
    def allowed_origins(self) -> list[str]:
        """WebSocket 握手白名单，逗号分隔。"""
        return [origin.strip() for origin in self.ws_allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
