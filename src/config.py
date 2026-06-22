"""Centralized application configuration.

Every tunable lives here and is sourced from environment variables (loaded from
`.env` locally, injected via `--env-file` in Docker). Ingestion and serving both
import this single object so the embedding model, vector dimension, collection
name, and chunking parameters can never drift apart between the two phases.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI (one key powers both embeddings and chat) ---
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings + chat")
    embedding_model: str = Field(
        "text-embedding-3-small", description="OpenAI embedding model name"
    )
    embedding_dim: int = Field(
        1536, description="Vector size of the embedding model; must match the Qdrant collection"
    )
    chat_model: str = Field("gpt-4o-mini", description="OpenAI chat/generation model")
    chat_temperature: float = Field(
        0.0, description="0 = deterministic, factual answers (no creative drift)"
    )

    # --- Qdrant Cloud (cloud-hosted vector store) ---
    qdrant_url: str = Field(..., description="Qdrant Cloud cluster URL")
    qdrant_api_key: str = Field(..., description="Qdrant Cloud API key")
    qdrant_collection: str = Field(
        "iphone_user_guide", description="Name of the Qdrant collection holding the chunks"
    )

    # --- Chunking (justified in the ingestion script / README) ---
    chunk_size: int = Field(1000, description="Max characters per chunk")
    chunk_overlap: int = Field(150, description="Character overlap between adjacent chunks")

    # --- Retrieval ---
    retrieval_k: int = Field(5, description="Number of chunks retrieved per query")

    # --- Source document ---
    pdf_path: str = Field(
        "data/iphone-user-guide.pdf", description="Path to the source PDF (ingestion only)"
    )

    # --- App ---
    app_port: int = Field(8000, description="Port the Chainlit app listens on")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read once per process)."""
    return Settings()
