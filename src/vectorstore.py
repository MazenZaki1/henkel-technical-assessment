"""Shared embedding + Qdrant access.

Both the ingestion script (write path) and the RAG app (read path) import these
helpers so they are guaranteed to use the same embedding model and the same
collection. This is the single most important safeguard against the classic RAG
failure mode: indexing with one model and querying with another.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import Settings


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """OpenAI embedding model used identically at ingestion and query time."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def get_client(settings: Settings) -> QdrantClient:
    """Raw Qdrant Cloud client."""
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def get_vector_store(settings: Settings) -> QdrantVectorStore:
    """Connect to the already-populated collection for querying (read path).

    Used by the chatbot at runtime. Assumes ingestion has already created and
    filled the collection; raises if the collection is missing.
    """
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(settings),
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
