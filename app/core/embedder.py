"""Local embedding model singleton using sentence-transformers."""

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Get the embedding model instance. Preloaded on startup."""
    global _model
    if _model is None:
        logger.info("Loading embedding model from %s", settings.embedding_model_path)
        _model = SentenceTransformer(settings.embedding_model_path)
        logger.info("Embedding model loaded successfully")
    return _model


def encode_texts(texts: list[str]) -> list[list[float]]:
    """Batch encode text list into vectors."""
    model = get_embedder()
    embeddings = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


@lru_cache(maxsize=settings.query_embedding_cache_size)
def encode_query(text: str) -> list[float]:
    """Encode a single query with an in-process LRU cache."""
    return encode_texts([text])[0]
