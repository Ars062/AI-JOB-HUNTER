from functools import lru_cache

import numpy as np

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    cleaned = [t[:5000] if t else "" for t in texts]
    vectors = _get_model().encode(
        cleaned, normalize_embeddings=True, convert_to_numpy=True, batch_size=32, show_progress_bar=False
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])
