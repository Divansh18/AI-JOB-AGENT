"""Local embedding encoder.

fastembed / bge-small-en-v1.5 runs on CPU and costs nothing per call, which is
what makes it affordable to embed every job. A deterministic hashing encoder
is used as a fallback so the pipeline never hard-fails if the model is absent.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384


class HashingEncoder:
    """Deterministic bag-of-words fallback. Lexical, not semantic."""

    model_name = "hashing-fallback-v1"
    dim = DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9\.\+#]+", (text or "").lower()):
                h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)


class FastEmbedEncoder:
    model_name = MODEL_NAME
    dim = DIM

    def __init__(self):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = list(self._model.embed(texts))
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)


@lru_cache(maxsize=1)
def get_encoder(prefer_local_model: bool = True):
    if prefer_local_model:
        try:
            return FastEmbedEncoder()
        except Exception:
            pass
    return HashingEncoder()


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes, dim: int = DIM) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(dim)


def cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row against a query vector (both L2-normed)."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    return matrix @ query
