"""
core/embeddings.py
------------------
Lightweight LangChain-compatible embeddings backed by fastembed (ONNX runtime).

Replaces sentence-transformers / langchain-huggingface with the same
all-MiniLM-L6-v2 model but without PyTorch — fastembed runs it via ONNX,
which is dramatically lighter.

Vectors are identical in shape (384-dim) and compatible with
sentence-transformers/all-MiniLM-L6-v2.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings wrapper around fastembed TextEmbedding.

    Drop-in replacement for HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2").
    Requires: fastembed>=0.3.0
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding
        from pathlib import Path
        # Use a flat cache dir to avoid Windows HuggingFace hub symlink issues
        cache_dir = Path.home() / ".cache" / "fastembed"
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()
