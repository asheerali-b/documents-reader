from __future__ import annotations

import os
from functools import lru_cache

from langchain_huggingface.embeddings import HuggingFaceEmbeddings


@lru_cache(maxsize=2)
def _get_model(model_name: str, device: str) -> HuggingFaceEmbeddings:
    # Xet-backed downloads can fail with 403 in some enterprise/proxy setups.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # Prevent HuggingFace from reaching out to the hub on every load.
    # If the model is already cached locally this is safe; if not, the load
    # will fail with a clear "not cached" message rather than a network error.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )


class HuggingFaceEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = _get_model(self._model_name, self._device)
        return model.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        model = _get_model(self._model_name, self._device)
        return model.embed_query(query)
