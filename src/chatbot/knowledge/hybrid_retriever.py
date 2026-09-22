from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.chatbot.knowledge.chroma_store import ChromaKnowledgeStore
from src.chatbot.knowledge.embedding_provider import HuggingFaceEmbedder
from src.chatbot.knowledge.types import ChunkRecord, RetrievedChunk


@dataclass
class RetrievalConfig:
    alpha: float
    dense_top_k: int
    sparse_top_k: int
    final_top_k: int


class HybridRetriever:
    def __init__(
        self,
        store: ChromaKnowledgeStore,
        embedder: HuggingFaceEmbedder,
        config: RetrievalConfig,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config
        self._all_chunks: list[ChunkRecord] = []
        self._bm25: BM25Okapi | None = None

    def refresh(self) -> int:
        self._all_chunks = self._store.get_all_chunks()
        corpus = [self._tokenize(chunk.text) for chunk in self._all_chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        return len(self._all_chunks)

    def search(self, query: str) -> list[RetrievedChunk]:
        if not query.strip() or not self._all_chunks:
            return []

        dense_hits = self._store.query(
            query_embedding=self._embedder.embed_query(query),
            n_results=self._config.dense_top_k,
        )

        sparse_scores = self._sparse_scores(query)
        sparse_hits = sorted(sparse_scores.items(), key=lambda item: item[1], reverse=True)
        sparse_hits = sparse_hits[: self._config.sparse_top_k]

        dense_map = {chunk.chunk_id: (chunk, score) for chunk, score in dense_hits}
        sparse_map = {chunk_id: score for chunk_id, score in sparse_hits}
        chunks_by_id = {chunk.chunk_id: chunk for chunk in self._all_chunks}

        dense_max = max((score for _, score in dense_hits), default=1.0)
        sparse_max = max((score for _, score in sparse_hits), default=1.0)

        combined_ids = set(dense_map.keys()) | set(sparse_map.keys())
        merged: list[RetrievedChunk] = []

        for chunk_id in combined_ids:
            dense_result = dense_map.get(chunk_id)
            if dense_result is not None:
                chunk, dense_score = dense_result
            else:
                chunk = chunks_by_id.get(chunk_id)
                if chunk is None:
                    # The index changed while searching; ignore this stale sparse hit.
                    continue
                dense_score = 0.0
            sparse_score = sparse_map.get(chunk_id, 0.0)

            dense_norm = dense_score / dense_max if dense_max > 0 else 0.0
            sparse_norm = sparse_score / sparse_max if sparse_max > 0 else 0.0

            score = (self._config.alpha * dense_norm) + ((1.0 - self._config.alpha) * sparse_norm)
            merged.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                )
            )

        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[: self._config.final_top_k]

    def _sparse_scores(self, query: str) -> dict[str, float]:
        if not self._bm25:
            return {}

        tokenized = self._tokenize(query)
        raw_scores = self._bm25.get_scores(tokenized)

        results: dict[str, float] = {}
        for idx, score in enumerate(raw_scores):
            if score <= 0:
                continue
            results[self._all_chunks[idx].chunk_id] = float(score)
        return results

    def _chunk_by_id(self, chunk_id: str) -> ChunkRecord:
        for chunk in self._all_chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise KeyError(f"Chunk not found for id: {chunk_id}")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
