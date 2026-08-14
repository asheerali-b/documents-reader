from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from src.chatbot.knowledge.types import ChunkRecord


class ChromaKnowledgeStore:
    def __init__(self, db_dir: Path, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=str(db_dir))
        self._collection_name = collection_name
        self._collection = self._get_or_create_collection()

    def reset_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._get_or_create_collection()

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        ids = [item.chunk_id for item in chunks]
        docs = [item.text for item in chunks]
        metadatas = [
            {
                "source": item.source,
                "chunk_index": item.chunk_index,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "content_hash": item.content_hash,
            }
            for item in chunks
        ]

        self._with_collection_retry(
            lambda: self._collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        )

    def query(self, query_embedding: list[float], n_results: int) -> list[tuple[ChunkRecord, float]]:
        result = self._with_collection_retry(
            lambda: self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[tuple[ChunkRecord, float]] = []
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] or {}
            chunk = ChunkRecord(
                chunk_id=chunk_id,
                source=str(metadata.get("source", "unknown")),
                chunk_index=int(metadata.get("chunk_index", idx)),
                start_line=int(metadata.get("start_line", 1)),
                end_line=int(metadata.get("end_line", 1)),
                content_hash=str(metadata.get("content_hash", "")),
                text=str(docs[idx]),
            )
            distance = float(distances[idx])
            dense_score = max(0.0, 1.0 - distance)
            hits.append((chunk, dense_score))
        return hits

    def get_all_chunks(self) -> list[ChunkRecord]:
        result = self._with_collection_retry(
            lambda: self._collection.get(include=["documents", "metadatas"])
        )
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        records: list[ChunkRecord] = []
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] or {}
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    source=str(metadata.get("source", "unknown")),
                    chunk_index=int(metadata.get("chunk_index", idx)),
                    start_line=int(metadata.get("start_line", 1)),
                    end_line=int(metadata.get("end_line", 1)),
                    content_hash=str(metadata.get("content_hash", "")),
                    text=str(docs[idx]),
                )
            )
        return records

    def delete_source(self, source: str) -> int:
        source_name = source.strip()
        if not source_name:
            return 0

        result = self._with_collection_retry(
            lambda: self._collection.get(where={"source": source_name}, include=[])
        )
        ids = result.get("ids", [])

        if ids:
            self._with_collection_retry(lambda: self._collection.delete(ids=ids))
            return len(ids)

        self._with_collection_retry(lambda: self._collection.delete(where={"source": source_name}))
        return 0

    def count(self) -> int:
        return self._with_collection_retry(lambda: self._collection.count())

    def _get_or_create_collection(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _with_collection_retry(self, operation):
        try:
            return operation()
        except Exception as ex:
            if "does not exist" in str(ex).lower() and "collection" in str(ex).lower():
                self._collection = self._get_or_create_collection()
                return operation()
            raise
