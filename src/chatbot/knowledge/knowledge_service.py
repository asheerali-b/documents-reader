from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

from src.chatbot.constant import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    DOCUMENTS_DIR,
    EMBEDDING_DEVICE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    FINAL_TOP_K,
    HYBRID_ALPHA,
    KNOWLEDGE_CHUNK_OVERLAP,
    KNOWLEDGE_CHUNK_SIZE,
    LOGS_DIR,
    RESET_COLLECTION_WHEN_EMPTY,
    SPARSE_TOP_K,
    VECTOR_TOP_K,
)
from src.chatbot.knowledge.chroma_store import ChromaKnowledgeStore
from src.chatbot.knowledge.document_loader import chunk_document, load_txt_documents
from src.chatbot.knowledge.embedding_provider import HuggingFaceEmbedder
from src.chatbot.knowledge.hybrid_retriever import HybridRetriever, RetrievalConfig
from src.chatbot.knowledge.types import ChunkRecord, DocumentStatus, RetrievedChunk


class KnowledgeService:
    def __init__(self) -> None:
        # Load .env so HF_TOKEN and related variables are available in ingestion runs.
        load_dotenv()
        self._documents_dir = Path(DOCUMENTS_DIR)
        self._logs_dir = Path(LOGS_DIR)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._logger = _build_logger(self._logs_dir / "knowledge_queries.log")
        self._store = ChromaKnowledgeStore(
            db_dir=Path(CHROMA_DB_DIR),
            collection_name=CHROMA_COLLECTION_NAME,
        )
        self._embedder = HuggingFaceEmbedder(
            model_name=EMBEDDING_MODEL_NAME,
            device=EMBEDDING_DEVICE,
        )
        self._retriever = HybridRetriever(
            store=self._store,
            embedder=self._embedder,
            config=RetrievalConfig(
                alpha=HYBRID_ALPHA,
                dense_top_k=VECTOR_TOP_K,
                sparse_top_k=SPARSE_TOP_K,
                final_top_k=FINAL_TOP_K,
            ),
        )
        self._retriever.refresh()

    def build_index(self) -> dict[str, int]:
        documents = load_txt_documents(self._documents_dir)
        existing_chunks = self._store.get_all_chunks()
        indexed_hashes_by_source = self._group_hashes(existing_chunks)

        chunks_to_upsert: list[ChunkRecord] = []
        ingested_documents = 0
        skipped_documents = 0

        for document in documents:
            existing_hashes = indexed_hashes_by_source.get(document.source, set())
            if document.content_hash in existing_hashes:
                skipped_documents += 1
                continue

            # Document changed or first-time ingest: remove old chunks for this source before upsert.
            if existing_hashes:
                self._store.delete_source(document.source)

            new_chunks = chunk_document(document, KNOWLEDGE_CHUNK_SIZE, KNOWLEDGE_CHUNK_OVERLAP)
            chunks_to_upsert.extend(new_chunks)
            ingested_documents += 1

        if chunks_to_upsert:
            embeddings = self._embedder.embed_texts([chunk.text for chunk in chunks_to_upsert])
            self._validate_embedding_dimensions(embeddings)
            try:
                self._store.upsert_chunks(chunks_to_upsert, embeddings)
            except Exception as ex:
                # Handle migration between embedding models with different vector dimensions.
                if "dimension" in str(ex).lower():
                    self._logger.warning(
                        "Embedding dimension mismatch detected. Rebuilding full collection. details=%s",
                        ex,
                    )
                    return self._rebuild_full_index(documents)
                raise

        indexed_count = self._retriever.refresh()
        statuses = self.get_document_statuses()
        embedded_documents = sum(1 for status in statuses if status.embedded)

        self._logger.info(
            "Ingestion complete | total_docs=%s | ingested_now=%s | skipped_same=%s | embedded_docs=%s | total_chunks=%s",
            len(documents),
            ingested_documents,
            skipped_documents,
            embedded_documents,
            indexed_count,
        )

        return {
            "documents_total": len(documents),
            "documents_embedded": embedded_documents,
            "documents_ingested_now": ingested_documents,
            "documents_skipped": skipped_documents,
            "chunks": indexed_count,
        }

    def delete_document_embeddings(self, source: str) -> bool:
        source_name = source.strip()
        if not source_name:
            return False

        before_count = self._store.count()
        deleted_chunks = self._store.delete_source(source_name)
        after_count = self._store.count()
        deleted = deleted_chunks > 0 or after_count < before_count

        if after_count == 0 and RESET_COLLECTION_WHEN_EMPTY:
            self._store.reset_collection()
            after_count = self._store.count()

        self._retriever.refresh()

        self._logger.info(
            "Delete embeddings | source=%s | deleted=%s | deleted_chunks=%s | chunks_before=%s | chunks_after=%s",
            source_name,
            deleted,
            deleted_chunks,
            before_count,
            after_count,
        )
        return deleted

    def delete_all_embeddings(self) -> int:
        """Delete all embeddings from the knowledge store."""
        before_count = self._store.count()
        self._store.reset_collection()
        after_count = self._store.count()
        
        self._retriever.refresh()
        
        self._logger.info(
            "Delete all embeddings | chunks_before=%s | chunks_after=%s | deleted=%s",
            before_count,
            after_count,
            before_count - after_count,
        )
        return before_count - after_count

    def save_document_file(self, filename: str, content: str) -> bool:
        """Save an uploaded document file to the documents directory."""
        filename = filename.strip()
        if not filename:
            return False

        # Ensure .txt extension
        if not filename.lower().endswith(".txt"):
            filename += ".txt"

        file_path = self._documents_dir / filename
        self._documents_dir.mkdir(parents=True, exist_ok=True)

        try:
            file_path.write_text(content, encoding="utf-8")
            self._logger.info("Saved document file | filename=%s | path=%s | size=%s bytes", filename, file_path, len(content))
            return True
        except Exception as ex:
            self._logger.error("Failed to save document file | filename=%s | error=%s", filename, ex)
            return False

    def delete_document_file(self, source: str) -> bool:
        """Delete a document file from the documents directory."""
        source_name = source.strip()
        if not source_name:
            return False

        file_path = self._documents_dir / source_name
        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            self._logger.info("Deleted document file | source=%s | path=%s", source_name, file_path)
            return True
        except Exception as ex:
            self._logger.error("Failed to delete document file | source=%s | error=%s", source_name, ex)
            return False

    def get_document_file_content(self, source: str) -> str | None:
        """Read document file content for download."""
        source_name = source.strip()
        if not source_name:
            return None

        file_path = self._documents_dir / source_name
        if not file_path.exists():
            return None

        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as ex:
            self._logger.error("Failed to read document file | source=%s | error=%s", source_name, ex)
            return None

    def search(self, query: str) -> list[RetrievedChunk]:
        hits = self._retriever.search(query)
        for idx, hit in enumerate(hits, start=1):
            print(
                f"[Knowledge Hit {idx}] source={hit.chunk.source} lines={hit.chunk.start_line}-{hit.chunk.end_line} "
                f"dense={hit.dense_score:.3f} sparse={hit.sparse_score:.3f} hybrid={hit.score:.3f}"
            )
            print(hit.chunk.text)
            print("-" * 80)
        return hits

    def has_index(self) -> bool:
        return self._store.count() > 0

    def get_document_statuses(self) -> list[DocumentStatus]:
        documents = load_txt_documents(self._documents_dir)
        chunks = self._store.get_all_chunks()

        chunk_counts_by_source: dict[str, int] = {}
        hashes_by_source: dict[str, set[str]] = {}

        for chunk in chunks:
            chunk_counts_by_source[chunk.source] = chunk_counts_by_source.get(chunk.source, 0) + 1
            hashes_by_source.setdefault(chunk.source, set()).add(chunk.content_hash)

        statuses: list[DocumentStatus] = []
        for document in documents:
            source_hashes = hashes_by_source.get(document.source, set())
            embedded = bool(source_hashes)
            up_to_date = document.content_hash in source_hashes
            statuses.append(
                DocumentStatus(
                    source=document.source,
                    embedded=embedded,
                    up_to_date=up_to_date,
                    chunk_count=chunk_counts_by_source.get(document.source, 0),
                )
            )

        statuses.sort(key=lambda item: item.source.lower())
        return statuses

    def get_index_stats(self) -> dict[str, int]:
        statuses = self.get_document_statuses()
        return {
            "documents_total": len(statuses),
            "documents_embedded": sum(1 for item in statuses if item.embedded),
            "documents_outdated": sum(1 for item in statuses if item.embedded and not item.up_to_date),
            "documents_not_embedded": sum(1 for item in statuses if not item.embedded),
            "chunks": self._store.count(),
        }

    def log_query(self, question: str, hits: list[RetrievedChunk], answer: str) -> None:
        hit_lines = [
            (
                f"source={hit.chunk.source} lines={hit.chunk.start_line}-{hit.chunk.end_line} "
                f"dense={hit.dense_score:.3f} sparse={hit.sparse_score:.3f} hybrid={hit.score:.3f}"
            )
            for hit in hits
        ]
        self._logger.info("Query=%s", question)
        self._logger.info("Hits=%s", " | ".join(hit_lines) if hit_lines else "none")
        self._logger.info("Answer=%s", answer)
        self._logger.info("%s", "=" * 120)

    @staticmethod
    def context_from_hits(hits: list[RetrievedChunk]) -> str:
        lines: list[str] = []
        for index, hit in enumerate(hits, start=1):
            lines.append(
                f"[{index}] Source: {hit.chunk.source} (lines {hit.chunk.start_line}-{hit.chunk.end_line})"
            )
            lines.append(hit.chunk.text)
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _group_hashes(chunks: list[ChunkRecord]) -> dict[str, set[str]]:
        grouped: dict[str, set[str]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.source, set()).add(chunk.content_hash)
        return grouped

    def _rebuild_full_index(self, documents: list) -> dict[str, int]:
        all_chunks: list[ChunkRecord] = []
        for document in documents:
            all_chunks.extend(
                chunk_document(document, KNOWLEDGE_CHUNK_SIZE, KNOWLEDGE_CHUNK_OVERLAP)
            )

        self._store.reset_collection()

        if all_chunks:
            embeddings = self._embedder.embed_texts([chunk.text for chunk in all_chunks])
            self._validate_embedding_dimensions(embeddings)
            self._store.upsert_chunks(all_chunks, embeddings)

        indexed_count = self._retriever.refresh()
        statuses = self.get_document_statuses()
        embedded_documents = sum(1 for status in statuses if status.embedded)

        self._logger.info(
            "Full rebuild complete | total_docs=%s | embedded_docs=%s | total_chunks=%s",
            len(documents),
            embedded_documents,
            indexed_count,
        )

        return {
            "documents_total": len(documents),
            "documents_embedded": embedded_documents,
            "documents_ingested_now": len(documents),
            "documents_skipped": 0,
            "chunks": indexed_count,
        }

    @staticmethod
    def _validate_embedding_dimensions(embeddings: list[list[float]]) -> None:
        if not embeddings:
            return
        expected_dim = EMBEDDING_DIMENSION
        actual_dim = len(embeddings[0])
        if actual_dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}. "
                "Update EMBEDDING_DIMENSION or switch embedding model."
            )


def _build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("knowledge_chatbot")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
