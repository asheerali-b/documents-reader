from __future__ import annotations

import hashlib
from pathlib import Path

from src.chatbot.knowledge.types import ChunkRecord, SourceDocument


def load_txt_documents(documents_dir: Path) -> list[SourceDocument]:
    if not documents_dir.exists():
        return []

    documents: list[SourceDocument] = []
    for file_path in sorted(documents_dir.rglob("*.txt")):
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        source = str(file_path.relative_to(documents_dir)).replace("\\", "/")
        documents.append(SourceDocument(source=source, text=text, content_hash=content_hash))
    return documents


def chunk_document(document: SourceDocument, chunk_size: int, chunk_overlap: int) -> list[ChunkRecord]:
    text = document.text
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[ChunkRecord] = []
    step = chunk_size - chunk_overlap
    for index, start in enumerate(range(0, len(text), step)):
        end = start + chunk_size
        piece = text[start:end]
        if piece.strip():
            start_line = _line_number_for_offset(text, start)
            end_line = _line_number_for_offset(text, min(end, len(text)))
            chunk_id = f"{document.source}:{document.content_hash}:{index}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    source=document.source,
                    chunk_index=index,
                    start_line=start_line,
                    end_line=end_line,
                    content_hash=document.content_hash,
                    text=piece.strip(),
                )
            )
        if end >= len(text):
            break
    return chunks


def _line_number_for_offset(text: str, offset: int) -> int:
    if not text:
        return 1
    bounded_offset = max(0, min(offset, len(text)))
    return text.count("\n", 0, bounded_offset) + 1
