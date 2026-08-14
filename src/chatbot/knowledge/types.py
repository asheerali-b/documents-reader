from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    source: str
    text: str
    content_hash: str


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source: str
    chunk_index: int
    start_line: int
    end_line: int
    content_hash: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: ChunkRecord
    score: float
    dense_score: float
    sparse_score: float


@dataclass(frozen=True)
class DocumentStatus:
    source: str
    embedded: bool
    up_to_date: bool
    chunk_count: int
