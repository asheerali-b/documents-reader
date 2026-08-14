import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.knowledge.knowledge_service import KnowledgeService


def main() -> None:
    service = KnowledgeService()
    stats = service.build_index()
    print(
        "Ingestion complete. "
        f"Total docs: {stats['documents_total']} | "
        f"Embedded docs: {stats['documents_embedded']} | "
        f"Ingested now: {stats['documents_ingested_now']} | "
        f"Skipped unchanged: {stats['documents_skipped']} | "
        f"Chunks: {stats['chunks']}"
    )


if __name__ == "__main__":
    main()
