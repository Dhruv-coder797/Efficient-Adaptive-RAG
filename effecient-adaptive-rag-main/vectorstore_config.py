import os
from pathlib import Path

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", ".chroma"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag-chroma")


class VectorStoreNotFoundError(RuntimeError):
    """Raised when the persisted vector store has not been created yet."""


def validate_vectorstore_files() -> None:
    if not CHROMA_PATH.exists() or not (CHROMA_PATH / "chroma.sqlite3").exists():
        raise VectorStoreNotFoundError(
            f"Persisted Chroma vector database was not found at '{CHROMA_PATH}'. "
            "Run `python ingestion.py` first to build the knowledge base."
        )


def validate_vectorstore_collection() -> None:
    validate_vectorstore_files()

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception as exc:
        raise VectorStoreNotFoundError(
            f"Persisted Chroma collection '{CHROMA_COLLECTION_NAME}' could not "
            f"be loaded from '{CHROMA_PATH}'. Run `python ingestion.py` first "
            "to build the knowledge base."
        ) from exc

    if collection.count() == 0:
        raise VectorStoreNotFoundError(
            "Persisted Chroma vector database exists but contains no documents. "
            "Run `python ingestion.py` to rebuild the knowledge base."
        )
