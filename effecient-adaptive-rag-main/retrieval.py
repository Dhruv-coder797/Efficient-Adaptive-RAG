from langchain_chroma import Chroma
from chromadb.config import Settings

from embedding_provider import embed_model
from vectorstore_config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    validate_vectorstore_collection,
)


def _validate_chroma_store() -> None:
    validate_vectorstore_collection()


def load_vectorstore() -> Chroma:
    _validate_chroma_store()

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embed_model,
        persist_directory=str(CHROMA_PATH),
        client_settings=Settings(
            is_persistent=True,
            persist_directory=str(CHROMA_PATH),
            anonymized_telemetry=False,
        ),
        create_collection_if_not_exists=False,
    )


def get_retriever():
    return load_vectorstore().as_retriever()


# Reusable vector store instance (also used by the score-based retriever).
vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever()


def retrieve_with_scores(query: str, k: int = 4):
    """
    Retrieve documents together with their Chroma relevance scores.

    Higher scores mean higher relevance. Used by FAST_MODE to filter documents
    by a similarity threshold instead of paying for an LLM grader call.

    Args:
        query: The user's question
        k: Number of documents to retrieve

    Returns:
        list[tuple[Document, float]]: (document, relevance_score) pairs,
        sorted by relevance descending.
    """
    return vectorstore.similarity_search_with_relevance_scores(query, k=k)
