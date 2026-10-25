from typing import Any, Dict

from graph.config import fast_mode, similarity_threshold
from graph.state import GraphState
from retrieval import retrieve_with_scores, retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    metadata = state.get("metadata", {})

    if fast_mode():
        # FAST_MODE: filter by Chroma relevance score instead of an LLM grader.
        scored = retrieve_with_scores(question, k=4)
        threshold = similarity_threshold()

        print(f"   FAST_MODE: similarity threshold = {threshold}")
        retrieved = [doc for doc, _ in scored]
        filtered = [doc for doc, score in scored if score >= threshold]
        for doc, score in scored:
            print(f"   score={score:.4f}  {'KEEP' if score >= threshold else 'DROP'}  {doc.page_content[:60]}...")

        metadata["retrieved_document_count"] = len(retrieved)
        metadata["documents_filtered"] = len(retrieved) - len(filtered)
        metadata["similarity_threshold"] = threshold

        # If nothing survives the threshold, fall back to web search.
        web_search = len(filtered) == 0
        if web_search:
            print("   FAST_MODE: no documents above threshold -> web search")

        return {
            "documents": filtered,
            "question": question,
            "web_search": web_search,
            "metadata": metadata,
        }

    # STRICT_MODE: retrieve everything, LLM grader filters later.
    documents = retriever.invoke(question)
    metadata["retrieved_document_count"] = len(documents)

    return {
        "documents": documents,
        "question": question,
        "metadata": metadata,
    }