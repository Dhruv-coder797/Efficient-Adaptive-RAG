from typing import Any, Dict

from graph.chains.generation import generation_chain, web_generation_chain
from graph.llm_counter import increment_llm_calls
from graph.state import GraphState


def _is_web_document(doc) -> bool:
    metadata = getattr(doc, "metadata", None) or {}
    return bool(metadata.get("web"))


def _document_label(doc) -> str:
    metadata = getattr(doc, "metadata", None) or {}
    title = metadata.get("title")
    if title:
        return title
    preview = getattr(doc, "page_content", "")
    return preview[:80] + ("..." if len(preview) > 80 else "")


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # Debug: verify exactly which documents reach generation.
    print(f"Documents passed to generation: {len(documents)}")
    for i, doc in enumerate(documents, 1):
        kind = "web" if _is_web_document(doc) else "kb"
        print(f"  {i}. [{kind}] {_document_label(doc)}")

    has_web_documents = any(_is_web_document(doc) for doc in documents)
    if has_web_documents:
        print("   Using WEB generation chain (one section per search result)")
        chain = web_generation_chain
        context = "\n\n".join(
            f"[{i}] {getattr(doc, 'page_content', '')}"
            for i, doc in enumerate(documents, 1)
        )
    else:
        print("   Using RAG generation chain")
        chain = generation_chain
        context = documents

    increment_llm_calls(state["metadata"])
    generation = chain.invoke({"context": context, "question": question})

    return {"documents": documents, "question": question, "generation": generation}