from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage

from graph.chains.retrieval_grader import retrieval_grader
from graph.llm_counter import increment_llm_calls
from graph.state import GraphState


def _score_to_string(score: Any) -> Optional[str]:
    """
    Best-effort extraction of the binary relevance grade from the grader output.

    The provider's structured-output wrapper may return a variety of shapes:
      - A GradeDocuments pydantic model with a `.binary_score` attribute
      - A plain string like "yes" or "no"
      - An AIMessage whose content holds the grade
      - None (when the provider failed to produce structured output)

    Never raises; returns None when the verdict cannot be extracted.

    Args:
        score: Raw output from retrieval_grader.invoke(...)

    Returns:
        str | None: normalized lowercase 'yes'/'no', or None if unparseable.
    """
    # 1. Pydantic model with an attribute
    value = getattr(score, "binary_score", None)
    if value is None and isinstance(score, str):
        value = score
    elif value is None and isinstance(score, AIMessage):
        content = getattr(score, "content", None)
        if isinstance(content, str):
            value = content

    if value is None:
        return None

    value = str(value).strip().lower()
    # A structured object may also be dumped to JSON, e.g. '{"binary_score":"yes"}'
    if value.startswith("{") and '"binary_score"' in value:
        import json

        try:
            value = str(json.loads(value).get("binary_score", "")).strip().lower()
        except Exception:
            return None

    return value if value in {"yes", "no"} else None


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.

    If any document is not relevant, we will set a flag to run web search.

    Retrieval grading is a quality gate, not a hard dependency: if the grader
    fails to produce structured output (rate limit, unsupported tool calling,
    malformed response), we log it and conservatively keep the document so the
    vector store path can still produce an answer instead of crashing.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    metadata = state.get("metadata", {})

    filtered_docs = []
    web_search = False

    for d in documents:
        print("-" * 60)
        print("Grading document:")
        print(f"  {d.page_content[:200]}...")

        # Invoke the grader, capturing any provider failure rather than crashing.
        try:
            increment_llm_calls(metadata)
            raw = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
        except Exception as exc:
            print(f"  GRADER ERROR (not fatal): {type(exc).__name__}: {exc}")
            print("  RAW: <grader raised exception>")
            print("  PARSED: 'yes' (fallback: relevancy assumed, grader unavailable)")
            print("  DECISION: keep document")
            filtered_docs.append(d)
            continue

        # Log the raw output from the structured LLM.
        print(f"  RAW retrieval grader output: {type(raw).__name__} -> {raw!r}")

        # Parse and validate before ever dereferencing .binary_score.
        parsed: Optional[str] = _score_to_string(raw)
        print(f"  PARSED output: {parsed!r}")

        if parsed is None:
            # Graceful fallback: the provider failed to emit usable structured
            # output. Keep the document (conservative keep) so VectorStore path
            # execution succeeds rather than raising AttributeError.
            print(
                "  NO USABLE GRADE (None/malformed) "
                "-> fallback: keep document as relevant"
            )
            print("  DECISION: keep document")
            filtered_docs.append(d)
            continue

        if parsed == "yes":
            print("  DECISION: DOCUMENT RELEVANT")
            filtered_docs.append(d)
        else:
            print("  DECISION: DOCUMENT NOT RELEVANT -> mark web_search")
            web_search = True
            continue

    print("-" * 60)
    print(f"  GRADING SUMMARY: kept {len(filtered_docs)}/{len(documents)} docs, "
          f"web_search={web_search}")
    metadata["documents_filtered"] = len(documents) - len(filtered_docs)
    return {"documents": filtered_docs, "question": question, "web_search": web_search}