"""
Lightweight LLM-invocation counter for RAG execution statistics.

Each node that performs an LLM call increments the counter stored in the graph
metadata dictionary (metadata["llm_calls"]), so the final execution summary can
report exactly how many inference calls the pipeline made.
"""


def increment_llm_calls(metadata: dict) -> None:
    """Increment the llm_calls counter stored in the metadata dict."""
    metadata["llm_calls"] = metadata.get("llm_calls", 0) + 1


def get_llm_calls(metadata: dict) -> int:
    return metadata.get("llm_calls", 0)