"""
Retrieval pipeline configuration for the Efficient Adaptive RAG.

Two execution modes:

- FAST_MODE=true
    Retrieve → Similarity Threshold → Generate
    Uses Chroma relevance scores to filter documents; skips ALL quality
    graders. Exactly one LLM call (generation).

- FAST_MODE=false (default, STRICT_MODE)
    Preserves the existing validation pipeline:
    Retrieve → Document Grader (LLM) → Generate → Hallucination Grader (LLM)
    → Answer Grader (LLM).

Individual graders can be toggled independently via environment variables:
    ENABLE_HALLUCINATION_GRADING  (default: true, used in STRICT mode)
    ENABLE_ANSWER_GRADING         (default: true, used in STRICT mode)

This module centralises all mode logic so the graph nodes stay clean.
"""

import os


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def fast_mode() -> bool:
    """Whether the cheap Retrieve→Threshold→Generate path is active."""
    return _truthy(os.getenv("FAST_MODE", "false"))


def similarity_threshold() -> float:
    """
    Chroma relevance-score cutoff for FAST_MODE document filtering.

    Scores come from `vectorstore.similarity_search_with_relevance_scores` and
    are higher = more relevant. Returns the configured threshold.
    """
    try:
        return float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


def enable_hallucination_grading() -> bool:
    """Hallucination grading is active in STRICT mode when enabled."""
    if fast_mode():
        return False
    return _truthy(os.getenv("ENABLE_HALLUCINATION_GRADING", "true"))


def enable_answer_grading() -> bool:
    """Answer grading is active in STRICT mode when enabled."""
    if fast_mode():
        return False
    return _truthy(os.getenv("ENABLE_ANSWER_GRADING", "true"))