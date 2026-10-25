"""
Tests for the retrieval-pipeline optimization.

Covers:
- config parsing (FAST_MODE, progress flags)
- FAST_MODE retrieve node: similarity-threshold filtering via a mocked
  score retriever, and the LLM grader being bypassed.

No external API calls; retrieval functions are monkey-patched.
"""
import os
from unittest.mock import patch

from graph.config import (
    enable_answer_grading,
    enable_hallucination_grading,
    fast_mode,
    similarity_threshold,
)
from graph.nodes.retrieve import retrieve


def _set(env: dict):
    snapshot = {k: os.environ.get(k) for k in env}
    for k in ["FAST_MODE", "SIMILARITY_THRESHOLD",
              "ENABLE_HALLUCINATION_GRADING", "ENABLE_ANSWER_GRADING"]:
        os.environ.pop(k, None)
    os.environ.update(env)
    return snapshot


def _restore(snapshot):
    for k in ["FAST_MODE", "SIMILARITY_THRESHOLD",
              "ENABLE_HALLUCINATION_GRADING", "ENABLE_ANSWER_GRADING"]:
        if snapshot.get(k) is not None:
            os.environ[k] = snapshot[k]
        else:
            os.environ.pop(k, None)


class FakeDoc:
    def __init__(self, content):
        self.page_content = content


def test_fast_mode_flag():
    snap = _set({"FAST_MODE": "true"})
    try:
        assert fast_mode() is True
    finally:
        _restore(snap)


def test_threshold_default():
    snap = _set({})
    try:
        assert similarity_threshold() == 0.5
    finally:
        _restore(snap)


def test_threshold_custom():
    snap = _set({"SIMILARITY_THRESHOLD": "0.7"})
    try:
        assert similarity_threshold() == 0.7
    finally:
        _restore(snap)


def test_graders_disabled_when_fast():  # requirement 4 & 5 & 6
    snap = _set({"FAST_MODE": "true"})
    try:
        assert enable_hallucination_grading() is False
        assert enable_answer_grading() is False
    finally:
        _restore(snap)


def test_graders_default_on_in_strict():
    snap = _set({"FAST_MODE": "false"})
    try:
        assert enable_hallucination_grading() is True
        assert enable_answer_grading() is True
    finally:
        _restore(snap)


def test_graders_independently_toggleable_in_strict():
    snap = _set({"FAST_MODE": "false",
                 "ENABLE_HALLUCINATION_GRADING": "false",
                 "ENABLE_ANSWER_GRADING": "true"})
    try:
        assert enable_hallucination_grading() is False
        assert enable_answer_grading() is True
    finally:
        _restore(snap)


def test_fast_retrieve_filters_and_sets_web_search():
    snap = _set({"FAST_MODE": "true", "SIMILARITY_THRESHOLD": "0.5"})
    scored = [
        (FakeDoc("relevant about planning"), 0.85),
        (FakeDoc("somewhat related"), 0.6),
        (FakeDoc("totally irrelevant pizza"), 0.1),
    ]
    state = {
        "question": "What is planning?",
        "web_search": False,
        "metadata": {"retrieved_document_count": 0, "documents_filtered": 0},
    }
    try:
        with patch("graph.nodes.retrieve.retrieve_with_scores", return_value=scored):
            result = retrieve(state)
        assert len(result["documents"]) == 2
        assert result["web_search"] is False
        assert result["metadata"]["retrieved_document_count"] == 3
        assert result["metadata"]["documents_filtered"] == 1
    finally:
        _restore(snap)


def test_fast_retrieve_falls_back_to_web_search_when_all_dropped():
    snap = _set({"FAST_MODE": "true", "SIMILARITY_THRESHOLD": "0.9"})
    scored = [(FakeDoc("low relevance"), 0.2)]
    state = {
        "question": "What is planning?",
        "web_search": False,
        "metadata": {"retrieved_document_count": 0, "documents_filtered": 0},
    }
    try:
        with patch("graph.nodes.retrieve.retrieve_with_scores", return_value=scored):
            result = retrieve(state)
        assert result["documents"] == []
        assert result["web_search"] is True
        assert result["metadata"]["documents_filtered"] == 1
    finally:
        _restore(snap)


if __name__ == "__main__":
    import sys, traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)