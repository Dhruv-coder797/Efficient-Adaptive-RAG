"""
Unit tests for retrieval grading resilience.

Verifies that the document grading node never crashes on malformed structured
output (None, strings, AIMessage content, exceptions) and always extracts a
usable verdict or falls back gracefully.

No external API calls: retrieval_grader is monkey-patched.
"""

from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

from graph.nodes.grade_documents import _score_to_string, grade_documents


class FakeGrader:
    """Stub replacement for the retrieval_grader chain with a scripted invoke."""

    def __init__(self, result, raise_exc=False):
        self._result = result
        self._raise_exc = raise_exc
        self.call_count = 0

    def invoke(self, *args, **kwargs):
        self.call_count += 1
        if self._raise_exc:
            raise self._raise_exc
        if isinstance(self._result, list):
            return self._result.pop(0)
        return self._result


class FakeDoc:
    def __init__(self, content):
        self.page_content = content


def make_state(question, docs):
    return {"question": question, "documents": docs, "web_search": False}


def test_score_to_string_pydantic():
    assert _score_to_string(SimpleNamespace(binary_score="yes")) == "yes"


def test_score_to_string_plain_string():
    assert _score_to_string("no") == "no"


def test_score_to_string_aimessage_content():
    assert _score_to_string(AIMessage(content="yes")) == "yes"


def test_score_to_string_none():
    assert _score_to_string(None) is None


def test_score_to_string_json_dump():
    assert _score_to_string('{"binary_score": "yes"}') == "yes"


def test_grade_documents_none_output_keeps_doc_and_no_crash():
    state = make_state("What is planning?", [FakeDoc("planning content")])
    grader = FakeGrader(None)
    with patch("graph.nodes.grade_documents.retrieval_grader", new=grader):
        result = grade_documents(state)
    assert grader.call_count == 1
    assert result["web_search"] is False
    assert len(result["documents"]) == 1  # graceful keep, no crash


def test_grade_documents_exception_keeps_doc_and_no_crash():
    state = make_state("What is memory?", [FakeDoc("memory content")])
    grader = FakeGrader(None, raise_exc=RuntimeError("rate limited"))
    with patch("graph.nodes.grade_documents.retrieval_grader", new=grader):
        result = grade_documents(state)
    assert grader.call_count == 1
    assert result["web_search"] is False
    assert len(result["documents"]) == 1


def test_grade_documents_no_grade_triggers_web_search():
    state = make_state("What is planning?", [FakeDoc("planning content")])
    grader = FakeGrader(SimpleNamespace(binary_score="no"))
    with patch("graph.nodes.grade_documents.retrieval_grader", new=grader):
        result = grade_documents(state)
    assert grader.call_count == 1
    assert result["web_search"] is True
    assert result["documents"] == []


def test_grade_documents_yes_keeps_doc():
    state = make_state("What is planning?", [FakeDoc("planning content")])
    grader = FakeGrader(SimpleNamespace(binary_score="yes"))
    with patch("graph.nodes.grade_documents.retrieval_grader", new=grader):
        result = grade_documents(state)
    assert grader.call_count == 1
    assert result["web_search"] is False
    assert len(result["documents"]) == 1


def test_grade_documents_mixed_grades():
    state = make_state(
        "What is planning?",
        [
            FakeDoc("planning content"),
            FakeDoc("pizza recipe, irrelevant"),
        ],
    )
    grader = FakeGrader([
        SimpleNamespace(binary_score="yes"),
        SimpleNamespace(binary_score="no"),
    ])
    with patch("graph.nodes.grade_documents.retrieval_grader", new=grader):
        result = grade_documents(state)
    assert grader.call_count == 2
    assert len(result["documents"]) == 1
    assert result["web_search"] is True


if __name__ == "__main__":
    import sys
    import traceback

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