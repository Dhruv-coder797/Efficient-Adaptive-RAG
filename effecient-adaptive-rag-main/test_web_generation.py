"""
Tests for web-search generation improvements.

Verifies that:
- multiple search results are preserved as SEPARATE documents (not collapsed),
- titles/sources/dates are carried as metadata,
- results are ordered chronologically when timestamps exist,
- the generate node routes to the dedicated web chain and passes every doc.

No live API: the Tavily tool and chain flyweights are monkey-patched.
"""
import os
from unittest.mock import patch

from langchain.schema import Document

from graph.nodes.web_search import _extract_domain, _sort_chronologically, web_search
from graph.nodes import generate as generate_module


def _is_web_doc(doc):
    return bool((getattr(doc, "metadata", None) or {}).get("web"))


def test_web_search_preserves_separate_documents():
    state = {"question": "Top AI news today", "documents": []}
    results = [
        {"title": "Story A", "url": "https://www.cnn.com/a", "content": "AAA", "published_date": "2026-08-07T10:00:00Z"},
        {"title": "Story B", "url": "https://www.bbc.com/b", "content": "BBB", "published_date": "2026-08-07T09:00:00Z"},
        {"title": "Story C", "url": "https://reuters.com/c", "content": "CCC", "published_date": ""},
    ]
    fake_tool = type("T", (), {"invoke": lambda self, d: {"results": results}})()
    with patch("graph.nodes.web_search.web_search_tool", fake_tool):
        out = web_search(state)

    docs = out["documents"]
    assert len(docs) == 3, "each result must stay a separate document"
    assert all(_is_web_doc(d) for d in docs)
    # Chronologically sorted: B (09:00) before A (10:00), then undated C.
    titles = [d.metadata["title"] for d in docs]
    assert titles == ["Story B", "Story A", "Story C"]


def test_web_search_chronological_ordering():
    # B is newest, A older; undated C stays last.
    results = [
        {"title": "old", "url": "https://x.com/old", "content": "1", "published_date": "2026-08-01T00:00:00Z"},
        {"title": "new", "url": "https://x.com/new", "content": "2", "published_date": "2026-08-07T12:00:00Z"},
        {"title": "undated", "url": "https://x.com/u", "content": "3", "published_date": ""},
    ]
    ordered = _sort_chronologically(results)
    titles = [r["title"] for r in ordered]
    assert titles == ["old", "new", "undated"]


def test_domain_extraction():
    assert _extract_domain("https://www.cnn.com/a") == "cnn.com"
    assert _extract_domain("https://reuters.com/x") == "reuters.com"


def test_generate_uses_web_chain_when_web_docs_present():
    docs = [Document(page_content="kb", metadata={}),
            Document(page_content="Headline: H\ncontent\nSource: cnn\nPublished: x",
                     metadata={"web": True, "title": "H"})]
    state = {"question": "news", "documents": docs, "metadata": {"llm_calls": 0}}

    with patch.object(generate_module, "web_generation_chain") as web_chain, \
         patch.object(generate_module, "generation_chain") as rag_chain, \
         patch.object(generate_module, "increment_llm_calls") as inc:
        web_chain.invoke.return_value = "1. H ..."
        rag_chain.invoke.return_value = "rag answer"
        out = generate_module.generate(state)

    web_chain.invoke.assert_called_once()
    rag_chain.invoke.assert_not_called()
    assert "Headline: H" in web_chain.invoke.call_args.args[0]["context"]
    inc.assert_called_once()


def test_generate_uses_rag_chain_without_web_docs():
    docs = [Document(page_content="kb content", metadata={})]
    state = {"question": "q", "documents": docs, "metadata": {"llm_calls": 0}}
    with patch("graph.nodes.generate.web_generation_chain"), \
         patch.object(generate_module, "generation_chain") as rag_chain:
        rag_chain.invoke.return_value = "agg answer"
        generate_module.generate(state)
    rag_chain.invoke.assert_called_once()


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