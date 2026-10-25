from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def _extract_domain(url: str) -> str:
    """Return the human-readable site name (e.g. 'cnn.com') from a URL."""
    try:
        host = urlparse(url).netloc
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    """Best-effort parse of an ISO-8601 publication timestamp."""
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _sort_chronologically(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Order items by publication date when timestamps exist.

    Dated items are sorted ascending (oldest first). Undated items keep their
    original search relevance order and are placed after the dated ones, so the
    generation step receives a stable, chronological layout.
    """
    dated = [(it, _parse_date(it.get("published_date"))) for it in items]
    with_date = [(it, dt) for it, dt in dated if dt is not None]
    without_date = [it for it, dt in dated if dt is None]
    with_date.sort(key=lambda pair: pair[1])
    return [it for it, _ in with_date] + without_date


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]

    documents = state.get("documents", [])

    tavily_results = web_search_tool.invoke({"query": question})["results"]

    # Debug: show how many results were retrieved and their headlines.
    print(f"Retrieved search results: {len(tavily_results)}")
    print("Titles:")
    for i, result in enumerate(tavily_results, 1):
        title = (result.get("title") or result.get("content", "")[:80])
        print(f"  - {i}. {title}")

    ordered = _sort_chronologically(tavily_results)

    # Preserve each search result as its OWN document so the generator can
    # produce one section per story instead of one synthesized paragraph.
    web_documents = []
    for result in ordered:
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        content = result.get("content", "").strip()
        published = result.get("published_date") or ""
        source = _extract_domain(url) or result.get("source", "")

        body = content
        if title:
            body = f"Headline: {title}\n{body}"
        if source:
            body = f"{body}\nSource: {source}"
        if published:
            body = f"{body}\nPublished: {published}"

        web_documents.append(
            Document(
                page_content=body,
                metadata={
                    "web": True,
                    "title": title,
                    "url": url,
                    "source": source,
                    "published_date": published,
                },
            )
        )

    # Append the per-story documents to any existing (RAG) documents.
    if documents:
        documents.extend(web_documents)
    else:
        documents = web_documents

    return {"documents": documents, "question": question}