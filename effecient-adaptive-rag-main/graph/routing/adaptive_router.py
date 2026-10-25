"""
Efficient Adaptive Router.

Replaces the LLM-first routing decision with a cost-ordered cascade of
routing stages:

    Stage 1  Rule Router                (no LLM)  greetings / KB refs / current info
    Stage 2  Knowledge Topic Matcher    (no LLM)  matches KB topics via keywords + embeddings
    Stage 3  Current Information Recheck(no LLM)  temporal / live signal safety net
    Stage 4  Fallback LLM Router        (LLM)     only when Stages 1-3 are uncertain

Philosophy: use the cheapest decision mechanism possible. The LLM router is
the LAST decision maker, not the first.
"""

import os
import re

from graph.consts import DIRECT_LLM, RETRIEVE, WEBSEARCH

# ============================================================================
# KNOWLEDGE BASE TOPICS
# ============================================================================
# The persisted vector store is built around these topics. Any query that
# clearly belongs to one of these topics MUST go to the VECTOR STORE even if
# the LLM already knows the answer, because retrieval grounds the answer in
# the uploaded material.
# ============================================================================

KNOWLEDGE_TOPICS = {
    "AI Agents": [
        "ai agent", "ai agents", "autonomous agent", "agentic",
        "multi-agent", "multi agent", "what are agents",
    ],
    "Planning": [
        "planning", "planner", "planning in", "replanning",
    ],
    "Memory": [
        "memory", "remembering", "short-term memory", "long-term memory",
        "agent memory", "memory system",
    ],
    "Reflection": [
        "reflection", "reflect", "self-reflect", "self reflection",
        "self-reflection", "introspection",
    ],
    "Tool Use": [
        "tool use", "tool calling", "tool-calling", "using tools",
        "tools and", "tool usage", "function calling", "function-calling",
    ],
    "Agent Architectures": [
        "agent architecture", "agentic architecture", "architectures",
    ],
    "Prompt Engineering": [
        "prompt engineering", "prompt design", "prompt crafting",
        "engineering prompts",
    ],
    "Chain of Thought": [
        "chain of thought", "chain-of-thought", "cot prompting",
    ],
    "Few Shot": [
        "few-shot", "few shot", "few shot prompting",
    ],
    "Zero Shot": [
        "zero-shot", "zero shot", "zero shot prompting",
    ],
    "ReAct": [
        "react", "reasoning and acting", "react prompting",
    ],
    "Self Consistency": [
        "self consistency", "self-consistency",
    ],
    "Prompt Injection": [
        "prompt injection", "prompt-injection", "injection attacks",
    ],
    "Adversarial Attacks": [
        "adversarial attack", "adversarial attacks", "adversarial",
    ],
    "Jailbreak": [
        "jailbreak", "jail-break", "jail breaking",
    ],
    "LLM Security": [
        "llm security", "model security", "llm safety",
    ],
}

# Topics that overlap with generic language are matched last so that more
# specific topics take priority when several keyword sets hit.
_OVERLAPPING_TOPICS = {"Prompt Engineering", "ReAct", "AI Agents"}

# ============================================================================
# STAGE 1 + STAGE 3 VOCABULARIES
# ============================================================================

GREETING_PATTERNS = [
    r"\b(hello|hi|hey)\b",
    r"\bhow are you\b",
    r"\bthanks\b|\bthank you\b",
    r"\bgood (morning|afternoon|evening)\b",
    r"\bwelcome\b",
]

# Explicit references to the local knowledge base -> VECTOR STORE
KNOWLEDGE_BASE_PATTERNS = [
    r"\bknowledge ?base\b",
    r"\buploaded (document|file|paper|material)s?\b",
    r"\bdocumentation\b",
    r"\bpaper(s)?\b",
    r"\barticle(s)?\b",
    r"\bcourse material(s)?\b",
    r"\bthe document\b",
    r"\bsummariz(e|ing)\s+(the\s+)?(document|paper|article|material|uploaded|research|course)\b",
    r"\baccording to (the|our) (document|paper|article|documentation|knowledge base)\b",
    r"\bwhat does (the|our) (document|paper|article|documentation)\b",
    r"\bfrom the knowledge base\b",
]

# Current / time-sensitive / live information -> WEB SEARCH
CURRENT_INFO_PATTERNS = [
    r"\btoday('s)?\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\byesterday\b",
    r"\blatest\b",
    r"\bcurrent(ly)?\b",
    r"\bnews\b",
    r"\brecent(ly)?\b",
    r"\bbreaking\b",
    r"\blive\b",
    r"\btrending\b",
    r"\bnew(est)?\b",
    r"\bupdates?\b",
    r"\bstock (price|prices|market)?s?\b",
    r"\bprice(s)?\b",
    r"\bweather\b",
    r"\brelease(d)?\b",
    r"\bversion\b",
    r"\breal[- ]time\b",
    r"\bnow\b",
]

# ============================================================================
# HELPERS
# ============================================================================

def _normalize(text: str) -> str:
    return text.lower()


def _find_pattern(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def _print_decision(stage: str, detail: str, route: str) -> None:
    print(f"{stage}")
    print(f"  Matched {detail}")
    print(f"↓")
    print(f"{route}")
    print("-" * 50)


def _rule_match(text: str) -> tuple[str | None, str | None]:
    """
    Stage 1 keyword scan. Returns (route, reason_detail) or (None, None).
    """
    if matched := _find_pattern(r"|".join(GREETING_PATTERNS), text):
        return DIRECT_LLM, f"greeting: '{matched.strip()}'"

    for pattern in CURRENT_INFO_PATTERNS:
        if matched := _find_pattern(pattern, text):
            return WEBSEARCH, f"current information: '{matched.strip()}'"

    for pattern in KNOWLEDGE_BASE_PATTERNS:
        if matched := _find_pattern(pattern, text):
            return RETRIEVE, f"knowledge base reference: '{matched.strip()}'"

    return None, None


# ============================================================================
# STAGE 2: KNOWLEDGE TOPIC MATCHER (keywords + optional embeddings)
# ============================================================================

# Enables embedding-similarity topic matching as a SECONDARY signal.
# Keyword matching is authoritative and always on. Embeddings add noise (e.g.
# "REST" vs "ReAct") and a network call, so they are opt-in via env var.
_USE_TOPIC_EMBEDDINGS = os.getenv("EFFICIENT_ROUTER_USE_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}


def _topic_keyword_match(text: str) -> tuple[str | None, str | None]:
    """
    Deterministic keyword topic matcher.

    Returns (topic_name, matched_keyword) if the query clearly belongs to a
    knowledge base topic, else (None, None).
    """
    ordered_topics = [
        t for t in KNOWLEDGE_TOPICS
        if t not in _OVERLAPPING_TOPICS
    ] + list(_OVERLAPPING_TOPICS)

    best_topic, best_keyword = None, None
    for topic in ordered_topics:
        for keyword in KNOWLEDGE_TOPICS[topic]:
            if keyword in text:
                if best_topic is None or topic in _OVERLAPPING_TOPICS:
                    best_topic, best_keyword = topic, keyword
                if topic not in _OVERLAPPING_TOPICS:
                    return topic, keyword
    return best_topic, best_keyword


def _topic_embedding_match(text: str, threshold: float = 0.5) -> str | None:
    """
    Optional embedding-similarity topic match.

    Falls back silently to None when embeddings are unavailable or fail so the
    router degrades gracefully to keyword-only behaviour.
    """
    try:
        from embedding_provider import embed_model

        topic_texts = list(KNOWLEDGE_TOPICS.keys())
        query_embedding = embed_model.embed_query(text)
        topic_embeddings = embed_model.embed_documents(topic_texts)

        best_topic, best_score = None, -1.0
        for topic, embedding in zip(topic_texts, topic_embeddings):
            score = sum(a * b for a, b in zip(query_embedding, embedding))
            if score > best_score:
                best_topic, best_score = topic, score
        if best_score >= threshold:
            return best_topic
    except Exception:
        pass
    return None


def _knowledge_topic_match(text: str) -> tuple[str | None, str | None]:
    """
    Hybrid Stage 2: keyword matching first (fast, deterministic), then
    embedding similarity as a secondary signal when keywords are uncertain.
    """
    topic, keyword = _topic_keyword_match(text)
    if topic:
        return topic, f"matched {topic} topic ('{keyword}')"

    if _USE_TOPIC_EMBEDDINGS:
        if topic := _topic_embedding_match(text):
            return topic, f"matched {topic} topic (embedding similarity)"

    return None, None


# ============================================================================
# STAGE 3: CURRENT INFORMATION RECHECK
# ============================================================================

def _current_info_recheck(text: str) -> tuple[str | None, str | None]:
    """
    Secondary sweep for temporal / live / price signals. Primarily a safety
    net for queries whose phrasing obscures the time-sensitivity keyword but
    still clearly needs current data.
    """
    for pattern in CURRENT_INFO_PATTERNS:
        if matched := _find_pattern(pattern, text):
            return WEBSEARCH, f"current information: '{matched.strip()}'"
    return None, None


# ============================================================================
# STAGE 4: FALLBACK LLM ROUTER
# ============================================================================

def _fallback_llm_route(text: str) -> str:
    from graph.chains.router import question_router

    source = question_router.invoke({"question": text})
    return source.datasource


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================

def efficient_route_question(question: str) -> str:
    """
    Route a query through the cost-ordered cascade of routing stages.

    Returns one of: DIRECT_LLM, WEBSEARCH, RETRIEVE.
    """
    text = _normalize(question.strip())

    # --- Stage 1: Rule Router (no LLM) -------------------------------------
    route, detail = _rule_match(text)
    if route:
        _print_decision("Rule Router", detail, route)
        return route

    # --- Stage 2: Knowledge Base Topic Matcher (no LLM) --------------------
    topic, detail = _knowledge_topic_match(text)
    if topic:
        _print_decision("Knowledge Router", detail, RETRIEVE)
        return RETRIEVE

    # --- Stage 3: Current Information Recheck (no LLM) ---------------------
    route, detail = _current_info_recheck(text)
    if route:
        _print_decision("Rule Router", detail, route)
        return route

    # --- Stage 4: Fallback LLM Router (LLM) --------------------------------
    print("Fallback LLM Router")
    print("  Used because no rule matched")
    print("↓")
    route = _fallback_llm_route(question)
    print(route)
    print("-" * 50)
    return route
