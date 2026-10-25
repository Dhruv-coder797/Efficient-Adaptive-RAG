"""
Test the Efficient Adaptive Router's success criteria deterministically
without invoking the graph, LLM, or web search.

The last category (DIRECT LLM fallback) exercises the full cascade; for the
fallback cases that must reach the LLM router, we import and check the module.
"""

from graph.routing.adaptive_router import (
    _current_info_recheck,
    _knowledge_topic_match,
    _rule_match,
    efficient_route_question,
    DIRECT_LLM,
    RETRIEVE,
    WEBSEARCH,
)


def case_rule(question, expected):
    text = question.lower()
    route, detail = _rule_match(text)
    if route is None:
        route, detail = _current_info_recheck(text)
    return route == expected, route, detail


def test_success_criteria():
    print("=" * 60)
    print("Efficient Adaptive Router — Success Criteria")
    print("=" * 60)

    vector_cases = [
        "What is planning in AI agents?",
        "Explain Chain of Thought prompting.",
        "What is agent memory?",
        "What is ReAct?",
        "What are adversarial attacks on LLMs?",
        "What is prompt injection?",
    ]

    web_cases = [
        "What happened in AI today?",
        "What is the latest LangGraph release?",
        "What are today's NVIDIA stock prices?",
    ]

    llm_cases = [
        "Hello",
        "What is REST?",
        "Write a Python function to reverse a string.",
    ]

    failures = 0
    total = 0

    print("\n--- VECTOR STORE (Stage 1 KB ref / Stage 2 topic) ---")
    for q in vector_cases:
        total += 1
        route = _knowledge_topic_match(q.lower())
        pass_ = route[0] is not None
        print(f"  [{'PASS' if pass_ else 'FAIL'}] VECTOR -> {q!r}")
        if not pass_:
            failures += 1

    print("\n--- WEB SEARCH (Stage 1 / Stage 3 current info) ---")
    for q in web_cases:
        total += 1
        ok, route, _ = case_rule(q, WEBSEARCH)
        print(f"  [{'PASS' if ok else 'FAIL'}] WEB  -> {q!r}")
        if not ok:
            failures += 1

    print("\n--- DIRECT LLM (no rule matched -> fallback) ---")
    for q in llm_cases:
        total += 1
        _, route, _ = case_rule(q, DIRECT_LLM)
        ok = route in (None, DIRECT_LLM, "llm")
        print(f"  [{'PASS' if ok else 'FAIL'}] LLM  -> {q!r} (rule route={route or 'fallback'})")
        if not ok:
            failures += 1

    print("\n--- Full cascade smoke test (rules that hit first stages) ---")
    smoke = [
        ("What is planning in AI agents?", RETRIEVE),
        ("What is the latest LangGraph release?", WEBSEARCH),
        ("Hello", DIRECT_LLM),
    ]
    for q, want in smoke:
        total += 1
        got = efficient_route_question(q)
        ok = got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {q!r} -> {got} (want {want})")
        if not ok:
            failures += 1

    print("\n=== Summary ===")
    print(f"Total: {total}  Passed: {total - failures}  Failed: {failures}")
    return failures == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if test_success_criteria() else 1)