from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
import time
import os

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.config import (
    enable_answer_grading,
    enable_hallucination_grading,
    fast_mode,
)
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH, DIRECT_LLM
from graph.nodes.retrieve import retrieve
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.web_search import web_search
from graph.nodes.direct_llm import direct_llm
from graph.state import GraphState
from graph.execution_metadata import ExecutionMetadata
from graph.llm_counter import increment_llm_calls

load_dotenv()


# ============================================================================
# HELPER: Get provider and model info for metadata
# ============================================================================

def _get_provider_info() -> tuple[str, str]:
    """Get the current LLM provider and model name for metadata tracking."""
    from providers.factory import _selected_provider_name, _get_provider
    
    provider_name = _selected_provider_name()
    provider = _get_provider()
    
    # Get model name based on provider
    if provider_name == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    elif provider_name == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif provider_name == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.1")
    elif provider_name == "openrouter":
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    else:
        model = "unknown"
    
    return provider_name, model


# ============================================================================
# ROUTING LOGIC HELPERS
# ============================================================================

def log_route_decision(route: str, reason: str = None) -> None:
    """
    Log the adaptive router's decision with structured formatting.
    
    Args:
        route: The selected route (DIRECT_LLM, RETRIEVE, or WEBSEARCH)
        reason: Optional explanation for the routing decision
    """
    print(f"\n🔀 Adaptive Router Decision: {route}")
    if reason:
        print(f"   Reason: {reason}")


def route_question(state: GraphState) -> str:
    """
    Route question to one of three paths: direct LLM, vector store (RAG), or web search.

    This is the entry point that classifies incoming queries into:
    - "llm": Direct LLM response without retrieval
    - "vectorstore": RAG path with document retrieval and grading
    - "websearch": Web search path for current information

    Uses the cost-ordered Efficient Adaptive Router: deterministic rule stages
    first, the LLM router only as a final fallback.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call (DIRECT_LLM, RETRIEVE, or WEBSEARCH)
    """
    print("=" * 70)
    print("📋 PROCESSING NEW QUERY")
    print("=" * 70)
    question = state["question"]
    print(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
    print("-" * 70)

    from graph.routing.adaptive_router import efficient_route_question

    decision = efficient_route_question(question)

    # Track provider info and routing decision in metadata
    provider_name, model_name = _get_provider_info()
    state["metadata"]["llm_provider"] = provider_name
    state["metadata"]["model_name"] = model_name

    reason_map = {
        DIRECT_LLM: "General knowledge question (rule or fallback)",
        WEBSEARCH: "Requires current/live information",
        RETRIEVE: "Requires indexed knowledge base",
    }
    state["metadata"]["route"] = decision
    state["metadata"]["route_reason"] = reason_map[decision]
    log_route_decision(decision, reason_map[decision])
    return decision


# ============================================================================
# GENERATION QUALITY CHECKING HELPERS
# ============================================================================

def decide_to_generate(state):
    """
    Decide whether to generate with retrieved documents or perform web search.
    
    After document grading, determines if retrieved documents are relevant enough
    to proceed with generation, or if web search should supplement the answer.
    """
    print("\n📊 ASSESSING GRADED DOCUMENTS")

    if state["web_search"]:
        print("   Decision: Documents insufficient → Including web search")
        state["metadata"]["used_web_search"] = True
        return WEBSEARCH
    else:
        print("   Decision: Documents relevant → Proceeding to generation")
        return GENERATE


def after_retrieve(state):
    """
    Branch after the RETRIEVE node.

    In FAST_MODE the documents are already filtered by similarity score, so we
    skip the LLM document grader entirely and go straight to generation (or web
    search if nothing passed the threshold). In STRICT_MODE we preserve the
    existing LLM-based document grading pipeline.
    """
    if fast_mode():
        if state["web_search"]:
            print("\n📊 NO DOCS PASSED THRESHOLD → Including web search")
            state["metadata"]["used_web_search"] = True
            return WEBSEARCH
        print("\n📊 FAST_MODE: similarity-filtered docs → generation")
        return GENERATE
    return GRADE_DOCUMENTS


def after_generate(state):
    """
    Decide what to do once a response has been generated.

    In FAST_MODE we skip hallucination and answer graders and finish (one LLM
    call total). In STRICT_MODE we run the quality validators, honoring the
    ENABLE_HALLUCINATION_GRADING / ENABLE_ANSWER_GRADING switches.
    """
    if fast_mode():
        return "useful"

    if not enable_hallucination_grading() and not enable_answer_grading():
        return "useful"

    return grade_generation_grounded_in_documents_and_question(state)


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    Validate that the generated response is grounded in documents and addresses the question.
    
    Performs two checks:
    1. Hallucination check: Is generation grounded in retrieved documents?
    2. Relevance check: Does generation actually address the question?
    
    Each grader can be disabled via ENABLE_HALLUCINATION_GRADING and
    ENABLE_ANSWER_GRADING. For RAG path only (direct LLM and websearch skip
    this validation).
    """
    print("\n✔️  VALIDATING GENERATED RESPONSE")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    grounded = True
    if enable_hallucination_grading():
        # Track hallucination check
        state["metadata"]["hallucination_check_performed"] = True
        try:
            increment_llm_calls(state["metadata"])
            hallucination_score = hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
        except Exception as exc:
            print(f"   ⚠️ Hallucination grader error (treated as passed): {exc}")
            hallucination_grade = "yes"
        else:
            hallucination_grade = getattr(hallucination_score, "binary_score", None)
            print(
                f"   RAW hallucination grader output: "
                f"{type(hallucination_score).__name__} -> {hallucination_score!r}"
            )
            print(f"   PARSED hallucination grade: {hallucination_grade!r}")

        # Grader unavailable (None/malformed) is treated as grounded so the
        # vector store path completes instead of crashing or looping.
        if hallucination_grade in (None, "yes"):
            print("   ✓ Response is grounded in documents")
            state["metadata"]["hallucination_check_passed"] = True
        else:
            print("   ✗ Response not grounded in documents → Retrying")
            state["metadata"]["hallucination_check_passed"] = False
            grounded = False
    else:
        print("   ⏭️  Hallucination grading disabled")

    if not grounded:
        return "not supported"

    if enable_answer_grading():
        print("   Checking if response addresses question...")
        # Track answer validation
        state["metadata"]["answer_validation_performed"] = True
        try:
            increment_llm_calls(state["metadata"])
            answer_score = answer_grader.invoke({"question": question, "generation": generation})
        except Exception as exc:
            print(f"   ⚠️ Answer grader error (treated as passed): {exc}")
            answer_grade = "yes"
        else:
            answer_grade = getattr(answer_score, "binary_score", None)
            print(
                f"   RAW answer grader output: "
                f"{type(answer_score).__name__} -> {answer_score!r}"
            )
            print(f"   PARSED answer grade: {answer_grade!r}")

        if answer_grade in (None, "yes"):
            print("   ✓ Response addresses the question")
            state["metadata"]["answer_validation_passed"] = True
            return "useful"
        else:
            print("   ✗ Response doesn't address question → Retrying")
            state["metadata"]["answer_validation_passed"] = False
            return "not useful"
    else:
        print("   ⏭️  Answer grading disabled → response accepted")
        return "useful"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

workflow = StateGraph(GraphState)

# Add all nodes
workflow.add_node(DIRECT_LLM, direct_llm)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# Set conditional entry point based on routing decision
workflow.set_conditional_entry_point(
    route_question,
    {
        DIRECT_LLM: DIRECT_LLM,
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

# DIRECT_LLM path: goes straight to END (response already generated)
workflow.add_edge(DIRECT_LLM, END)

# RETRIEVE path: in FAST_MODE score-filtered docs go straight to generation;
# in STRICT_MODE documents are graded by the LLM before deciding.
workflow.add_conditional_edges(
    RETRIEVE,
    after_retrieve,
    {
        GRADE_DOCUMENTS: GRADE_DOCUMENTS,
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

# GENERATE path: FAST_MODE ends immediately; STRICT_MODE validates quality.
workflow.add_conditional_edges(
    GENERATE,
    after_generate,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)

# WEBSEARCH path: goes to GENERATE after getting web results
workflow.add_edge(WEBSEARCH, GENERATE)

# Compile the workflow into an executable application
app = workflow.compile()


# ============================================================================
# EXECUTION WRAPPER
# ============================================================================

def execute_with_metadata(question: str) -> dict:
    """
    Execute the adaptive RAG workflow and return both response and metadata.
    
    Args:
        question: User's input question
        
    Returns:
        Dictionary with 'generation' and 'metadata' keys
    """
    from graph.execution_metadata import ExecutionResult
    
    # Initialize execution metadata
    metadata = ExecutionMetadata()
    
    # Record start time
    start_time = time.time()
    
    # In-place dict used by nodes; seed the LLM call counter.
    metadata_dict = metadata.model_dump()
    metadata_dict["llm_calls"] = 0
    
    # Execute the graph
    result = app.invoke({
        "question": question,
        "generation": "",
        "web_search": False,
        "documents": [],
        "metadata": metadata_dict
    })
    
    # Record execution time
    execution_time = time.time() - start_time
    result["metadata"]["execution_time_seconds"] = execution_time
    _print_execution_statistics(result["metadata"], question)
    
    # Return structured result
    return {
        "generation": result.get("generation", ""),
        "metadata": result.get("metadata", metadata_dict)
    }


def _print_execution_statistics(metadata: dict, question: str = "") -> None:
    """
    Print RAG execution statistics for observability.

    Includes: fast-mode flag, documents retrieved vs filtered, number of LLM
    calls the pipeline made, and total wall-clock execution time.
    """
    print("\n" + "=" * 60)
    print("📊 EXECUTION STATISTICS")
    print("=" * 60)
    print(f"   Question: {question[:60]}{'...' if len(question) > 60 else ''}")
    print(f"   Fast Mode Enabled: {str(fast_mode()).upper()}")
    print(f"   Documents Retrieved: {metadata.get('retrieved_document_count', 0)}")
    print(f"   Documents Filtered: {metadata.get('documents_filtered', 0)}")
    print(f"   LLM Calls: {metadata.get('llm_calls', 0)}")
    print(
        "   Execution Time: "
        f"{metadata.get('execution_time_seconds', 0.0):.2f}s"
    )
    print("=" * 60)


