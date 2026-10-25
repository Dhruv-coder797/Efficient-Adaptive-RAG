from typing import Any, Dict
from graph.chains.direct_generation import direct_generation_chain
from graph.state import GraphState

def direct_llm(state: GraphState) -> Dict[str, Any]:
    """
    Generate a response directly from the LLM without retrieval.
    
    This node is used for queries that don't require document retrieval:
    - Greetings and small talk
    - General knowledge questions
    - Creative writing and brainstorming
    - Coding help and explanations
    
    This path bypasses:
    - Document retrieval
    - Document grading
    - Hallucination checking against documents
    
    Uses a dedicated generation chain optimized for direct responses
    without any retrieved document context.
    
    Args:
        state: Graph state containing the question
        
    Returns:
        Updated state with generation field populated
    """
    print("---DIRECT LLM---")
    question = state["question"]
    
    # Generate response using the dedicated direct LLM chain
    generation = direct_generation_chain.invoke({"question": question})
    
    return {"generation": generation}


"""
The direct_llm node handles queries that can be answered directly by the LLM without needing 
to retrieve documents from the vector store. This is an optimization path that improves 
response latency for common queries like greetings and general knowledge questions.

By bypassing retrieval, document grading, and hallucination checking, we reduce computational 
overhead while maintaining response quality for appropriate query types.

This node uses the dedicated direct_generation_chain which has a system prompt optimized 
for general-purpose responses, keeping it semantically distinct from the RAG generation chain.
"""
