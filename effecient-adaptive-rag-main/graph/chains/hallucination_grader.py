"""
Hallucination grading chain for factual accuracy validation.

This chain is the critical guardrail of the RAG system. It verifies that generated
responses are grounded in the retrieved documents, preventing hallucinations
(plausible-sounding but factually incorrect information).

This is the most important quality validation in RAG, ensuring that every claim
in the response can be traced back to source documents.

Used after generation in the RAG path to validate factual accuracy.
If hallucinations are detected, the system can retry with better context
or escalate to web search for clarification.

Architecture:
- Uses a provider-agnostic structured output abstraction
- Returns a binary score (yes/no) for whether generation is grounded in facts
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from llm_provider import get_structured_llm


class GradeHallucinations(BaseModel):
    """
    Binary grading output for hallucination detection.
    
    Represents whether a generated response is grounded in retrieved facts.
    """
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# Create a structured LLM for hallucination grading using the provider abstraction
structured_llm_grader = get_structured_llm(GradeHallucinations)

# System prompt for hallucination detection
system = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

# Prompt template for hallucination checking
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# Complete hallucination detection chain
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

"""
Critical safeguard for RAG reliability.

The hallucination grader prevents the RAG system from generating misleading responses.
Every generated claim must be verifiable against retrieved documents.

When hallucinations are detected:
1. System retries generation with better context
2. If retries don't help, escalates to web search
3. Ensures users receive accurate, trustworthy responses

This is the most important quality control in the entire adaptive RAG system.
"""
