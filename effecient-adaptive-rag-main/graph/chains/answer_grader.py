"""
Answer grading chain for RAG quality validation.

This chain evaluates whether a generated response actually addresses the user's question.
It's a critical part of the RAG pipeline's quality control, ensuring that responses are
not only factually grounded in documents but also relevant to what was asked.

Used in the adaptive RAG system after generation to validate response quality.
If the answer doesn't address the question, the system can trigger web search
or retry generation with better context.

Architecture:
- Uses a provider-agnostic structured output abstraction
- Returns a binary score (yes/no) for whether the answer addresses the question
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from llm_provider import get_structured_llm


class GradeAnswer(BaseModel):
    """
    Binary grading output for answer quality assessment.
    
    Represents whether a generated response actually addresses the user's question.
    """
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# Create a structured LLM for answer grading using the provider abstraction
structured_llm_grader = get_structured_llm(GradeAnswer)

# System prompt for answer evaluation
system = """
You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer addresses / resolves the question.
"""

# Prompt template for answer grading
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", 
         "User question: \n\n {question} \n\n LLM generation:\n\n{generation}"),
    ]
)

# Complete answer grading chain
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader

"""
Quality assurance component for RAG responses.

The answer grader ensures that generated responses are both:
1. Factually grounded (checked by hallucination_grader)
2. Relevant to the question (checked here)

This dual validation maintains response quality in the adaptive RAG system.
"""

