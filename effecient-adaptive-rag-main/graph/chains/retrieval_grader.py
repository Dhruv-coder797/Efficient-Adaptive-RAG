"""
Retrieval grading chain for document relevance validation.

This chain evaluates whether retrieved documents are actually relevant to the user's question.
It's a crucial quality control mechanism in RAG because vector similarity alone doesn't
guarantee relevance — documents might be semantically similar but contextually inappropriate.

Checks for both explicit keywords and semantic meaning to provide comprehensive
relevance assessment. Prevents irrelevant documents from contaminating the generation process.

Used after retrieval to filter documents before generation.
If retrieved documents are insufficient, triggers web search as fallback.

Architecture:
- Uses a provider-agnostic structured output abstraction
- Returns a binary score (yes/no) for whether document is relevant
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm_provider import get_structured_llm


class GradeDocuments(BaseModel):
    """
    Binary grading output for document relevance assessment.
    
    Represents whether a retrieved document is relevant to the user's question.
    """
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# Create a structured LLM for document grading using the provider abstraction
structured_llm_grader = get_structured_llm(GradeDocuments)

# System prompt for retrieval grading
system = """
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
"""

# Prompt template for document grading
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Complete retrieval grading chain
retrieval_grader = grade_prompt | structured_llm_grader

"""
Quality filter for vector retrieval results.

The retrieval grader ensures that only relevant documents are used for generation.
This is essential because vector databases may return semantically similar documents
that are not actually relevant to answering the specific question.

The grader looks for:
1. Explicit keywords from the question
2. Semantic meaning and context
3. Relevance to the specific query context

When documents are insufficient (too many irrelevant), the system escalates to web search.
"""
