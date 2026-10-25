"""
Execution metadata tracking for the Adaptive Agentic RAG system.

This module defines the structure for capturing execution information
at each stage of the workflow, enabling frontend visualization and debugging.

Metadata is collected by each node and aggregated into a final structure
that can be returned alongside the generated response.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class ExecutionMetadata(BaseModel):
    """
    Structured execution metadata for adaptive RAG workflow.
    
    Captures detailed information about which route was taken,
    what processing was performed, and how long it took.
    
    Fields are populated by individual nodes as they execute.
    Unused fields for a particular route are set to sensible defaults.
    """
    
    # Routing information
    route: str = Field(
        default="unknown",
        description="Selected execution route: direct_llm, vectorstore, or websearch"
    )
    route_reason: Optional[str] = Field(
        default=None,
        description="Explanation for why this route was selected"
    )
    
    # Provider information
    llm_provider: str = Field(
        default="unknown",
        description="The LLM provider used (gemini, openai, ollama, openrouter)"
    )
    model_name: str = Field(
        default="unknown",
        description="The specific model name from the provider"
    )
    
    # Timing
    execution_time_seconds: float = Field(
        default=0.0,
        description="Total execution time for the entire workflow"
    )
    
    # Retrieval information
    retrieved_document_count: int = Field(
        default=0,
        description="Number of documents retrieved from vector store"
    )
    retrieved_documents: List[str] = Field(
        default_factory=list,
        description="The actual retrieved document contents"
    )
    
    # Web search information
    used_web_search: bool = Field(
        default=False,
        description="Whether web search was used during execution"
    )
    
    # Quality checks for RAG path
    hallucination_check_performed: bool = Field(
        default=False,
        description="Whether hallucination checking was performed"
    )
    hallucination_check_passed: Optional[bool] = Field(
        default=None,
        description="Result of hallucination check (None if not performed)"
    )
    
    # Quality checks for answer validation
    answer_validation_performed: bool = Field(
        default=False,
        description="Whether answer validation was performed"
    )
    answer_validation_passed: Optional[bool] = Field(
        default=None,
        description="Result of answer validation (None if not performed)"
    )


class ExecutionResult(BaseModel):
    """
    Final execution result combining response and metadata.
    
    This is the clean public API that frontends should consume.
    """
    
    generation: str = Field(
        description="The final generated response to the user's question"
    )
    metadata: ExecutionMetadata = Field(
        description="Structured metadata about the execution"
    )
