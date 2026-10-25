"""
Public API for executing the adaptive RAG system with metadata.

This module provides clean entry points for:
1. Console-based interaction (returns just the response)
2. Programmatic usage (returns response + structured metadata)
3. Frontend integration (returns ExecutionResult with full metadata)
"""

from graph.graph import execute_with_metadata, app
from graph.execution_metadata import ExecutionResult, ExecutionMetadata


def query(question: str) -> str:
    """
    Simple query interface - returns just the generated response.
    
    Use this for basic console interaction.
    
    Args:
        question: User's input question
        
    Returns:
        str: The generated response
        
    Example:
        response = query("What is machine learning?")
        print(response)
    """
    result = execute_with_metadata(question)
    return result.get("generation", "")


def query_with_metadata(question: str) -> ExecutionResult:
    """
    Query interface that returns both response and execution metadata.
    
    Use this for frontend integration or debugging.
    
    Args:
        question: User's input question
        
    Returns:
        ExecutionResult: Contains 'generation' and 'metadata' fields
        
    Example:
        result = query_with_metadata("Explain quantum computing")
        print(f"Response: {result.generation}")
        print(f"Route: {result.metadata.route}")
        print(f"Time: {result.metadata.execution_time_seconds}s")
    """
    result = execute_with_metadata(question)
    return ExecutionResult(
        generation=result.get("generation", ""),
        metadata=ExecutionMetadata(**result.get("metadata", {}))
    )


def get_metadata(result: ExecutionResult) -> dict:
    """
    Extract metadata from an ExecutionResult as a dictionary.
    
    Useful for serializing metadata to JSON for frontend consumption.
    
    Args:
        result: ExecutionResult from query_with_metadata()
        
    Returns:
        dict: Metadata as a dictionary
        
    Example:
        result = query_with_metadata("What is recursion?")
        metadata_dict = get_metadata(result)
        # Send to frontend as JSON
        json.dumps(metadata_dict)
    """
    return result.metadata.model_dump()
