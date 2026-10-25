from typing import List, TypedDict
from graph.execution_metadata import ExecutionMetadata


class GraphState(TypedDict):
    """
    Represent the state of our graph
    
    Attributes:
        question: User's input question
        generation: LLM generation/response
        web_search: Whether to add web search
        documents: List of retrieved documents
        metadata: Structured execution metadata for tracking workflow information
    """
    
    question: str
    generation: str
    web_search: bool
    documents: List[str]
    metadata: ExecutionMetadata
    
    
"""
This GraphState class acts as the central data structure that flows through every node in our graph workflow. 
The question field holds the user's input query, generation stores the LLM's response, web_search is a boolean flag that determines whether we need to search the web for additional information, and documents contains all the retrieved documents from both local and web sources.
By using TypedDict, we ensure type safety while maintaining the flexibility needed for our dynamic workflow.
"""