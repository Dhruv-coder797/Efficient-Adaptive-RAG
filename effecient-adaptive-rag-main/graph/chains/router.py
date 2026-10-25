"""
Router chain for three-way adaptive query routing.

This chain classifies incoming queries into one of three categories:
- llm: Direct LLM response (general knowledge)
- vectorstore: RAG with document retrieval
- websearch: Web search for current information

The router is the critical first decision point in the adaptive RAG system.
Its accuracy directly impacts system efficiency and response quality.

Architecture:
- Uses a dedicated routing prompt with explicit decision criteria
- Leverages provider-agnostic structured output abstraction
- Returns a RouteQuery Pydantic model with the classification
"""

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm_provider import get_structured_llm
from graph.chains.router_prompts import ROUTING_PROMPT_SYSTEM


class RouteQuery(BaseModel):
    """
    Structured output model for routing decisions.
    
    Represents the router's classification of a user query into one of three paths.
    Used by the adaptive routing system to determine the optimal execution path.
    """
    datasource: Literal["llm", "vectorstore", "websearch"] = Field(..., 
        description="Given a user question, route it to llm (direct response), vectorstore (RAG), or websearch (web search)."),


# Create a structured LLM for routing decisions using the provider abstraction
structured_llm_router = get_structured_llm(RouteQuery)

# Routing prompt with explicit decision criteria and many examples
route_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTING_PROMPT_SYSTEM),
    ("human", "{question}"),
])

# Complete routing chain: prompt → structured LLM → RouteQuery
question_router = route_prompt | structured_llm_router



"""
The query router is the system's first decision point that classifies incoming queries into one of three paths:

1. **Direct LLM**: For questions answerable from general knowledge without retrieval
   - Greetings, small talk, general conversation
   - Creative writing, brainstorming, coding help
   - Generic explanations and model knowledge
   
2. **Vector Store (RAG)**: For questions requiring specific documents from the knowledge base
   - Questions about uploaded documents and materials
   - Specific technical topics covered in the knowledge base
   - Document-dependent research questions

3. **Web Search**: For current/live information unavailable in local knowledge
   - Recent news, live information, real-time facts
   - Topics not in the local knowledge base

We define a RouteQuery Pydantic model that constrains the router's output to "llm", "vectorstore", or "websearch", ensuring reliable parsing.
The system prompt clearly defines the classification criteria and knowledge base scope.
This intelligent routing optimizes resource usage by avoiding unnecessary retrieval or web searches when the model can answer directly.
"""    
