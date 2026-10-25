"""
FastAPI example for serving the Adaptive RAG system with execution metadata.

This is a production-ready example showing how to expose the RAG system
with structured metadata via an HTTP API for Streamlit or other frontends.

To run:
    pip install fastapi uvicorn
    uvicorn api_example:app --reload
    
Then query:
    curl -X POST "http://localhost:8000/query" \
         -H "Content-Type: application/json" \
         -d '{"question": "Explain machine learning"}'
"""

from fastapi import FastAPI
from pydantic import BaseModel
from graph.api import query_with_metadata
import json

app = FastAPI(title="Adaptive RAG API", version="1.0")


class QueryRequest(BaseModel):
    """HTTP request body for queries."""
    question: str


class QueryResponse(BaseModel):
    """HTTP response with generation and metadata."""
    success: bool
    generation: str
    metadata: dict


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Execute a query and return response with execution metadata.
    
    Args:
        request: Query request containing the question
        
    Returns:
        QueryResponse with generation and metadata
    """
    try:
        # Execute query with metadata
        result = query_with_metadata(request.question)
        
        return QueryResponse(
            success=True,
            generation=result.generation,
            metadata=result.metadata.model_dump()
        )
    except Exception as e:
        return QueryResponse(
            success=False,
            generation=f"Error: {str(e)}",
            metadata={"error": str(e)}
        )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
def root():
    """API documentation."""
    return {
        "name": "Adaptive RAG API",
        "version": "1.0",
        "endpoints": {
            "POST /query": "Execute a query and get response with metadata",
            "GET /health": "Health check",
            "GET /": "This documentation"
        },
        "example": {
            "request": {"question": "What is machine learning?"},
            "response": {
                "success": True,
                "generation": "Machine learning is...",
                "metadata": {
                    "route": "direct_llm",
                    "route_reason": "General knowledge question",
                    "llm_provider": "gemini",
                    "model_name": "gemini-flash-latest",
                    "execution_time_seconds": 2.34,
                    "retrieved_document_count": 0,
                    "used_web_search": False,
                    "hallucination_check_performed": False,
                    "hallucination_check_passed": None
                }
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
