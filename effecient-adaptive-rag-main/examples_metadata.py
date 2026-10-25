"""
Example usage of the Execution Metadata API.

This script demonstrates:
1. Simple query interface (console interaction)
2. Full metadata interface (programmatic/frontend usage)
3. Serialization to JSON for HTTP APIs
"""

from graph.api import query, query_with_metadata, get_metadata
import json


def example_simple_query():
    """Example 1: Simple query - get just the response."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Query (Console Interaction)")
    print("="*70)
    
    response = query("What is machine learning?")
    print(f"\nResponse:\n{response}")


def example_full_metadata():
    """Example 2: Full metadata - response + execution info."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Full Metadata Query (Frontend Ready)")
    print("="*70)
    
    result = query_with_metadata("Explain recursion in programming")
    
    print(f"\nGeneration:\n{result.generation}")
    print(f"\nMetadata:")
    print(f"  Route: {result.metadata.route}")
    print(f"  Route Reason: {result.metadata.route_reason}")
    print(f"  Provider: {result.metadata.llm_provider}")
    print(f"  Model: {result.metadata.model_name}")
    print(f"  Execution Time: {result.metadata.execution_time_seconds:.2f}s")
    print(f"  Retrieved Documents: {result.metadata.retrieved_document_count}")
    print(f"  Used Web Search: {result.metadata.used_web_search}")
    print(f"  Hallucination Check Performed: {result.metadata.hallucination_check_performed}")
    print(f"  Hallucination Check Passed: {result.metadata.hallucination_check_passed}")


def example_json_serialization():
    """Example 3: JSON serialization for HTTP APIs."""
    print("\n" + "="*70)
    print("EXAMPLE 3: JSON Serialization (HTTP API Ready)")
    print("="*70)
    
    result = query_with_metadata("Hello there!")
    
    # Get metadata as dictionary
    metadata_dict = get_metadata(result)
    
    # Create API response
    api_response = {
        "success": True,
        "response": result.generation,
        "metadata": metadata_dict
    }
    
    # Serialize to JSON
    json_response = json.dumps(api_response, indent=2)
    print(f"\nJSON Response:\n{json_response}")


def example_frontend_display():
    """Example 4: Data for frontend display."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Frontend Display Information")
    print("="*70)
    
    result = query_with_metadata("What's the latest AI news?")
    metadata = result.metadata
    
    print(f"\n📋 Frontend Display Components:")
    print(f"\n🔀 Routing Info:")
    print(f"   Selected: {metadata.route.upper()}")
    print(f"   Reason: {metadata.route_reason}")
    
    print(f"\n⚙️  Execution Info:")
    print(f"   Provider: {metadata.llm_provider}")
    print(f"   Model: {metadata.model_name}")
    print(f"   Time: {metadata.execution_time_seconds:.2f}s")
    
    print(f"\n📚 Retrieval Info:")
    print(f"   Documents Retrieved: {metadata.retrieved_document_count}")
    if metadata.retrieved_documents:
        for i, doc in enumerate(metadata.retrieved_documents[:3], 1):
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            print(f"     {i}. {preview}")
    
    print(f"\n✔️  Quality Checks:")
    print(f"   Hallucination Check: {'Not performed' if not metadata.hallucination_check_performed else f'Passed: {metadata.hallucination_check_passed}'}")
    print(f"   Answer Validation: {'Not performed' if not metadata.answer_validation_performed else f'Passed: {metadata.answer_validation_passed}'}")
    print(f"   Web Search Used: {'Yes' if metadata.used_web_search else 'No'}")
    
    print(f"\n💬 Response:")
    print(f"   {result.generation[:150]}...")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXECUTION METADATA API EXAMPLES")
    print("="*70)
    
    # Uncomment examples to run:
    
    # example_simple_query()
    # example_full_metadata()
    # example_json_serialization()
    # example_frontend_display()
    
    print("\n✅ Examples available - uncomment to run")
    print("   1. example_simple_query() - Basic console usage")
    print("   2. example_full_metadata() - Full execution metadata")
    print("   3. example_json_serialization() - HTTP API format")
    print("   4. example_frontend_display() - Frontend-ready components")
    print("\n" + "="*70 + "\n")
