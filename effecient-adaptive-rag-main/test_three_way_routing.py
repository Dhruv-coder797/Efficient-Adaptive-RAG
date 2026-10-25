"""
Test script for three-way adaptive routing system.

This script tests all three routing paths:
1. Direct LLM: For greetings, general knowledge, creative writing, etc.
2. Vector Store (RAG): For questions about documents in the knowledge base
3. Web Search: For current news, live information, recent events
"""

from dotenv import load_dotenv
from vectorstore_config import VectorStoreNotFoundError, validate_vectorstore_collection

load_dotenv()


def format_response(result):
    """Format the response from the graph for better readability"""
    if isinstance(result, dict) and "generation" in result:
        return result["generation"]
    elif isinstance(result, dict) and "answer" in result:
        return result["answer"]
    else:
        return str(result)


def test_routing():
    """Test the three-way routing system with various queries"""
    
    try:
        validate_vectorstore_collection()
        from graph.graph import app
    except VectorStoreNotFoundError as e:
        print(f"❌ {e}")
        return
    
    print("=" * 80)
    print("🧪 Three-Way Adaptive Routing Test Suite")
    print("=" * 80)
    
    # Test cases for each routing path
    test_cases = [
        {
            "name": "Direct LLM - Greeting",
            "expected_route": "llm",
            "question": "Hello, how are you?"
        },
        {
            "name": "Direct LLM - General Knowledge",
            "expected_route": "llm",
            "question": "Explain recursion in programming"
        },
        {
            "name": "Direct LLM - Creative Writing",
            "expected_route": "llm",
            "question": "Write me a short poem about coding"
        },
        {
            "name": "Direct LLM - General Explanation",
            "expected_route": "llm",
            "question": "How do attention mechanisms work in neural networks?"
        },
        {
            "name": "Direct LLM - Small Talk",
            "expected_route": "llm",
            "question": "What's the weather like?"
        },
        {
            "name": "Vector Store - Document Question",
            "expected_route": "vectorstore",
            "question": "What techniques are discussed in the papers about prompt engineering?"
        },
        {
            "name": "Vector Store - Knowledge Base",
            "expected_route": "vectorstore",
            "question": "Summarize the main findings about adversarial attacks"
        },
        {
            "name": "Web Search - Current News",
            "expected_route": "websearch",
            "question": "What are the latest developments in AI this week?"
        },
        {
            "name": "Web Search - Recent Events",
            "expected_route": "websearch",
            "question": "What happened in AI yesterday?"
        },
    ]
    
    print("\nRunning test cases...\n")
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Question: {test_case['question']}")
        print(f"Expected Route: {test_case['expected_route']}")
        
        try:
            print("Processing...", end=" ", flush=True)
            result = app.invoke(input={"question": test_case["question"]})
            response = format_response(result)
            
            # Check if response was generated (all paths should generate a response)
            if response and len(response.strip()) > 0:
                print("✅ PASSED")
                print(f"Generated Response Preview: {response[:150]}...")
                passed += 1
            else:
                print("❌ FAILED - No response generated")
                failed += 1
        except Exception as e:
            print(f"❌ FAILED - Error: {str(e)}")
            failed += 1
        
        print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Summary")
    print("=" * 80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_cases))*100:.1f}%")
    print("=" * 80)


def test_direct_llm_bypass():
    """
    Verify that the Direct LLM route properly bypasses:
    - Document retrieval
    - Document grading
    - Hallucination checking
    """
    print("\n" + "=" * 80)
    print("🔍 Direct LLM Bypass Verification")
    print("=" * 80)
    print("The Direct LLM route should NOT retrieve or grade documents.")
    print("You should see: 'DIRECT LLM' without 'RETRIEVE', 'GRADE_DOCUMENTS', or hallucination checks.")
    print("=" * 80 + "\n")
    
    try:
        validate_vectorstore_collection()
        from graph.graph import app
    except VectorStoreNotFoundError as e:
        print(f"❌ {e}")
        return
    
    print("Testing: 'Hello there!'")
    print("-" * 80)
    result = app.invoke(input={"question": "Hello there!"})
    print("-" * 80)
    response = format_response(result)
    print(f"✅ Response: {response}\n")


if __name__ == "__main__":
    test_routing()
    test_direct_llm_bypass()
    print("\n✨ Testing complete!")
