from dotenv import load_dotenv
from vectorstore_config import VectorStoreNotFoundError, validate_vectorstore_collection

load_dotenv()

def format_response(result):
    """
    Format the response from the graph for better readability.
    Handles both legacy format (dict with generation) and new format (ExecutionResult).
    """
    if isinstance(result, dict):
        if "generation" in result:
            return result["generation"]
        elif "answer" in result:
            return result["answer"]
    return str(result)
    
    
def main():
    try:
        validate_vectorstore_collection()
        from graph.graph import execute_with_metadata
    except VectorStoreNotFoundError as e:
        print(f"❌ {e}")
        return

    print("=" * 60)
    print("🤖 Agentic Adaptive RAG")
    print("=" * 60)
    print("Welcome! Ask me anything or type 'quit', 'exit' or 'bye' to stop.")
    print("-" * 60)
    
    while True:
        try:
            # Get User Input
            user_question = input("\n💬 You: ").strip()
            
            # Check for exit commands
            if user_question.lower() in ["quit", "exit", "bye", "q"]:
                print("\n👋 Goodbye! Thanks for chatting!")
                break
            
            # Skip empty inputs
            if not user_question:
                print("Please enter a question.")
                continue
            
            # Show processing indicator
            print("\n🤔 Bot: Thinking...")
            
            # Process the question through the graph with metadata tracking
            result = execute_with_metadata(user_question)
            
            # Extract and display the response
            response = format_response(result)
            print(f"🤖 Bot: {response}")
            
            # Optionally display metadata (can be extended for debugging)
            # Uncomment to show metadata:
            # metadata = result.get("metadata", {})
            # print(f"\n📊 Metadata: Route={metadata.get('route')}, Time={metadata.get('execution_time_seconds'):.2f}s")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"\n❌ Sorry, I encountered an error: {str(e)}")
            print("Please try asking your question again.")
            
if __name__ == "__main__":
    main()
