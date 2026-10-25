RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"
DIRECT_LLM = "direct_llm"

"""
These constants define the names of our graph nodes and help maintain consistency throughout our codebase. 
Having centralized constants makes it easier to refactor and reduces the risk of typos when referencing node names in our workflow definitions.

Node types:
- DIRECT_LLM: Direct response generation without retrieval or grading
- RETRIEVE: Retrieve documents from vector store
- GRADE_DOCUMENTS: Grade retrieved documents for relevance
- GENERATE: Generate response from documents
- WEBSEARCH: Search the web for additional information
"""