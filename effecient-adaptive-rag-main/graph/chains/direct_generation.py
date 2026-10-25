"""
Direct LLM Generation Chain

This chain generates responses for queries that do not require document retrieval.
It's optimized for direct question-answering without any context documents.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm_provider import get_llm

llm = get_llm()

# System prompt for direct LLM responses - no mention of documents or context
DIRECT_LLM_SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant. Answer the user's question directly and clearly.

Provide accurate, concise responses based on your knowledge. If you're unsure about something, be honest about the limitations of your knowledge.

Be friendly and engaging in your responses."""

# Create the prompt template for direct responses
direct_prompt = ChatPromptTemplate.from_messages([
    ("system", DIRECT_LLM_SYSTEM_PROMPT),
    ("human", "{question}"),
])

# Chain that generates direct responses without any retrieval context
direct_generation_chain = direct_prompt | llm | StrOutputParser()


"""
The direct_generation_chain is specifically designed for answering questions without
relying on retrieved documents. It uses a simple, clean system prompt that:

1. Does NOT mention documents or retrieved context
2. Does NOT describe the LLM as constrained to specific knowledge
3. Treats the LLM as a general-purpose assistant
4. Encourages clear, accurate responses

This keeps the direct LLM path semantically different from the RAG path,
which is appropriate since they answer different types of questions.
"""
