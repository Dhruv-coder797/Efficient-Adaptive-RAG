from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm_provider import get_llm

llm = get_llm()

client = Client()

prompt = client.pull_prompt(
    "rlm/rag-prompt",
    include_model=False,
    dangerously_pull_public_prompt=True,
)

generation_chain = prompt | llm | StrOutputParser()


# ============================================================================
# Dedicated WEB SEARCH generation chain
# ============================================================================
# Unlike RAG, web search must NOT collapse all retrieved results into one
# synthesized paragraph. Each retrieved story is preserved as its own
# document and rendered as its own numbered section.
# ============================================================================

WEB_GENERATION_SYSTEM = """You are a factual news briefing assistant.

You are given a user question and a set of retrieved web search results. Each
result stands on its own and starts with a "Headline:", followed by optional
"Source:" and "Published:" lines.

RULES:
1. Write ONE numbered section per retrieved result. NEVER merge multiple
   stories into a single synthesized paragraph.
2. Format each story as:
   {index}. {Headline}
      {2-4 factual sentences summarizing the story}
      ({Source}, {Published})
3. Base the summary ONLY on the content provided. Prefer factual,
   objective statements over creative rewriting or added interpretation.
4. Mention the publication/source name whenever a Source is provided.
5. Keep the results in the order they are given to you (they are already
   ordered chronologically when timestamps exist).
6. If a result has no usable content, you may skip it, but do not merge it
   into another story's section.
"""

web_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WEB_GENERATION_SYSTEM),
        ("human", "User question: {question}\n\nSearch results:\n{context}"),
    ]
)

web_generation_chain = web_prompt | llm | StrOutputParser()