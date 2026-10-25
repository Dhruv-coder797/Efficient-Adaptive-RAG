from dotenv import load_dotenv
from vectorstore_config import CHROMA_COLLECTION_NAME, CHROMA_PATH

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

def load_documents():
    from langchain_community.document_loaders import WebBaseLoader

    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]


def split_documents(docs_list):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )
    return text_splitter.split_documents(docs_list)


def build_vectorstore():
    from langchain_chroma import Chroma

    from embedding_provider import embed_model

    docs_list = load_documents()
    doc_splits = split_documents(docs_list)

    print(f"Loaded documents: {len(docs_list)}")
    print(f"Number of chunks: {len(doc_splits)}")

    return Chroma.from_documents(
        documents=doc_splits,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding=embed_model,
        persist_directory=str(CHROMA_PATH),
    )


def main():
    build_vectorstore()
    print(f"Persisted Chroma vector store at {CHROMA_PATH}")


if __name__ == "__main__":
    main()


def __getattr__(name):
    if name == "retriever":
        from retrieval import retriever

        return retriever
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


"""
This ingestion pipeline forms the backbone of our local knowledge base. 
We start by loading environment variables, then define a curated list of URLs containing high-quality content about AI agents, prompt engineering, and adversarial attacks.
The WebBaseLoader fetches content from these URLs and loads them into document objects. 
We then use the RecursiveCharacterTextSplitter to break down these documents into smaller, manageable chunks of 250 tokens each, which is optimal for embedding and retrieval. 
The splitter uses tiktoken encoding to ensure accurate token counting.
Finally, we create a Chroma vector store that will persist our embeddings locally, using Google's text-embedding-004 model for high-quality semantic representations.
"""
