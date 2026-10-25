from openai import OpenAI
from langchain_core.embeddings import Embeddings


class JinaEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "jina-embeddings-v3"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.jina.ai/v1",
        )
        self.model = model

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding