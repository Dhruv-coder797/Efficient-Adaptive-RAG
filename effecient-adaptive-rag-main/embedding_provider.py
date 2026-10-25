import os

from dotenv import load_dotenv

from jina_embeddings import JinaEmbeddings

load_dotenv()

embed_model = JinaEmbeddings(
    api_key=os.environ["JINA_API_KEY"],
    model="jina-embeddings-v3",
)
