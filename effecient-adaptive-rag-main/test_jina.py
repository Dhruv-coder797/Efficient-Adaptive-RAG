from dotenv import load_dotenv
from jina_embeddings import JinaEmbeddings
import os

load_dotenv()

emb = JinaEmbeddings(
    api_key=os.environ["JINA_API_KEY"]
)

vec = emb.embed_query("Hello world")

print(len(vec))
print(vec[:5])