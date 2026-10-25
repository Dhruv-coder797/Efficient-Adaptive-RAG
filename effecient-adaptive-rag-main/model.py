from embedding_provider import embed_model
from llm_provider import get_llm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

llm_model = get_llm()
