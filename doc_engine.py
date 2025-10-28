import os
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load the Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

# ✅ Use HuggingFace for embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize the Gemini LLM
llama_llm = Gemini(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)

# ✅ Load documents from the 'data' directory
documents = SimpleDirectoryReader("data").load_data()

# ✅ Create a VectorStoreIndex from the documents
index = VectorStoreIndex.from_documents(documents)

# ✅ Create a query engine with Gemini LLM + HuggingFace embeddings
query_engine = index.as_query_engine(llm=llama_llm)

def query_documents(user_query: str) -> str:
    """
    Queries the documents using the query engine,
    but only returns a short 2–3 line summary with the most relevant info.
    """
    instruction = """
    You are a precise assistant.
    - Pick only the most relevant insights from the documents.
    - Reply in maximum 2–3 short sentences.
    - Be clear and to the point.
    - Avoid long explanations or unnecessary details.
    """

    full_query = f"{instruction}\nUser: {user_query}\nAI:"
    response = query_engine.query(full_query)

    # Convert to string
    answer = str(response).strip()

    # Enforce 2–3 lines max
    sentences = answer.replace("\n", " ").split(". ")
    short_answer = ". ".join(sentences[:3])  # only 2–3 sentences max
    return short_answer.strip()
