import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter # Import the text splitter
from dotenv import load_dotenv
from langchain_core.documents import Document as LangchainDocument # Alias to avoid conflict if needed
from langchain_community.vectorstores import Chroma # Import ChromaDB
# Load environment variables from .env file
load_dotenv()

# --- 1. Configuration and Setup ---
#
# IMPORTANT: Replace "YOUR_API_KEY" with your actual Google API Key or set it as an environment variable.
# You can get an API key from Google AI Studio: https://aistudio.google.com/app/apikey
# It's recommended to set it as an environment variable: export GOOGLE_API_KEY="your_api_key_here"
# If running in an environment where __api_key__ is provided (e.g., Canvas), it will be used.
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    # Fallback for Canvas environment if __api_key__ is not set as GOOGLE_API_KEY
    # In a real application, you'd handle this more robustly (e.g., prompt user, raise error).
    print("Warning: GOOGLE_API_KEY environment variable not set. Please set it or provide it directly.")
    # Attempt to use a global variable if available, common in some interactive environments
    api_key = globals().get('__api_key__', '')
    if not api_key:
        raise ValueError("Google API Key is not set. Please set GOOGLE_API_KEY environment variable.")
# from google.colab import userdata
# api_key = userdata.get('GOOGLE_API_KEY_1')

# Initialize the Generative AI model
# We'll use gemini-2.0-flash for faster responses.
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Initialize the embedding model for ChromaDB
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def vector_db_setup(documents: List[LangchainDocument], persist_directory: str = "chroma_db"):
    """
    Initializes a Chroma vector store from the provided documents.

    Args:
        documents (List[LangchainDocument]): A list of LangchainDocument objects to be stored.
        persist_directory (str): The directory where the ChromaDB instance will be saved.
    """
    print("Initializing ChromaDB with sample documents...")
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    print("ChromaDB initialized.")
    return vectorstore
