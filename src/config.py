"""Settings and API-key checks."""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "company-policy")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")


def require_keys(*names: str) -> None:
    """Exit with a clear message if any required key is missing."""
    values = {
        "GROQ_API_KEY": GROQ_API_KEY,
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "TAVILY_API_KEY": TAVILY_API_KEY,
    }
    missing = [n for n in names if not values.get(n)]
    if missing:
        print(f"Missing keys in .env: {', '.join(missing)}")
        sys.exit(1)
