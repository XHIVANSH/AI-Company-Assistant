"""Load PDFs from docs/, split them, embed them and store them in Pinecone.

Run once (and again whenever the PDFs change):  python -m src.ingest
"""
import glob
import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src import config


def ensure_index(pc: Pinecone, dimension: int) -> None:
    existing = [i["name"] for i in pc.list_indexes()]
    if config.PINECONE_INDEX in existing:
        return
    print(f"Creating Pinecone index '{config.PINECONE_INDEX}' (dim={dimension})...")
    pc.create_index(
        name=config.PINECONE_INDEX,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
    )
    while not pc.describe_index(config.PINECONE_INDEX).status["ready"]:
        time.sleep(2)


def main() -> None:
    config.require_keys("GOOGLE_API_KEY", "PINECONE_API_KEY")

    pdf_paths = sorted(glob.glob(os.path.join(config.DOCS_DIR, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {config.DOCS_DIR}. Add some and run again.")
        return

    pages = []
    for path in pdf_paths:
        print(f"Loading {os.path.basename(path)}")
        pages.extend(PyPDFLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL, google_api_key=config.GOOGLE_API_KEY
    )
    dimension = len(embeddings.embed_query("dimension check"))

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    ensure_index(pc, dimension)

    PineconeVectorStore.from_documents(
        chunks, embedding=embeddings, index_name=config.PINECONE_INDEX
    )
    print("Done. Documents are stored in Pinecone.")


if __name__ == "__main__":
    main()
