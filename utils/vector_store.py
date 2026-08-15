from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Environment
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. "
        "Please add it to the .env file in the project root."
    )


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "3gpp"

VECTOR_DB_DIR = DATA_DIR / "faiss_index"


# --------------------------------------------------
# PDF files
# --------------------------------------------------

PDF_FILES = [
    DATA_DIR / "TS_23_501.pdf",
    DATA_DIR / "TS_23_502.pdf",
    DATA_DIR / "TS_23_503.pdf",
]


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

# IMPORTANT:
# This must be the same embedding model used by backend/rag.py
EMBEDDING_MODEL = "text-embedding-3-small"


# --------------------------------------------------
# Create vector store
# --------------------------------------------------

def create_vector_store():

    print("=" * 70)
    print("3GPP RAG - VECTOR DATABASE CREATION")
    print("=" * 70)

    all_documents = []

    # --------------------------------------------------
    # Load PDFs
    # --------------------------------------------------

    for pdf_path in PDF_FILES:

        if not pdf_path.exists():

            print(f"ERROR: File not found: {pdf_path}")
            continue

        print()
        print(f"Loading: {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))

        documents = loader.load()

        print(f"Pages loaded: {len(documents)}")

        # Add document name to metadata
        for document in documents:

            document.metadata["document"] = pdf_path.name

        all_documents.extend(documents)


    # --------------------------------------------------
    # Check documents
    # --------------------------------------------------

    if not all_documents:

        raise RuntimeError(
            "No PDF documents were loaded."
        )


    print()
    print(f"Total pages loaded: {len(all_documents)}")


    # --------------------------------------------------
    # Split documents
    # --------------------------------------------------

    print()
    print("Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(
        all_documents
    )

    print(f"Total chunks created: {len(chunks)}")


    # --------------------------------------------------
    # Create OpenAI embeddings
    # --------------------------------------------------

    print()
    print("Configuring OpenAI embedding model...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )


    # --------------------------------------------------
    # Create FAISS database
    # --------------------------------------------------

    print()
    print("Creating FAISS vector database...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )


    # --------------------------------------------------
    # Save FAISS database
    # --------------------------------------------------

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(VECTOR_DB_DIR)
    )


    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("VECTOR DATABASE CREATED SUCCESSFULLY")
    print("=" * 70)

    print(f"Documents : {len(PDF_FILES)}")
    print(f"Pages     : {len(all_documents)}")
    print(f"Chunks    : {len(chunks)}")
    print(f"Embedding : {EMBEDDING_MODEL}")
    print(f"Location  : {VECTOR_DB_DIR}")

    print("=" * 70)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    create_vector_store()
