"""
ingest.py
---------
One-time (or re-run) script that:
1. Reads all PDFs from the data/ folder
2. Extracts text page by page
3. Splits text into chunks
4. Creates embeddings with sentence-transformers (all-MiniLM-L6-v2)
5. Stores vectors + metadata in a persistent ChromaDB collection

Run this script whenever you add or update policy PDFs:
    python ingest.py
"""

import os
import sys
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "hr_policies"


def load_pdfs(data_dir: Path) -> List:
    """Load every PDF in data_dir and return a list of LangChain Documents."""
    documents = []
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDF files found in {data_dir}")
        print("   Please place your HR policy PDFs inside the data/ folder.")
        return []

    print(f"📄 Found {len(pdf_files)} PDF(s):")
    for pdf_path in pdf_files:
        print(f"   • {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Enrich metadata
            for page in pages:
                page.metadata["file_name"] = pdf_path.name
                # page.metadata already contains 'page' (0-indexed from pypdf)
                # We convert to 1-indexed for human readability later
                page.metadata["page"] = page.metadata.get("page", 0) + 1

            documents.extend(pages)
            print(f"     → Loaded {len(pages)} page(s)")
        except Exception as e:
            print(f"     ⚠️  Failed to load {pdf_path.name}: {e}")

    return documents


def split_documents(documents: List) -> List:
    """Split documents into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    # Add a unique chunk_id for easier debugging
    for i, chunk in enumerate(chunks):
        file_stem = Path(chunk.metadata.get("file_name", "unknown")).stem.lower()
        chunk.metadata["chunk_id"] = f"{file_stem}_{i}"

    print(f"✂️  Created {len(chunks)} text chunks")
    return chunks


def create_vector_store(chunks: List):
    """Embed chunks and persist them in ChromaDB."""
    print("🧠 Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("💾 Creating / updating ChromaDB collection...")
    # Remove existing collection if present so we get a clean rebuild
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    print(f"✅ Vector store created successfully at {CHROMA_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Total vectors: {vectorstore._collection.count()}")
    return vectorstore


def main():
    print("=" * 60)
    print("HR Policy Assistant – Knowledge Base Ingestion")
    print("=" * 60)

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"📁 Created data/ folder. Please add PDF files and re-run.")
        sys.exit(1)

    documents = load_pdfs(DATA_DIR)
    if not documents:
        sys.exit(1)

    chunks = split_documents(documents)
    create_vector_store(chunks)

    print("\n🎉 Ingestion complete! You can now start the Streamlit app.")


if __name__ == "__main__":
    main()