"""
Run this script once to build the FAISS vector store from medical documents.
Usage: python build_vectordb.py
"""

import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

DOCUMENTS_DIR = "documents"
VECTOR_STORE_DIR = "vector_store"
CHUNK_SIZE = 300  # characters per chunk

def load_and_chunk_documents(docs_dir: str, chunk_size: int) -> list[dict]:
    """Load all .txt files and split into chunks."""
    chunks = []
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            # Split into overlapping chunks
            for i in range(0, len(text), chunk_size - 50):
                chunk = text[i:i + chunk_size].strip()
                if chunk:
                    chunks.append({"text": chunk, "source": filename})
    return chunks

def build_vectordb():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading and chunking documents...")
    chunks = load_and_chunk_documents(DOCUMENTS_DIR, CHUNK_SIZE)
    texts = [c["text"] for c in chunks]

    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "index.faiss"))

    # Save chunk metadata alongside the index
    with open(os.path.join(VECTOR_STORE_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"Vector store saved to '{VECTOR_STORE_DIR}/' with {len(texts)} chunks.")

if __name__ == "__main__":
    build_vectordb()
