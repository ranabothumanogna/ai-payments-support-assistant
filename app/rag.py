"""
rag.py — the "RAG" (Retrieval-Augmented Generation) engine.

WHAT THIS FILE DOES (in plain English):
1. Reads a text file full of Q&A knowledge.
2. Splits it into small chunks (so we don't overload the LLM with irrelevant text).
3. Converts each chunk into a vector (a list of numbers) using an embedding model.
   -> Similar meaning = similar vector. This is what makes "search by meaning" possible.
4. Stores those vectors in FAISS (a vector database) for fast similarity search.
5. At query time: embed the user's question -> find the closest chunks -> return them.
   Those chunks get stuffed into the LLM prompt so it can "ground" its answer in facts.

This is the exact pattern the JD calls "RAG", "Embeddings", and "Vector Databases".
"""

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Config ---
CHUNK_SIZE = 3          # how many Q&A blocks per chunk
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # small, free, runs locally (no API key needed)

# --- Load the embedding model once (reused across requests) ---
# This model turns text into a 384-dimensional vector.
_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


def load_and_chunk(filepath: str) -> list[str]:
    """Read the knowledge file and split it into chunks.

    We split on blank lines (each Q&A block), then group a few blocks
    together so each chunk has enough context but isn't too large.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]

    chunks = []
    for i in range(0, len(blocks), CHUNK_SIZE):
        chunk = "\n\n".join(blocks[i:i + CHUNK_SIZE])
        chunks.append(chunk)
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    """Convert a list of strings into a matrix of embedding vectors."""
    vectors = _embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")


class VectorStore:
    """A thin wrapper around a FAISS index + the original text chunks.

    FAISS only stores numbers (vectors) and returns index positions —
    we keep the original text ourselves so we can map back to it.
    """

    def __init__(self):
        self.index: faiss.Index | None = None
        self.chunks: list[str] = []

    def build(self, chunks: list[str]):
        self.chunks = chunks
        vectors = embed_texts(chunks)
        dim = vectors.shape[1]

        # IndexFlatIP = exact search using inner product (cosine similarity,
        # since our vectors are normalized). Simple and fine for small datasets.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        if self.index is None:
            raise RuntimeError("Vector store is empty. Call build() first.")

        query_vector = embed_texts([query])
        scores, indices = self.index.search(query_vector, top_k)

        results = [self.chunks[i] for i in indices[0] if i != -1]
        return results


# --- Module-level singleton, built once at app startup ---
store = VectorStore()


def init_store(data_path: str):
    chunks = load_and_chunk(data_path)
    store.build(chunks)
    print(f"[rag] Indexed {len(chunks)} chunks from {data_path}")


def retrieve_context(question: str, top_k: int = 3) -> str:
    """Get the most relevant chunks for a question, joined into one context string."""
    chunks = store.search(question, top_k=top_k)
    return "\n\n---\n\n".join(chunks)
