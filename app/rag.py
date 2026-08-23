"""
rag.py — the "RAG" (Retrieval-Augmented Generation) engine.

WHAT THIS FILE DOES (in plain English):
1. Reads a text file full of Q&A knowledge.
2. Splits it into small chunks (so we don't overload the LLM with irrelevant text).
3. Converts each chunk into a vector using TF-IDF (a classic, lightweight way
   to represent text as numbers based on word importance).
4. At query time: convert the question into a vector the same way, then find
   the chunks whose vectors are most similar (cosine similarity).

NOTE ON THIS VERSION: the original build used neural embeddings
(sentence-transformers) + FAISS, which is the more "modern" RAG approach and
worth knowing for interviews. We switched to TF-IDF here specifically for
free-tier cloud hosting — neural embedding models need ~1GB+ RAM (via
PyTorch), which exceeds Render's free 512MB limit. TF-IDF needs only a few
MB and starts instantly, so the app actually stays up on free hosting.
This vectorizer -> matrix -> similarity-search pattern is the same core
idea as FAISS + embeddings, just with lighter-weight vectors — good
architecture trade-off to be able to explain in an interview.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 3  # how many Q&A blocks per chunk


def load_and_chunk(filepath: str) -> list[str]:
    """Read the knowledge file and split it into chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]

    chunks = []
    for i in range(0, len(blocks), CHUNK_SIZE):
        chunk = "\n\n".join(blocks[i:i + CHUNK_SIZE])
        chunks.append(chunk)
    return chunks


class VectorStore:
    """Wraps a TF-IDF vectorizer + the chunk vectors for similarity search."""

    def __init__(self):
        self.vectorizer = None
        self.chunk_vectors = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)

    def search(self, query, top_k=3):
        if self.vectorizer is None:
            raise RuntimeError("Vector store is empty. Call build() first.")

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]

        top_indices = similarities.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]


# --- Module-level singleton, built once at app startup ---
store = VectorStore()


def init_store(data_path):
    chunks = load_and_chunk(data_path)
    store.build(chunks)
    print(f"[rag] Indexed {len(chunks)} chunks from {data_path}")


def retrieve_context(question, top_k=3):
    """Get the most relevant chunks for a question, joined into one context string."""
    chunks = store.search(question, top_k=top_k)
    return "\n\n---\n\n".join(chunks)
