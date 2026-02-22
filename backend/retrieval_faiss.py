from typing import List, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer
from backend.rag_pipeline import DocumentChunk


class FaissRetriever:
    """
    Semantic retrieval using SentenceTransformers embeddings + FAISS.
    Returns (DocumentChunk, score) where score is cosine similarity.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.emb = None
        self.chunks: List[DocumentChunk] = []

    def build(self, chunks: List[DocumentChunk]):
        if faiss is None:
            raise RuntimeError("faiss nije instaliran. Instaliraj faiss-cpu.")

        self.chunks = chunks

        texts = [c.text for c in chunks]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = emb.astype("float32")

        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb = emb / norms
        self.emb = emb

        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if self.index is None:
            return []

        q = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1
        q = q / q_norm

        scores, idxs = self.index.search(q, top_k)

        results: List[Tuple[DocumentChunk, float]] = []
        for idx, score in zip(idxs[0], scores[0]):
            if idx < 0:
                continue
            results.append((self.chunks[int(idx)], float(score)))
        return results
