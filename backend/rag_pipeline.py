from dataclasses import dataclass
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
import os
from dotenv import load_dotenv
from groq import Groq
import json 
from pathlib import Path
from backend.pdf_parser import extract_text_from_pdf
from backend.chunker import chunk_text


load_dotenv()


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_id: int


class RagPipeline:
    """
    Jednostavan RAG sistem baziran na:
    - bag-of-words
    - TF-IDF ručnim vektorisanjem
    - Cosine-similarity retrieval
    - Groq LLaMA model odgovorima
    """

    def __init__(self):
        self.reset()
        self.retrieval_mode = "tfidf"  # ili "faiss"
        self.faiss_retriever = None

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY nije definisan u okruženju.")

        self.llm = Groq(api_key=api_key)

    # ---------------------------------------
    # RESET PIPELINE
    # ---------------------------------------
    def reset(self):
        """Obriši sve što je vezano za prethodno indeksirane dokumente."""
        self.chunks: List[DocumentChunk] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.doc_matrix: Optional[np.ndarray] = None

    # ---------------------------------------
    # TOKENIZACIJA
    # ---------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\wšđčćž]+", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]

    # ---------------------------------------
    # INDEX BUILDING
    # ---------------------------------------
    def build_index(self, uploaded_files):
        all_chunks: List[DocumentChunk] = []

        global_chunk_id = 0

        for file in uploaded_files:
            name = file.name
            pages = extract_text_from_pdf(file.read())

            for page_num, text in pages:
                parts = chunk_text(text)
                for p in parts:
                    all_chunks.append(
                        DocumentChunk(
                            text=p,
                            source=name,
                            page=page_num,
                            chunk_id=global_chunk_id
                        )
                    )
                    global_chunk_id += 1


        if not all_chunks:
            raise ValueError("PDF dokument nema čitljiv tekst.")

        self.chunks = all_chunks

        # --- gradimo TF-IDF ---
        vocab = {}
        tokens_per_doc = []

        for ch in self.chunks:
            toks = self._tokenize(ch.text)
            tokens_per_doc.append(toks)
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)

        if not vocab:
            raise ValueError("Vokabular prazan – nema dovoljno teksta.")

        self.vocab = vocab

        N = len(self.chunks)
        V = len(vocab)

        tf = np.zeros((N, V), dtype=np.float32)
        df = np.zeros(V, dtype=np.int32)

        for i, toks in enumerate(tokens_per_doc):
            counts: Dict[int, int] = {}
            for t in toks:
                idx = vocab[t]
                counts[idx] = counts.get(idx, 0) + 1

            for idx, c in counts.items():
                tf[i, idx] = c
                df[idx] += 1

        # Normalizacija TF
        row_sum = tf.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        tf = tf / row_sum

        # IDF
        df[df == 0] = 1
        idf = np.log((1 + N) / df) + 1
        self.idf = idf

        # TF-IDF matrica
        tfidf = tf * idf

        # L2 norm
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        tfidf = tfidf / norms

        self.doc_matrix = tfidf
        # --- build FAISS (semantic) index ---
        try:
            from backend.retrieval_faiss import FaissRetriever
            self.faiss_retriever = FaissRetriever()
            self.faiss_retriever.build(self.chunks)
        except Exception as e:
            # Ne rušimo app ako FAISS nije dostupan
            self.faiss_retriever = None
            print("FAISS build skipped:", e)
        print("DEBUG chunks:", len(self.chunks))
        print("DEBUG faiss_retriever:", "OK" if self.faiss_retriever is not None else "NONE")

    @property
    def is_ready(self):
        return self.doc_matrix is not None and len(self.chunks) > 0
    ### SAVE INDEX ###
    def save_index(self, dir_path: str = "models/index"):
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        # chunks metadata
        chunks_payload = [
            {"text": c.text, "source": c.source, "page": c.page, "chunk_id": c.chunk_id}
            for c in self.chunks
        ]
        (p / "chunks.json").write_text(json.dumps(chunks_payload, ensure_ascii=False), encoding="utf-8")

        # vocab + idf + tfidf matrix
        np.savez_compressed(
            p / "tfidf.npz",
            doc_matrix=self.doc_matrix,
            idf=self.idf,
            vocab_keys=np.array(list(self.vocab.keys()), dtype=object),
            vocab_vals=np.array(list(self.vocab.values()), dtype=np.int32),
        )

        # faiss index (optional)
        if self.faiss_retriever is not None and getattr(self.faiss_retriever, "index", None) is not None:
            try:
                import faiss
                faiss.write_index(self.faiss_retriever.index, str(p / "faiss.index"))
            except Exception as e:
                print("FAISS save skipped:", e)

    def load_index(self, dir_path: str = "models/index") -> bool:
        p = Path(dir_path)
        chunks_file = p / "chunks.json"
        tfidf_file = p / "tfidf.npz"

        if not chunks_file.exists() or not tfidf_file.exists():
            return False

        # load chunks
        chunks_payload = json.loads(chunks_file.read_text(encoding="utf-8"))
        self.chunks = [
            DocumentChunk(text=x["text"], source=x["source"], page=int(x["page"]), chunk_id=int(x["chunk_id"]))
            for x in chunks_payload
        ]

        # load tfidf artifacts
        data = np.load(tfidf_file, allow_pickle=True)
        self.doc_matrix = data["doc_matrix"]
        self.idf = data["idf"]

        keys = data["vocab_keys"].tolist()
        vals = data["vocab_vals"].tolist()
        self.vocab = {k: int(v) for k, v in zip(keys, vals)}

        # load faiss index (optional)
        faiss_path = p / "faiss.index"
        if faiss_path.exists():
            try:
                import faiss
                from backend.retrieval_faiss import FaissRetriever
                self.faiss_retriever = FaissRetriever()
                self.faiss_retriever.chunks = self.chunks  # attach same order
                self.faiss_retriever.index = faiss.read_index(str(faiss_path))
            except Exception as e:
                self.faiss_retriever = None
                print("FAISS load skipped:", e)

        return True

    # ---------------------------------------
    # QUERY EMBEDDING
    # ---------------------------------------
    def _embed_query(self, q: str) -> np.ndarray:
        toks = self._tokenize(q)
        if not toks or not self.vocab:
            return np.zeros(len(self.vocab), dtype=np.float32)

        vec = np.zeros(len(self.vocab), dtype=np.float32)

        for t in toks:
            if t in self.vocab:
                vec[self.vocab[t]] += 1

        if vec.sum() > 0:
            vec /= vec.sum()

        if self.idf is not None:
            vec = vec * self.idf

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    # ---------------------------------------
    # RETRIEVAL
    # ---------------------------------------
    def _retrieve(self, q: str, top_k: int) -> List[Tuple[DocumentChunk, float]]:
        qvec = self._embed_query(q)

        if self.doc_matrix is None:
            return []

        sims = self.doc_matrix @ qvec  # cosine similarity
        idxs = sims.argsort()[::-1][:top_k]

        results = []
        for i in idxs:
            results.append((self.chunks[int(i)], float(sims[int(i)])))

        return results

    # ---------------------------------------
    # LLM ANSWER
    # ---------------------------------------
    def _llm_answer(self, question: str, context: str) -> str:
        prompt = f"""
Ti si AI sistem za pretragu dokumenata.

Kontekst:
{context}

Pitanje:
{question}

Uputstva:
- Koristi isključivo informacije iz konteksta.
- Ako informacija ne postoji, reci: "Informacija nije pronađena u dokumentu."
- Odgovaraj kratko i jasno.
"""

        resp = self.llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()

    # ---------------------------------------
    # MAIN ANSWER METHOD
    # ---------------------------------------
    def answer(self, question: str, top_k: int = 3, min_score: float = 0.05, skip_llm: bool = False):
        t0 = time.perf_counter()

        # --- choose retrieval ---
        t_retr0 = time.perf_counter()
        if getattr(self, "retrieval_mode", "tfidf") == "faiss" and getattr(self, "faiss_retriever", None) is not None:
            results = self.faiss_retriever.search(question, top_k=top_k)
            default_min_score = 0.15
            mode_used = "faiss"
        else:
            results = self._retrieve(question, top_k)
            default_min_score = 0.02
            mode_used = "tfidf"
        t_retr1 = time.perf_counter()

        # if user didn't pass min_score explicitly, keep your default behavior
        if min_score is None:
            min_score = default_min_score

        if not results:
            return "Informacija nije pronađena u dokumentu.", [], {
                "mode": mode_used,
                "retrieval_ms": round((t_retr1 - t_retr0) * 1000, 2),
                "llm_ms": 0.0,
                "total_ms": round((time.perf_counter() - t0) * 1000, 2),
                "best_score": None,
            }

        best_score = results[0][1]

        # generička pitanja (kratka) ne odbijamo
        generic = len(question.split()) <= 5
        if best_score < min_score and not generic:
            return "Informacija nije pronađena u dokumentu.", [], {
                "mode": mode_used,
                "retrieval_ms": round((t_retr1 - t_retr0) * 1000, 2),
                "llm_ms": 0.0,
                "total_ms": round((time.perf_counter() - t0) * 1000, 2),
                "best_score": round(best_score, 4),
            }

        context_parts = []
        citations = []
        for ch, score in results:
            context_parts.append(
                f"[{ch.source} – strana {ch.page} | chunk {ch.chunk_id} | score {score:.3f}]\n{ch.text}"
            )
            citations.append({
                "source": ch.source,
                "page": ch.page,
                "chunk_id": ch.chunk_id,
                "score": round(score, 4),
                "snippet": ch.text[:350] + ("..." if len(ch.text) > 350 else "")
            })

        # retrieval-only
        if skip_llm:
            return "", citations, {
                "mode": mode_used,
                "retrieval_ms": round((t_retr1 - t_retr0) * 1000, 2),
                "llm_ms": 0.0,
                "total_ms": round((time.perf_counter() - t0) * 1000, 2),
                "best_score": round(best_score, 4),
            }

        context = "\n\n".join(context_parts)

        t_llm0 = time.perf_counter()
        answer = self._llm_answer(question, context)
        t_llm1 = time.perf_counter()

        return answer, citations, {
            "mode": mode_used,
            "retrieval_ms": round((t_retr1 - t_retr0) * 1000, 2),
            "llm_ms": round((t_llm1 - t_llm0) * 1000, 2),
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
            "best_score": round(best_score, 4),
        }
