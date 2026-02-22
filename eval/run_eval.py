import sys
from pathlib import Path

# Dodaj root projekta u Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import pandas as pd

from backend.rag_pipeline import RagPipeline


# ---- Helper: dummy upload from disk (ako evaliraš nad lokalnim PDF-om) ----
class DummyUpload:
    def __init__(self, path: str):
        self.name = os.path.basename(path)
        self._path = path

    def read(self):
        with open(self._path, "rb") as f:
            return f.read()


def load_dataset(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize(s: str) -> str:
    return (s or "").lower()


def keyword_hit(chunks_text: str, keywords: List[str]) -> bool:
    txt = normalize(chunks_text)
    for kw in keywords:
        if normalize(kw) in txt:
            return True
    return False


def concat_topk_snippets(citations: List[Dict[str, Any]]) -> str:
    return "\n".join([c.get("snippet", "") for c in citations])


def run_mode(pipeline: RagPipeline, mode: str, dataset: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    pipeline.retrieval_mode = mode

    rows = []
    for ex in dataset:
        q = ex["question"]
        keywords = ex.get("keywords", [])
        doc_hint = ex.get("doc_hint", None)

        t0 = time.perf_counter()
        ans, cites, timings = pipeline.answer(q, top_k=top_k, skip_llm=False)  # koristi answer() koji vraća citations sa score
        t1 = time.perf_counter()

        # retrieval latency: nemamo posebno merenje ako nije izloženo, pa merimo total
        total_ms = (t1 - t0) * 1000.0

        top1_score = None
        if cites and "score" in cites[0]:
            top1_score = cites[0]["score"]

        # Hit@k: da li keywords postoje u top_k snippetovima
        hit = None
        if keywords:
            hit = keyword_hit(concat_topk_snippets(cites), keywords)

        # Doc hint check: da li bar 1 citation source sadrži doc_hint
        doc_ok = None
        if doc_hint:
            doc_ok = any(doc_hint.lower() in (c.get("source", "").lower()) for c in cites)

        rows.append({
            "mode": mode,
            "question": q,
            "top_k": top_k,
            "answer_len": len(ans or ""),
            "num_citations": len(cites) if cites else 0,
            "top1_score": top1_score,
            "hit_at_k": hit,
            "doc_hint_ok": doc_ok,
            "latency_total_ms": round(total_ms, 2),
            "retrieval_ms": timings["retrieval_ms"],
            "llm_ms": timings["llm_ms"],
            "total_ms": timings["total_ms"],
            "best_score": timings["best_score"],
            "mode_used": timings["mode"],

        })

    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # agregati po mode
    out = []
    for mode, g in df.groupby("mode"):
        hit_vals = g["hit_at_k"].dropna()
        hit_rate = float(hit_vals.mean()) if len(hit_vals) else None

        out.append({
            "mode": mode,
            "n": len(g),
            "hit_rate@k": round(hit_rate, 3) if hit_rate is not None else None,
            "avg_top1_score": round(g["top1_score"].dropna().mean(), 4) if g["top1_score"].notna().any() else None,
            "p50_retrieval_ms": round(g["retrieval_ms"].median(), 2),
            "p95_retrieval_ms": round(g["retrieval_ms"].quantile(0.95), 2),
            "p50_llm_ms": round(g["llm_ms"].median(), 2),
            "p95_llm_ms": round(g["llm_ms"].quantile(0.95), 2),
            "p50_total_ms": round(g["total_ms"].median(), 2),
            "p95_total_ms": round(g["total_ms"].quantile(0.95), 2),
            "avg_num_citations": round(g["num_citations"].mean(), 2),
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    # 1) Putanja do PDF-a koji evaliraš (stavi svoj)
    # Može i više PDF-ova
    pdfs = [
        r"data\sample_pdfs\ML_System_Design.pdf",
    ]

    dataset_path = r"eval\dataset.jsonl"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    dataset = load_dataset(dataset_path)

    # 2) Build index jednom
    pipeline = RagPipeline()
    uploads = [DummyUpload(p) for p in pdfs]
    pipeline.build_index(uploads)

    # 3) Run oba moda
    rows = []
    for mode in ["tfidf", "faiss"]:
        rows.extend(run_mode(pipeline, mode, dataset, top_k=3))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "eval_runs.csv"), index=False, encoding="utf-8")

    summary = summarize(df)
    summary.to_csv(os.path.join(results_dir, "eval_summary.csv"), index=False, encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(" - results/eval_runs.csv")
    print(" - results/eval_summary.csv")
