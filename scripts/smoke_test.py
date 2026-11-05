"""Smoke test for the CPU-CAST Stage-1 environment.

1. Load each downloaded cross-encoder locally and score a few pairs.
2. Load BGE-small and embed a short corpus.
3. Load a BEIR dataset (scifact) in the classic layout.
Writes results to infra/results/smoke_test.json.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
RESULTS = REPO / "infra" / "results"

QUERY = "How does photosynthesis work?"
DOCS = [
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The stock market fluctuated wildly during the trading session.",
    "Chlorophyll absorbs light in the blue and red spectrum.",
]


def bench_cross_encoder(name: str) -> dict:
    model = CrossEncoder(str(MODELS / name), device="cpu")
    pairs = [(QUERY, d) for d in DOCS]
    scores = model.predict(pairs)
    t0 = time.perf_counter()
    for _ in range(3):
        model.predict(pairs)
    dt = (time.perf_counter() - t0) / 3
    return {
        "model": name,
        "scores": [round(float(s), 4) for s in scores],
        "batch_of_3_mean_ms": round(dt * 1000 / 3, 2),
    }


def bench_embedder() -> dict:
    model = SentenceTransformer(str(MODELS / "bge-small-en-v1.5"), device="cpu")
    emb = model.encode(DOCS, normalize_embeddings=True)
    import numpy as np
    sim = float((emb[0] @ emb[2]))
    t0 = time.perf_counter()
    model.encode(DOCS * 8, batch_size=16, normalize_embeddings=True)
    dt = time.perf_counter() - t0
    return {
        "model": "bge-small-en-v1.5",
        "embedding_dim": int(emb.shape[1]),
        "cosine(query_doc0, doc2)": round(sim, 4),
        "ms_per_doc_batch16": round(dt * 1000 / (len(DOCS) * 8), 2),
    }


def check_beir(name: str) -> dict:
    d = REPO / "data" / name
    corpus = pd.read_json(d / "corpus.jsonl", lines=True)
    queries = pd.read_json(d / "queries.jsonl", lines=True)
    qrels_test = (d / "qrels" / "test.tsv").exists()
    return {
        "dataset": name,
        "corpus_rows": len(corpus),
        "query_rows": len(queries),
        "qrels_test_present": qrels_test,
        "corpus_columns": list(corpus.columns),
    }


def main() -> None:
    torch.set_num_threads(4)
    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "cross_encoders": [], "beir": []}
    for name in (
        "ms-marco-TinyBERT-L-2-v2",
        "ms-marco-MiniLM-L-2-v2",
        "ms-marco-MiniLM-L-4-v2",
        "ms-marco-MiniLM-L-6-v2",
    ):
        print(f"[cross-encoder {name}]")
        out["cross_encoders"].append(bench_cross_encoder(name))
    print("[embedder bge-small]")
    out["embedder"] = bench_embedder()
    for name in ("scifact", "nfcorpus", "arguana", "fiqa"):
        print(f"[beir {name}]")
        out["beir"].append(check_beir(name))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "smoke_test.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
