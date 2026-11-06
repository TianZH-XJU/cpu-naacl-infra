"""First-stage candidate generation: BM25, dense (bge-small), and RRF fusion.

Generates fixed top-K candidate runs for the official test queries so that
all rerankers are compared on identical candidate lists (plan section 5.5).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast import data, metrics  # noqa: E402

RRF_K = 60


def bm25_run(corpus, queries, qrels, budget):
    from rank_bm25 import BM25Okapi

    doc_ids = list(corpus.keys())
    texts = [(corpus[d]["title"] + " " + corpus[d]["text"]).lower().split() for d in doc_ids]
    print(f"  BM25 indexing {len(doc_ids)} docs...")
    bm25 = BM25Okapi(texts)
    run = {}
    t0 = time.perf_counter()
    for qid, qtext in queries.items():
        scores = bm25.get_scores(qtext.lower().split())
        top = np.argsort(-scores)[:budget]
        run[qid] = {doc_ids[i]: float(scores[i]) for i in top if scores[i] > 0}
    print(f"  BM25 done in {time.perf_counter() - t0:.1f}s")
    return run


def dense_run(corpus, queries, qrels, budget, model_path):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), device="cpu")
    doc_ids = list(corpus.keys())
    texts = [(corpus[d]["title"] + " " + corpus[d]["text"]).strip() for d in doc_ids]
    print(f"  encoding {len(doc_ids)} docs ({model_path})...")
    t0 = time.perf_counter()
    doc_emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    print(f"  encoded in {time.perf_counter() - t0:.1f}s")
    run = {}
    t0 = time.perf_counter()
    q_texts = list(queries.values())
    q_emb = model.encode(q_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    sims = q_emb @ doc_emb.T
    for i, qid in enumerate(queries.keys()):
        top = np.argsort(-sims[i])[:budget]
        run[qid] = {doc_ids[j]: float(sims[i][j]) for j in top}
    print(f"  queries done in {time.perf_counter() - t0:.1f}s")
    return run


def rrf_fuse(run_a, run_b, budget):
    fused = {}
    for qid in run_a:
        scores = {}
        for rank, did in enumerate(sorted(run_a[qid], key=lambda d: -run_a[qid][d])):
            scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, did in enumerate(sorted(run_b.get(qid, {}), key=lambda d: -run_b[qid][d])):
            scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(scores.items(), key=lambda x: -x[1])[:budget]
        fused[qid] = dict(top)
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--embedder", default="models/bge-small-en-v1.5")
    args = parser.parse_args()

    for name in args.datasets:
        print(f"[{name}]")
        d = data.load_split(f"data/{name}")
        d = data.restrict_to_test_queries(d)
        corpus, queries, qrels = d["corpus"], d["queries"], d["qrels"]
        print(f"  test queries: {len(queries)} (from qrels), corpus: {len(corpus)}")

        out_dir = Path("data/candidate_runs") / name
        out_dir.mkdir(parents=True, exist_ok=True)

        run_bm25 = bm25_run(corpus, queries, qrels, args.budget)
        run_dense = dense_run(corpus, queries, qrels, args.budget, args.embedder)
        run_rrf = rrf_fuse(run_bm25, run_dense, args.budget)

        manifest = {}
        for method, run in (("bm25", run_bm25), ("dense", run_rrf and run_dense), ("rrf", run_rrf)):
            path = out_dir / f"{method}_top{args.budget}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for qid, cands in run.items():
                    f.write(json.dumps({"qid": qid, "candidates": cands}) + "\n")
            r100 = metrics.recall_at_k(qrels, run, k=args.budget)
            manifest[method] = {"path": str(path), "queries": len(run), f"recall_at_{args.budget}": round(r100, 4)}
            print(f"  {method}: Recall@{args.budget}={r100:.4f} -> {path}")

        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  wrote {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
