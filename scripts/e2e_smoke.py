"""Stage-0 end-to-end smoke test for one query (plan section 8.1).

Query -> fixed candidate run -> cross-encoder rerank -> scores/ranking,
latency breakdown, and candidate-level nDCG@10, all written as JSONL.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast import data, metrics  # noqa: E402
from cpu_cast.logging_utils import JsonlLogger  # noqa: E402

from sentence_transformers import CrossEncoder  # noqa: E402


def load_run(path, qid):
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["qid"] == qid:
                return row["candidates"]
    raise KeyError(qid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--run", default="data/candidate_runs/scifact/rrf_top100.jsonl")
    parser.add_argument("--model", default="models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--qid", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out", default="artifacts/runs/e2e_smoke.jsonl")
    args = parser.parse_args()

    torch.set_num_threads(2)
    d = data.load_split(f"data/{args.dataset}")
    d = data.restrict_to_test_queries(d)
    qid = args.qid or sorted(d["queries"].keys())[0]
    query = d["queries"][qid]
    cands = load_run(args.run, qid)
    doc_ids = sorted(cands.keys())
    print(f"query {qid}: {query[:80]}... candidates: {len(doc_ids)}")

    model = CrossEncoder(args.model, device="cpu", max_length=args.max_length)
    pairs = [(query, d["corpus"][did]["title"] + " " + d["corpus"][did]["text"]) for did in doc_ids]

    t_tokenize = time.perf_counter()
    _ = model.tokenizer(pairs, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    t_tokenize = time.perf_counter() - t_tokenize

    t0 = time.perf_counter()
    scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    t_rerank = time.perf_counter() - t0

    ranking = sorted(zip(doc_ids, scores), key=lambda x: -x[1])
    run = {qid: {did: float(s) for did, s in ranking}}
    qrels = {qid: d["qrels"][qid]}
    m = metrics.evaluate(qrels, run)

    gold_rank = next((i for i, (did, _) in enumerate(ranking) if d["qrels"][qid].get(did, 0) > 0), None)
    record = {
        "event": "e2e_smoke",
        "dataset": args.dataset,
        "qid": qid,
        "query": query,
        "num_candidates": len(doc_ids),
        "model": args.model,
        "tokenize_ms": round(t_tokenize * 1000, 2),
        "rerank_ms": round(t_rerank * 1000, 2),
        "per_pair_us": round(t_rerank * 1e6 / len(doc_ids), 1),
        "first_relevant_rank": gold_rank,
        "candidate_ndcg_at_10": round(m["ndcg_cut_10"], 4),
        "candidate_mrr": round(m["recip_rank"], 4),
        "top5": [(did, round(float(s), 3)) for did, s in ranking[:5]],
    }
    logger = JsonlLogger(args.out, run_meta={"stage": "stage0", "model": args.model})
    logger.write(record)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
