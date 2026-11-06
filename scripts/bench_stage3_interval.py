"""Stage-3: compaction-interval grid on real exit patterns (plan section 9.2).

Extends the Gate A microbenchmark to full-run level: with a fixed exit
schedule (checkpoints after layers 2 and 4, trained exit-head scoring,
GAINER current_rank policy, per-checkpoint survival r), compare:

  m=1   compact every layer (default)
  m=2   compact every 2 layers
  m=3   compact every 3 layers
  m=inf never compact (always run the full stale batch)

Measured: wall-clock p50/p95, realized layer-equivalents, nDCG@10,
top-10 agreement. Also records the realized active fraction per layer.

Results: artifacts/runs/bench_stage3_interval.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast import data, metrics  # noqa: E402
from cpu_cast.active_set import ActiveSetRanker  # noqa: E402
from cpu_cast.logging_utils import JsonlLogger  # noqa: E402
from scripts.bench_stage2b_gainer import load_heads, rank_scores  # noqa: E402
from scripts.train_gainer_heads import MLP  # noqa: E402

CHECKPOINTS = [1, 3]


def topk_keep_fn(frac):
    def keep_fn(scores, layer_idx):
        k = max(1, int(round(frac * scores.numel())))
        thresh = torch.topk(scores, k).values[-1]
        return scores >= thresh
    return keep_fn


class HeadScorer:
    """keep_fn wrapper: score active rows with the trained exit head."""

    def __init__(self, heads, frac):
        self.heads = heads
        self.frac = frac
        self.stage = 0

    def __call__(self, hidden, idx_active, layer_idx):
        cls = hidden[idx_active, 0, :]
        with torch.inference_mode():
            eh = self.heads[self.stage](cls.float())
        k = max(1, int(round(self.frac * eh.numel())))
        thresh = torch.topk(eh, k).values[-1]
        self.stage += 1
        return eh >= thresh, eh


def run_with_heads(ranker, enc, heads, frac, compact_interval, compact_threshold):
    """run_active_set variant that scores exits with trained heads."""
    n = enc["input_ids"].shape[0]
    L = ranker.num_layers
    hidden, ext_mask = ranker.embed(
        enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
    active = torch.ones(n, dtype=torch.bool)
    exit_layer = torch.full((n,), L, dtype=torch.int64)
    scores = torch.zeros(n)
    scorer = HeadScorer(heads, frac)
    active_frac = []
    with torch.inference_mode():
        for i in range(L):
            idx = torch.nonzero(active, as_tuple=True)[0]
            active_frac.append(round(idx.numel() / n, 3))
            if (
                idx.numel() < n
                and idx.numel() / n < compact_threshold
                and i % compact_interval == 0
            ):
                h = hidden.index_select(0, idx)
                m = ext_mask.index_select(0, idx)
                h = ranker.forward_layer(h, m, i)
                hidden = hidden.index_copy(0, idx, h)
            else:
                hidden = ranker.forward_layer(hidden, ext_mask, i)
            if i in CHECKPOINTS:
                keep_local, eh = scorer(hidden, idx, i)
                idx_active = torch.nonzero(active, as_tuple=True)[0]
                keep = torch.zeros(n, dtype=torch.bool)
                keep[idx_active] = keep_local
                finished = active & ~keep
                exit_layer[finished] = i + 1
                scores[finished] = eh[~keep_local]
                active &= keep
        scores[active] = ranker.score_hidden(hidden)[active]
    return scores, exit_layer, active_frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--run", default="data/candidate_runs/scifact/rrf_top100.jsonl")
    parser.add_argument("--model", default="models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--heads", default="models/gainer_heads")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--survivals", type=float, nargs="+", default=[0.5, 0.25])
    parser.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 3, 999])
    parser.add_argument("--compact-threshold", type=float, default=0.75)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out", default="artifacts/runs/bench_stage3_interval.jsonl")
    args = parser.parse_args()

    torch.set_num_threads(2)
    d = data.load_split(f"data/{args.dataset}")
    d = data.restrict_to_test_queries(d)
    runs = {}
    with open(args.run, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["qid"] in d["queries"]:
                runs[row["qid"]] = row["candidates"]
    runs = dict(list(runs.items())[: args.num_queries])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ranker = ActiveSetRanker(args.model)
    heads, _ = load_heads(args.heads)
    log = JsonlLogger(args.out, run_meta={
        "stage": "stage3", "dataset": args.dataset, "run": args.run,
        "model": args.model, "compact_threshold": args.compact_threshold,
    })

    teacher_top10 = {}
    configs = [("full_depth", 0.0, 1)]
    for r in args.survivals:
        for m in args.intervals:
            configs.append((f"r{int(r*100)}_m{m}", r, m))

    for name, frac, m in configs:
        run_out, times, layers = {}, [], []
        active_frac_last = None
        for qid, cands in runs.items():
            doc_ids = sorted(cands.keys())
            pairs = [(d["queries"][qid], d["corpus"][did]["title"] + " " + d["corpus"][did]["text"]) for did in doc_ids]
            enc = tokenizer(pairs, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            t0 = time.perf_counter()
            if name == "full_depth":
                with torch.inference_mode():
                    scores, exit_layer = ranker.run_active_set(
                        enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
                af = None
            else:
                scores, exit_layer, af = run_with_heads(
                    ranker, enc, heads, frac, m, args.compact_threshold)
            times.append(time.perf_counter() - t0)
            layers.append(exit_layer.float().mean().item())
            active_frac_last = af
            run_out[qid] = rank_scores(list(zip(doc_ids, scores.tolist())))
            if name == "full_depth":
                teacher_top10[qid] = set(list(run_out[qid].keys())[:10])
        mtr = metrics.evaluate({q: d["qrels"][q] for q in run_out}, run_out)
        agree = [
            len(set(list(r.keys())[:10]) & teacher_top10[qid]) / 10.0
            for qid, r in run_out.items()
        ]
        record = {
            "event": "stage3_interval",
            "method": name,
            "ndcg_10": round(mtr["ndcg_cut_10"], 4),
            "mrr": round(mtr["recip_rank"], 4),
            "num_queries": len(run_out),
            "p50_ms": round(float(np.percentile(times, 50)) * 1000, 1),
            "p95_ms": round(float(np.percentile(times, 95)) * 1000, 1),
            "p99_ms": round(float(np.percentile(times, 99)) * 1000, 1),
            "mean_layers_eq": round(float(np.mean(layers)), 2),
            "top10_agreement_full_depth": round(float(np.mean(agree)), 4),
            "active_frac_profile": active_frac_last,
        }
        log.write(record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
