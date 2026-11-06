"""Stage-2 strong-baseline matrix (plan section 7.3 / Stage 2).

All methods run on the SAME fixed candidate lists, same tokenizer, same
weights (MiniLM-L6 unless stated), 2 threads, and are compared on:

  - candidate-level nDCG@10 / MRR (qrels)
  - wall-clock per query (p50/p95/p99)
  - top-10 agreement with the full-depth teacher
  - compute: mean layer-equivalents per candidate

Methods (all labelled with their fidelity to the original papers):
  full_depth        MiniLM-L6, all 6 layers (teacher / baseline)
  tinybert_only     static shallow TinyBERT-L2 (no dynamic compute)
  cascade_shallow   TinyBERT-L2 scores all K -> top-p go to full L6
  see_token         SEE-style pre-Transformer token max-similarity filter
                    (embedding layer only), top-p continue
  score_cascade     checkpoints after layers 2 and 4, keep top-p by checkpoint
                    score (score-based cascade; GAINER's baseline; the
                    intermediate head is NOT separately trained -- proxy)
  budget_alloc      GAINER-style simplified: single checkpoint after layer 2,
                    fixed budget = top-p candidates advance (untrained proxy)
  prism_cluster     Prism-style simplified: after layer 2, CV-triggered 1-D
                    k-means (k=3) on scores; lowest cluster dropped
                    (reimplementation from the paper's description, NOT
                    official code)

Results: artifacts/runs/bench_stage2.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast import data, metrics  # noqa: E402
from cpu_cast.active_set import ActiveSetRanker  # noqa: E402
from cpu_cast.logging_utils import JsonlLogger  # noqa: E402

CHECKPOINTS = [1, 3]  # after layer 2 and layer 4 (0-indexed)

_SHALLOW = {}


def load_shallow(path):
    if path not in _SHALLOW:
        _SHALLOW[path] = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return _SHALLOW[path]


def rank_scores(pairs_scores):
    order = sorted(pairs_scores, key=lambda x: -x[1])
    return {did: float(len(order) - i) for i, (did, _) in enumerate(order)}


def topk_keep_fn(frac):
    def keep_fn(scores, layer_idx):
        k = max(1, int(round(frac * scores.numel())))
        thresh = torch.topk(scores, k).values[-1]
        return scores >= thresh
    return keep_fn


def prism_keep_fn(cv_threshold):
    from sklearn.cluster import KMeans

    def keep_fn(scores, layer_idx):
        s = scores.cpu().numpy()
        cv = float(np.std(s) / (abs(np.mean(s)) + 1e-9))
        if cv < cv_threshold or len(s) < 6:
            return torch.ones_like(scores, dtype=torch.bool)
        km = KMeans(n_clusters=3, n_init=3, random_state=0).fit(s.reshape(-1, 1))
        centers = km.cluster_centers_.flatten()
        drop = km.labels_ == int(np.argmin(centers))
        keep = torch.from_numpy(~drop).to(scores.device)
        if keep.sum() == 0:
            return torch.ones_like(scores, dtype=torch.bool)
        return keep
    return keep_fn


def token_maxsim(ranker, enc):
    """SEE-style token-level max-similarity from the embedding layer only."""
    with torch.inference_mode():
        emb = ranker.bert.embeddings(
            input_ids=enc["input_ids"], token_type_ids=enc["token_type_ids"]
        )
    tt = enc["token_type_ids"].bool()
    am = enc["attention_mask"].bool()
    scores = []
    for b in range(emb.shape[0]):
        q = emb[b][tt[b] & am[b]]
        d = emb[b][(~tt[b]) & am[b]]
        sim = q @ d.T
        scores.append(float(sim.max(dim=1).values.mean()))
    return scores


def run_method(base, keep_frac, ranker, enc, doc_ids, args):
    t0 = time.perf_counter()
    layers_eq = None
    if base == "full_depth":
        with torch.inference_mode():
            scores, exit_layer = ranker.run_active_set(
                enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
        layers_eq = exit_layer.float().mean().item()
    elif base == "tinybert_only":
        shallow = load_shallow(args.tinybert)
        with torch.inference_mode():
            scores = shallow(**enc).logits.squeeze(-1)
        layers_eq = 0.0
    elif base == "score_cascade":
        with torch.inference_mode():
            scores, exit_layer = ranker.run_active_set(
                enc["input_ids"], enc["attention_mask"], enc["token_type_ids"],
                checkpoints=CHECKPOINTS, keep_fn=topk_keep_fn(keep_frac))
        layers_eq = exit_layer.float().mean().item()
    elif base == "budget_alloc":
        with torch.inference_mode():
            scores, exit_layer = ranker.run_active_set(
                enc["input_ids"], enc["attention_mask"], enc["token_type_ids"],
                checkpoints=[1], keep_fn=topk_keep_fn(keep_frac))
        layers_eq = exit_layer.float().mean().item()
    elif base == "prism_cluster":
        with torch.inference_mode():
            scores, exit_layer = ranker.run_active_set(
                enc["input_ids"], enc["attention_mask"], enc["token_type_ids"],
                checkpoints=[1], keep_fn=prism_keep_fn(args.cv_threshold))
        layers_eq = exit_layer.float().mean().item()
    elif base == "see_token":
        sim = token_maxsim(ranker, enc)
        k = max(1, int(round(keep_frac * len(doc_ids))))
        keep_idx = set(np.argsort(-np.asarray(sim))[:k].tolist())
        sub_idx = [i for i in range(len(doc_ids)) if i in keep_idx]
        sub_enc = {k2: v[sub_idx] for k2, v in enc.items()}
        with torch.inference_mode():
            s2, exit_layer = ranker.run_active_set(
                sub_enc["input_ids"], sub_enc["attention_mask"], sub_enc["token_type_ids"])
        scores = torch.full((len(doc_ids),), -1e9)
        for j, i in enumerate(sub_idx):
            scores[i] = s2[j]
        layers_eq = (exit_layer.float().sum().item() + (len(doc_ids) - len(sub_idx)) * 0.5) / len(doc_ids)
    elif base == "cascade_shallow":
        shallow = load_shallow(args.tinybert)
        with torch.inference_mode():
            s1 = shallow(**enc).logits.squeeze(-1)
        k = max(1, int(round(keep_frac * len(doc_ids))))
        keep_idx = set(torch.topk(s1, k).indices.tolist())
        sub_idx = sorted(keep_idx)
        sub_enc = {k2: v[sub_idx] for k2, v in enc.items()}
        with torch.inference_mode():
            s2, _ = ranker.run_active_set(
                sub_enc["input_ids"], sub_enc["attention_mask"], sub_enc["token_type_ids"])
        scores = torch.full((len(doc_ids),), -1e9)
        for j, i in enumerate(sub_idx):
            scores[i] = 1e3 + s2[j]
        for i in range(len(doc_ids)):
            if i not in keep_idx:
                scores[i] = float(s1[i])
        layers_eq = (6 * len(sub_idx) + 0.5 * (len(doc_ids) - len(sub_idx))) / len(doc_ids)
    else:
        raise ValueError(base)
    dt = time.perf_counter() - t0
    ranked = sorted(zip(doc_ids, scores.tolist()), key=lambda x: -x[1])
    return rank_scores(ranked), dt, layers_eq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--run", default="data/candidate_runs/scifact/rrf_top100.jsonl")
    parser.add_argument("--model", default="models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--tinybert", default="models/ms-marco-TinyBERT-L-2-v2")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--keep-fracs", type=float, nargs="+", default=[0.5, 0.25])
    parser.add_argument("--cv-threshold", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out", default="artifacts/runs/bench_stage2.jsonl")
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
    log = JsonlLogger(args.out, run_meta={
        "stage": "stage2", "dataset": args.dataset, "run": args.run,
        "model": args.model, "keep_fracs": args.keep_fracs,
    })

    methods = [("full_depth", 0.0), ("tinybert_only", 0.0)]
    for f in args.keep_fracs:
        for base in ("cascade_shallow", "see_token", "score_cascade", "budget_alloc"):
            methods.append((base, f))
    methods.append(("prism_cluster", args.cv_threshold))

    teacher_top10 = {}
    for base, param in methods:
        name = base if base in ("full_depth", "tinybert_only") else (
            f"{base}_cv{param}" if base == "prism_cluster" else f"{base}_p{int(param*100)}")
        keep_frac = param if base != "prism_cluster" else 0.0
        run_out, times, layers = {}, [], []
        for qid, cands in runs.items():
            doc_ids = sorted(cands.keys())
            pairs = [(d["queries"][qid], d["corpus"][did]["title"] + " " + d["corpus"][did]["text"]) for did in doc_ids]
            enc = tokenizer(pairs, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            r, dt, leq = run_method(base, keep_frac, ranker, enc, doc_ids, args)
            run_out[qid] = r
            times.append(dt)
            layers.append(leq)
            if base == "full_depth":
                teacher_top10[qid] = set(list(r.keys())[:10])
        m = metrics.evaluate({q: d["qrels"][q] for q in run_out}, run_out)
        agree = [
            len(set(list(r.keys())[:10]) & teacher_top10[qid]) / 10.0
            for qid, r in run_out.items()
        ]
        record = {
            "event": "stage2_baseline",
            "method": name,
            "ndcg_10": round(m["ndcg_cut_10"], 4),
            "mrr": round(m["recip_rank"], 4),
            "num_queries": len(run_out),
            "p50_ms": round(float(np.percentile(times, 50)) * 1000, 1),
            "p95_ms": round(float(np.percentile(times, 95)) * 1000, 1),
            "p99_ms": round(float(np.percentile(times, 99)) * 1000, 1),
            "mean_layers_eq": round(float(np.mean(layers)), 2),
            "top10_agreement_full_depth": round(float(np.mean(agree)), 4),
        }
        log.write(record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
