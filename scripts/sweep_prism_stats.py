"""Sweep Prism-style dispersion statistics on cached checkpoint scores.

The raw-logit CV is nearly constant across queries (median 0.013), so a fixed
CV threshold can never fire meaningfully (see prism_cv_calibration.jsonl).
This script tests scale-robust dispersion statistics on the SAME cached
checkpoint-1 scores, simulates the prism reimplementation offline for each
(statistic, threshold), and runs the best configs on the test subset with
true conditional execution.

Statistics:
  raw_cv     std/|mean| of raw logits            (scale-dependent, inert)
  minmax_cv  std/mean of min-max scores to [0,1] (scale-free)
  gap_ratio  (p75-p25)/(max-min+eps)             (scale-free, robust)

Results: appended to artifacts/runs/prism_cv_calibration.jsonl
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast import data, metrics  # noqa: E402
from cpu_cast.active_set import ActiveSetRanker  # noqa: E402
from cpu_cast.logging_utils import JsonlLogger  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "cpc", Path(__file__).resolve().parent / "calibrate_prism_cv.py")
cpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpc)


def dispersion(s, stat):
    if stat == "raw_cv":
        return float(np.std(s) / (abs(np.mean(s)) + 1e-9))
    if stat == "minmax_cv":
        lo, hi = s.min(), s.max()
        x = (s - lo) / (hi - lo + 1e-9)
        return float(np.std(x) / (x.mean() + 1e-9))
    if stat == "gap_ratio":
        lo, hi = s.min(), s.max()
        return float((np.percentile(s, 75) - np.percentile(s, 25)) / (hi - lo + 1e-9))
    raise ValueError(stat)


def simulate(ckpt, final, stat, threshold, drop_frac_max=0.5):
    """Offline prism simulation; returns (dispersion, dropped_mask or None)."""
    s = ckpt.numpy()
    d = dispersion(s, stat)
    if d < threshold or len(s) < 6:
        return d, None
    labels, centers = cpc.kmeans1d(s)
    drop = labels == int(np.argmin(centers))
    if drop.all() or drop.mean() > drop_frac_max:
        return d, None
    return d, drop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--model", default="models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cache", default="data/candidate_runs/scifact/prism_calib_scores.npz")
    parser.add_argument("--out", default="artifacts/runs/prism_cv_calibration.jsonl")
    args = parser.parse_args()

    torch.set_num_threads(2)
    z = np.load(args.cache, allow_pickle=True)
    qids = list(z["qid"])

    def _unpack(x):
        x = np.asarray(x)
        if x.dtype == object:
            x = np.array([float(v) for v in np.ravel(x)], dtype=np.float32)
        return x.astype(np.float32)

    ckpts = [_unpack(x) for x in z["ckpt"]]
    finals = [_unpack(x) for x in z["final"]]
    train_qrels = data.load_qrels(f"data/{args.dataset}/qrels/train.tsv")
    log = JsonlLogger(args.out, run_meta={
        "stage": "stage2b", "event_type": "prism_stat_sweep",
        "dataset": args.dataset, "cache": args.cache,
    })

    # per-query full-depth reference on train pools
    pools = {}
    with open(f"data/candidate_runs/{args.dataset}/train_rrf_top100.jsonl") as f:
        for line in f:
            row = json.loads(line)
            pools[row["qid"]] = sorted(row["candidates"].keys())
    run_full = {}
    for qid, fin in zip(qids, finals):
        doc_ids = pools[qid]
        run_full[qid] = cpc.rank_scores(list(zip(doc_ids, fin.tolist())))
    m_full = metrics.evaluate({q: train_qrels[q] for q in run_full}, run_full)
    print(f"full-depth train nDCG@10 = {m_full['ndcg_cut_10']:.4f}", flush=True)

    stats = {}
    for stat in ("raw_cv", "minmax_cv", "gap_ratio"):
        vals = [dispersion(np.asarray(c, dtype=np.float64), stat) for c in ckpts]
        stats[stat] = vals
        print(f"{stat}: p10={np.percentile(vals,10):.4f} p50={np.percentile(vals,50):.4f} "
              f"p90={np.percentile(vals,90):.4f} cv_of_stat={np.std(vals)/ (abs(np.mean(vals))+1e-9):.2f}", flush=True)

    sweep = {"raw_cv": [0.01, 0.012, 0.014, 0.016],
             "minmax_cv": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
             "gap_ratio": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]}
    best = []
    for stat, thresholds in sweep.items():
        for thr in thresholds:
            run, layers, trig = {}, [], 0
            for qid, ck, fin in zip(qids, ckpts, finals):
                doc_ids = pools[qid]
                d, drop = simulate(torch.from_numpy(ck), torch.from_numpy(fin), stat, thr)
                ft = torch.from_numpy(fin)
                if drop is None:
                    scores = ft.clone()
                    leq = 6.0
                else:
                    scores = ft.clone()
                    scores[torch.from_numpy(drop)] = torch.from_numpy(ck)[torch.from_numpy(drop)]
                    leq = 6.0 * (~drop).sum() / len(doc_ids) + 2.0 * drop.sum() / len(doc_ids)
                    trig += 1
                run[qid] = cpc.rank_scores(list(zip(doc_ids, scores.tolist())))
                layers.append(leq)
            m = metrics.evaluate({q: train_qrels[q] for q in run}, run)
            tr = trig / len(qids)
            rec = {
                "event": "prism_stat_train_sim", "statistic": stat, "threshold": thr,
                "ndcg_10": round(m["ndcg_cut_10"], 4),
                "ndcg_10_full": round(m_full["ndcg_cut_10"], 4),
                "trigger_rate": round(tr, 4),
                "mean_layers_eq": round(float(np.mean(layers)), 2),
                "num_queries": len(run),
            }
            log.write(rec)
            print(json.dumps(rec), flush=True)
            if 0.15 <= tr <= 0.9:
                best.append((rec["ndcg_10"], stat, thr))
    best.sort(reverse=True)
    print("best configs (trigger 15-90%):", best[:3], flush=True)

    if not best:
        print("no config triggers in 15-90% band", flush=True)
        return

    print("=== measured runs on test subset ===", flush=True)
    d = data.load_split(f"data/{args.dataset}")
    d_test = data.restrict_to_test_queries(d)
    from cpu_cast.active_set import ActiveSetRanker as R
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ranker = R(args.model)
    runs = {}
    with open(f"data/candidate_runs/{args.dataset}/rrf_top100.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["qid"] in d_test["queries"]:
                runs[row["qid"]] = row["candidates"]
    runs = dict(list(runs.items())[: args.num_test])

    def make_keep_fn(stat, thr):
        def keep_fn(scores, layer_idx):
            s = scores.cpu().numpy()
            if dispersion(s, stat) < thr or len(s) < 6:
                return torch.ones_like(scores, dtype=torch.bool)
            labels, centers = cpc.kmeans1d(s)
            drop = labels == int(np.argmin(centers))
            if drop.all() or drop.mean() > 0.5:
                return torch.ones_like(scores, dtype=torch.bool)
            keep = torch.from_numpy(~drop).to(scores.device)
            return keep if keep.any() else torch.ones_like(scores, dtype=torch.bool)
        return keep_fn

    for _, stat, thr in best[:2]:
        run_out, times, layers = {}, [], []
        kf = make_keep_fn(stat, thr)
        for qid, cands in runs.items():
            doc_ids = sorted(cands.keys())
            pairs = [(d["queries"][qid], d["corpus"][did]["title"] + " " + d["corpus"][did]["text"]) for did in doc_ids]
            enc = tokenizer(pairs, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            t0 = time.perf_counter()
            with torch.inference_mode():
                scores, exit_layer = ranker.run_active_set(
                    enc["input_ids"], enc["attention_mask"], enc["token_type_ids"],
                    checkpoints=[1], keep_fn=kf)
            times.append(time.perf_counter() - t0)
            layers.append(exit_layer.float().mean().item())
            run_out[qid] = cpc.rank_scores(list(zip(doc_ids, scores.tolist())))
        m = metrics.evaluate({q: d_test["qrels"][q] for q in run_out}, run_out)
        rec = {
            "event": "prism_stat_test_measured", "statistic": stat, "threshold": thr,
            "ndcg_10": round(m["ndcg_cut_10"], 4),
            "mrr": round(m["recip_rank"], 4),
            "mean_layers_eq": round(float(np.mean(layers)), 2),
            "p50_ms": round(float(np.percentile(times, 50)) * 1000, 1),
            "p95_ms": round(float(np.percentile(times, 95)) * 1000, 1),
            "num_queries": len(run_out),
        }
        log.write(rec)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
