"""Prism CV-threshold calibration (Stage 2 follow-up, finding F4).

The Prism baseline is our reimplementation from the paper description
(CV-triggered 1-D k-means k=3 on checkpoint scores, drop lowest cluster);
NOT official code.

Phase 1 (offline sweep): one full-depth pass over train-query pools records
checkpoint-1 scores and final scores per query. Because transformer rows are
independent, the prism behaviour (k-means on checkpoint scores, dropped
candidates keep their checkpoint score) can be simulated exactly offline for
every CV threshold: trigger rate, nDCG@10, layer-equivalents.
Phase 2 (measured): the selected thresholds are re-run with true conditional
execution on the 100-query test subset.

Results: artifacts/runs/prism_cv_calibration.jsonl
"""
import argparse
import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    "bcr", Path(__file__).resolve().parent / "build_candidate_runs.py")
bcr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcr)

CHECKPOINT = 1  # after layer 2 (0-indexed), as in bench_stage2_baselines


def rank_scores(pairs_scores):
    order = sorted(pairs_scores, key=lambda x: -x[1])
    return {did: float(len(order) - i) for i, (did, _) in enumerate(order)}


def build_train_pools(args, d):
    out = Path(f"data/candidate_runs/{args.dataset}/train_rrf_top100.jsonl")
    if out.exists():
        runs = {}
        with out.open() as f:
            for line in f:
                row = json.loads(line)
                runs[row["qid"]] = row["candidates"]
        return runs
    queries_all = data.load_queries(f"data/{args.dataset}/queries.jsonl")
    qrels = data.load_qrels(f"data/{args.dataset}/qrels/train.tsv")
    rng = np.random.RandomState(0)
    qids = sorted(set(qrels) & set(queries_all))
    if len(qids) > args.num_train:
        qids = list(rng.choice(qids, args.num_train, replace=False))
    queries = {q: queries_all[q] for q in qids}
    print(f"building pools for {len(queries)} train queries", flush=True)
    bm = bcr.bm25_run(d["corpus"], queries, qrels, args.budget)
    dn = bcr.dense_run(d["corpus"], queries, qrels, args.budget, args.bge)
    fused = bcr.rrf_fuse(bm, dn, args.budget)
    with out.open("w") as f:
        for qid in sorted(fused):
            f.write(json.dumps({"qid": qid, "candidates": fused[qid]}) + "\n")
    return fused


def kmeans1d(s, k=3, n_init=3, seed=0):
    """1-D k-means matching sklearn KMeans(n_init=3, random_state=0) closely
    enough for threshold calibration."""
    best_inertia, best_labels, best_centers = None, None, None
    x = s.reshape(-1, 1).astype(np.float64)
    rng = np.random.RandomState(seed)
    for _ in range(n_init):
        centers = x[rng.choice(len(x), k, replace=False)].flatten()
        for _ in range(50):
            labels = np.argmin(np.abs(x - centers.reshape(1, -1)), axis=1)
            new = np.array([x[labels == j].mean() if (labels == j).any() else centers[j]
                            for j in range(k)])
            if np.allclose(new, centers):
                centers = new
                break
            centers = new
        inertia = sum(((x[labels == j] - centers[j]) ** 2).sum() for j in range(k))
        if best_inertia is None or inertia < best_inertia:
            # keep labels AND centers from the same init (bug fix: previously
            # the returned centers came from the last init, not the best one)
            best_inertia, best_labels, best_centers = inertia, labels, centers.copy()
    return best_labels, best_centers


def simulate_prism(ckpt_scores, final_scores, doc_ids, cv):
    """Exact offline simulation of the prism reimplementation for one query."""
    s = ckpt_scores.numpy()
    cv_q = float(np.std(s) / (abs(np.mean(s)) + 1e-9))
    leq = 6.0
    scores = final_scores.clone()
    if cv_q >= cv and len(s) >= 6:
        labels, centers = kmeans1d(s)
        drop = labels == int(np.argmin(centers))
        if (~drop).any():
            for i in range(len(doc_ids)):
                if drop[i]:
                    scores[i] = ckpt_scores[i]
            leq = 6.0 * (~drop).sum() / len(doc_ids) + 2.0 * drop.sum() / len(doc_ids)
    return cv_q, leq, scores


def make_prism_keep_fn(cv_threshold):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--model", default="models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--bge", default="models/bge-small-en-v1.5")
    parser.add_argument("--num-train", type=int, default=150)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cv-sweep", type=float, nargs="+",
                        default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
    parser.add_argument("--out", default="artifacts/runs/prism_cv_calibration.jsonl")
    parser.add_argument("--cache-only", action="store_true",
                        help="stop after phase 1 (used to build the score cache)")
    args = parser.parse_args()

    torch.set_num_threads(2)
    d = data.load_split(f"data/{args.dataset}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ranker = ActiveSetRanker(args.model)
    log = JsonlLogger(args.out, run_meta={
        "stage": "stage2b", "event_type": "prism_cv_calibration",
        "dataset": args.dataset, "model": args.model,
    })

    print("=== phase 1: offline sweep on train pools ===", flush=True)
    train_runs = build_train_pools(args, d)
    train_qrels = data.load_qrels(f"data/{args.dataset}/qrels/train.tsv")
    qids = sorted(train_runs.keys())
    recs = {}
    sim_rows = []
    ckpt_cache, final_cache = [], []
    for qid in qids:
        doc_ids = sorted(train_runs[qid].keys())
        pairs = [(d["queries"][qid], d["corpus"][did]["title"] + " " + d["corpus"][did]["text"]) for did in doc_ids]
        enc = tokenizer(pairs, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        with torch.inference_mode():
            hidden, ext_mask = ranker.embed(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
            for i in range(CHECKPOINT + 1):
                hidden = ranker.forward_layer(hidden, ext_mask, i)
            ckpt_scores = ranker.score_hidden(hidden)
            for i in range(CHECKPOINT + 1, ranker.num_layers):
                hidden = ranker.forward_layer(hidden, ext_mask, i)
            final_scores = ranker.score_hidden(hidden)
        ckpt_cache.append(ckpt_scores.numpy().astype(np.float32))
        final_cache.append(final_scores.numpy().astype(np.float32))
        full = rank_scores(list(zip(doc_ids, final_scores.tolist())))
        sim_rows.append((qid, full))
        for cv in args.cv_sweep:
            cv_q, leq, scores = simulate_prism(ckpt_scores, final_scores, doc_ids, cv)
            r = recs.setdefault(cv, {"run": {}, "layers": [], "cvs": []})
            r["run"][qid] = rank_scores(list(zip(doc_ids, scores.tolist())))
            r["layers"].append(leq)
            r["cvs"].append(cv_q)
    run_full = {qid: full for qid, full in sim_rows}
    np.savez_compressed(
        f"data/candidate_runs/{args.dataset}/prism_calib_scores.npz",
        qid=np.array(qids), ckpt=np.array(ckpt_cache, dtype=object),
        final=np.array(final_cache, dtype=object))
    for cv in args.cv_sweep:
        r = recs[cv]
        m = metrics.evaluate({q: train_qrels[q] for q in r["run"]}, r["run"])
        m_full = metrics.evaluate({q: train_qrels[q] for q in run_full}, run_full)
        cvs = r["cvs"]
        rec = {
            "event": "prism_cv_train_sim", "cv_threshold": cv,
            "ndcg_10": round(m["ndcg_cut_10"], 4),
            "ndcg_10_full": round(m_full["ndcg_cut_10"], 4),
            "trigger_rate": round(float(np.mean([c >= cv for c in cvs])), 4),
            "cv_p50": round(float(np.percentile(cvs, 50)), 4),
            "cv_p90": round(float(np.percentile(cvs, 90)), 4),
            "mean_layers_eq": round(float(np.mean(r["layers"])), 2),
            "num_queries": len(r["run"]),
        }
        log.write(rec)
        print(json.dumps(rec), flush=True)

    if args.cache_only:
        print("cache-only: stopping after phase 1", flush=True)
        return

    print("=== phase 2: measured runs on test subset ===", flush=True)
    d_test = data.restrict_to_test_queries(d)
    runs = {}
    with open(f"data/candidate_runs/{args.dataset}/rrf_top100.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["qid"] in d_test["queries"]:
                runs[row["qid"]] = row["candidates"]
    runs = dict(list(runs.items())[: args.num_test])

    sim_recs = {}
    for line in Path(args.out).read_text().splitlines():
        rec = json.loads(line)
        if rec.get("event") == "prism_cv_train_sim":
            sim_recs[rec["cv_threshold"]] = rec
    trig = [cv for cv in args.cv_sweep
            if sim_recs.get(cv, {}).get("trigger_rate", 0) >= 0.05]
    selected = trig[:3] if trig else [args.cv_sweep[0]]
    print("selected thresholds:", selected, flush=True)

    for cv in selected:
        run_out, times, layers = {}, [], []
        kf = make_prism_keep_fn(cv)
        for qid, cands in runs.items():
            doc_ids = sorted(cands.keys())
            pairs = [(d["queries"][qid], d["corpus"][did]["title"] + " " + d["corpus"][did]["text"]) for did in doc_ids]
            enc = tokenizer(pairs, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            t0 = time.perf_counter()
            with torch.inference_mode():
                scores, exit_layer = ranker.run_active_set(
                    enc["input_ids"], enc["attention_mask"], enc["token_type_ids"],
                    checkpoints=[CHECKPOINT], keep_fn=kf)
            times.append(time.perf_counter() - t0)
            layers.append(exit_layer.float().mean().item())
            run_out[qid] = rank_scores(list(zip(doc_ids, scores.tolist())))
        m = metrics.evaluate({q: d_test["qrels"][q] for q in run_out}, run_out)
        rec = {
            "event": "prism_cv_test_measured", "cv_threshold": cv,
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
