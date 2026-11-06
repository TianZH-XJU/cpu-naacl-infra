"""Stage-2b: GAINER-faithful evaluation on scifact (Gate B requirement).

GAINER reproduction per the official notebook (GAINER_v3.ipynb), adapted to
MiniLM-L6 / our fixed RRF top-100 runs:

- exit heads: trained per GAINER recipe on nfcorpus train (leave-one-dataset-out)
- policies at each checkpoint (after layers 2 and 4):
    current_rank  rank by exit-head score at the current checkpoint
                  (GAINER's main baseline)
    gainer_gain   rank by the trained counterfactual-gain ranker
                  (GAINER's method; reduced scalar feature set)
    oracle        rank by true counterfactual nDCG utility (upper bound,
                  simulation only, uses future information)
- quality: simulated rollouts on recorded checkpoint scores (GAINER's own
  evaluation protocol); latency: true conditional execution on CPU.

Results: artifacts/runs/bench_stage2b_gainer.jsonl
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
from scripts.train_gainer_heads import (  # noqa: E402
    MLP, counterfactual_utility, features_at)

CHECKPOINTS = [1, 3]  # after layers 2 and 4 (0-indexed)


def load_heads(path):
    heads, rankers = [], []
    for i in range(2):
        h = MLP(384)
        h.load_state_dict(torch.load(Path(path) / f"exit_head_{i}.pt", weights_only=True))
        heads.append(h.eval())
    for i in range(2):
        p = Path(path) / f"gain_ranker_{i}.pt"
        if p.exists():
            ck = torch.load(p, weights_only=True)
            m = MLP(388)
            m.load_state_dict(ck["state"])
            rankers.append((m.eval(), ck["mu"], ck["sd"]))
        else:
            rankers.append(None)
    return heads, rankers


def rank_scores(pairs_scores):
    order = sorted(pairs_scores, key=lambda x: -x[1])
    return {did: float(len(order) - i) for i, (did, _) in enumerate(order)}


def features_flat(cur, prev):
    n = cur.numel()
    mean = cur.mean()
    std = cur.std(unbiased=False).clamp(min=1e-6)
    z = (cur - mean) / std
    rank = cur.argsort(descending=True).argsort().float() / max(1, n)
    return torch.stack([cur, rank, z, prev], dim=-1)


def run_gainer_measured(ranker, enc, heads, rankers, policy, r, compact_threshold=0.75):
    """True conditional execution with exit-head scoring and quota keep."""
    n = enc["input_ids"].shape[0]
    L = ranker.num_layers
    hidden, ext_mask = ranker.embed(
        enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
    active = torch.ones(n, dtype=torch.bool)
    scores = torch.zeros(n)
    exit_layer = torch.full((n,), L, dtype=torch.int64)
    prev_score = torch.zeros(n)
    stage = 0
    with torch.inference_mode():
        for i in range(L):
            idx = torch.nonzero(active, as_tuple=True)[0]
            if idx.numel() < n and idx.numel() / n < compact_threshold:
                h = hidden.index_select(0, idx)
                m = ext_mask.index_select(0, idx)
                h = ranker.forward_layer(h, m, i)
                hidden = hidden.index_copy(0, idx, h)
            else:
                hidden = ranker.forward_layer(hidden, ext_mask, i)
            if i in CHECKPOINTS:
                cls = hidden[:, 0, :]
                idx_active = torch.nonzero(active, as_tuple=True)[0]
                eh = heads[stage](cls[idx_active].float())
                if policy == "current_rank":
                    prio = eh
                else:
                    prev = eh if stage == 0 else prev_score[idx_active]
                    feats = features_flat(eh, prev)
                    x = torch.cat([cls[idx_active].float(), feats], dim=-1)
                    m_dl, mu, sd = rankers[stage]
                    prio = m_dl((x - mu) / sd)
                k = max(1, round(r * idx_active.numel()))
                keep_local = torch.zeros_like(prio, dtype=torch.bool)
                keep_local[prio.topk(k).indices] = True
                keep = torch.zeros(n, dtype=torch.bool)
                keep[idx_active] = keep_local
                finished = active & ~keep
                scores[finished] = eh[~keep_local]
                exit_layer[finished] = i + 1
                prev_score[idx_active[keep_local]] = eh[keep_local]
                active &= keep
                stage += 1
        scores[active] = ranker.score_hidden(hidden)[active]
    return scores, exit_layer


def sim_rollout(S, valid, r, prio_fn):
    Q, N, C = S.shape
    cur = S[..., 0].clone()
    active = valid.clone()
    depth = torch.zeros(Q, N, dtype=torch.long)
    for s in range(C - 1):
        if not bool(active.any()):
            break
        prev = S[..., s - 1] if s > 0 else cur
        if r >= 1.0:
            sel = active
        else:
            prio = prio_fn(s, active, cur, prev).masked_fill(~active, -1e9)
            counts = active.sum(1, keepdim=True)
            k = (r * counts).round().clamp(min=1)
            rank = prio.argsort(dim=1, descending=True).argsort(dim=1)
            sel = active & (rank < k)
        depth = torch.where(sel, torch.full_like(depth, s + 1), depth)
        cur = torch.where(sel, S[..., s + 1], cur)
        active = sel
    return cur, depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--run", default="data/candidate_runs/scifact/rrf_top100.jsonl")
    parser.add_argument("--model", default="models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--heads", default="models/gainer_heads")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--survivals", type=float, nargs="+", default=[0.5, 0.25])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out", default="artifacts/runs/bench_stage2b_gainer.jsonl")
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
    heads, rankers = load_heads(args.heads)
    log = JsonlLogger(args.out, run_meta={
        "stage": "stage2b", "dataset": args.dataset, "run": args.run,
        "model": args.model, "survivals": args.survivals,
    })

    disc = 1.0 / torch.log2(torch.arange(2, 12, dtype=torch.float32))
    sim_policies = ["current_rank", "gainer_gain", "oracle"]
    measured_policies = ["current_rank", "gainer_gain"]

    results = {f"{p}_r{int(r*100)}": {"scores": {}, "times": [], "layers": []}
               for p in sim_policies for r in args.survivals}
    results["full_depth"] = {"scores": {}, "times": [], "layers": []}
    teacher_top10 = {}

    for qid, cands in runs.items():
        doc_ids = sorted(cands.keys())
        N = len(doc_ids)
        pairs = [(d["queries"][qid], d["corpus"][did]["title"] + " " + d["corpus"][did]["text"]) for did in doc_ids]
        enc = tokenizer(pairs, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")

        t0 = time.perf_counter()
        with torch.inference_mode():
            hidden, ext_mask = ranker.embed(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
            cls_rec = {}
            for i in range(ranker.num_layers):
                hidden = ranker.forward_layer(hidden, ext_mask, i)
                if i in CHECKPOINTS:
                    cls_rec[i] = hidden[:, 0, :].clone()
            final = ranker.score_hidden(hidden)
        dt_full = time.perf_counter() - t0
        teacher = rank_scores(list(zip(doc_ids, final.tolist())))
        teacher_top10[qid] = set(list(teacher.keys())[:10])
        results["full_depth"]["scores"][qid] = teacher
        results["full_depth"]["times"].append(dt_full)
        results["full_depth"]["layers"].append(6.0)

        with torch.no_grad():
            eh = [heads[s](cls_rec[CHECKPOINTS[s]].float()) for s in range(len(CHECKPOINTS))]
        S = torch.stack(eh + [final], dim=-1)
        gains = torch.zeros(N)
        for i, did in enumerate(doc_ids):
            gains[i] = float(d["qrels"].get(qid, {}).get(did, 0))
        sg, _ = gains.sort(descending=True)
        idcg = (sg[:10] * disc).sum().clamp(min=1e-9)
        valid = torch.ones(N, dtype=torch.bool)

        for r in args.survivals:
            for policy in sim_policies:
                def prio_fn(s, active, cur, prev, _policy=policy):
                    if _policy == "current_rank":
                        return cur
                    if _policy == "oracle":
                        return counterfactual_utility(
                            cur, S.unsqueeze(0)[..., -1], active,
                            valid.unsqueeze(0), gains.unsqueeze(0), idcg.unsqueeze(0))[0]
                    cls_s = cls_rec[CHECKPOINTS[s]]
                    feats = features_at(cur, prev, active)[0]
                    x = torch.cat([cls_s, feats], dim=-1)
                    m_dl, mu, sd = rankers[s]
                    return m_dl((x - mu) / sd)

                cur, depth = sim_rollout(S.unsqueeze(0), valid.unsqueeze(0), r, prio_fn)
                scores = cur[0]
                key = f"{policy}_r{int(r*100)}"
                results[key]["scores"][qid] = rank_scores(list(zip(doc_ids, scores.tolist())))
                leq = (2.0 + torch.where(depth[0] >= 1, 2.0, 0.0)
                       + torch.where(depth[0] >= 2, 2.0, 0.0)).mean().item()
                results[key]["layers"].append(leq)

            for policy in measured_policies:
                t0 = time.perf_counter()
                scores_m, exit_layer = run_gainer_measured(
                    ranker, enc, heads, rankers, policy, r)
                dt = time.perf_counter() - t0
                key = f"{policy}_r{int(r*100)}"
                results[key]["times"].append(dt)
                results[key]["layers"][-1] = exit_layer.float().mean().item()

    for key, res in sorted(results.items()):
        m = metrics.evaluate({q: d["qrels"][q] for q in res["scores"]}, res["scores"])
        agree = [
            len(set(list(r.keys())[:10]) & teacher_top10[qid]) / 10.0
            for qid, r in res["scores"].items() if qid in teacher_top10
        ]
        record = {
            "event": "stage2b_gainer",
            "method": key,
            "ndcg_10": round(m["ndcg_cut_10"], 4),
            "mrr": round(m["recip_rank"], 4),
            "num_queries": len(res["scores"]),
            "mean_layers_eq": round(float(np.mean(res["layers"])), 2),
            "top10_agreement_full_depth": round(float(np.mean(agree)), 4),
        }
        if res["times"]:
            record.update({
                "p50_ms": round(float(np.percentile(res["times"], 50)) * 1000, 1),
                "p95_ms": round(float(np.percentile(res["times"], 95)) * 1000, 1),
                "p99_ms": round(float(np.percentile(res["times"], 99)) * 1000, 1),
            })
        log.write(record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
