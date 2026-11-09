"""Train GAINER-style exit heads and gain rankers (GAINER_v3.ipynb recipe, adapted).

Exit heads: MLP(hidden->160->1), GELU, dropout 0.1, AdamW lr 2e-3 wd 1e-4,
smooth-L1 regression of the official final score from the CLS hidden state at
each checkpoint (exact GAINER hyperparameters).

Gain rankers: one pairwise ranker per transition, trained on counterfactual
nDCG@10 utility (u = nDCG(cur with candidate i advanced) - nDCG(cur)) collected
from current-rank rollouts at several survival rates, following GAINER's
collect->sample_pairs->ranknet pipeline with a reduced scalar feature set
[cur, rank_frac, z_score, prev_score] (GAINER uses a longer scalar list).

Output: models/gainer_heads/{exit_head_{i}.pt, gain_ranker_{i}.pt, meta.json}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast import data  # noqa: E402

HEAD_HIDDEN = 160
EXIT_EPOCHS, EXIT_LR, EXIT_PATIENCE, EXIT_BATCH = 25, 2e-3, 5, 512
RANK_EPOCHS, RANK_LR, RANK_BATCH = 25, 2e-3, 512
PAIRS_PER_GROUP = 32
UTILITY_EPS = 1e-9
SURVIVALS = [0.25, 0.40, 0.55, 0.70, 0.85]
EVAL_K = 10


class MLP(torch.nn.Module):
    def __init__(self, d_in, hidden=HEAD_HIDDEN):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Dropout(0.10), torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def pad_per_query(df, N):
    """Rebuild padded per-query arrays from flat (qid, docid) rows."""
    qids = list(dict.fromkeys(df["qid"].tolist()))
    qidx = {q: i for i, q in enumerate(qids)}
    Q = len(qids)
    cls1 = torch.zeros(Q, N, df["cls1"].shape[1])
    cls2 = torch.zeros(Q, N, df["cls2"].shape[1])
    final = torch.zeros(Q, N)
    valid = torch.zeros(Q, N, dtype=torch.bool)
    counts = [0] * Q
    for i, q in enumerate(df["qid"]):
        j = qidx[q]
        c = counts[j]
        if c < N:
            cls1[j, c] = torch.from_numpy(df["cls1"][i].astype(np.float32))
            cls2[j, c] = torch.from_numpy(df["cls2"][i].astype(np.float32))
            final[j, c] = float(df["final"][i])
            valid[j, c] = True
            counts[j] += 1
    return qids, cls1, cls2, final, valid


def build_gains(qrels, qids, doc_lists, N):
    Q = len(qids)
    gains = torch.zeros(Q, N)
    for i, q in enumerate(qids):
        for j, d in enumerate(doc_lists[q]):
            gains[i, j] = float(qrels.get(q, {}).get(d, 0))
    sorted_g, _ = gains.sort(dim=1, descending=True)
    disc = 1.0 / torch.log2(torch.arange(2, EVAL_K + 2, dtype=torch.float32))
    idcg = (sorted_g[:, :EVAL_K] * disc).sum(1).clamp(min=1e-9)
    return gains, idcg


def ndcg_from_scores(scores, gains, valid, idcg, k=EVAL_K):
    """scores/gains/valid: (Q,N) -> nDCG@k per query."""
    s = scores.masked_fill(~valid, -1e9)
    idx = s.argsort(dim=1, descending=True)
    g = torch.gather(gains, 1, idx)[:, :k]
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
    return (g * disc).sum(1) / idcg


def counterfactual_utility(cur, new, cand, valid, gains, idcg, chunk=256):
    """u[q,i] = nDCG(cur with cur[q,i] := new[q,i]) - nDCG(cur), where cand."""
    Q, N = cur.shape
    out = torch.zeros_like(cur)
    eye = torch.eye(N, dtype=torch.bool)
    for a in range(0, Q, chunk):
        b = min(Q, a + chunk)
        c, n_, v, g, ig = cur[a:b], new[a:b], valid[a:b], gains[a:b], idcg[a:b]
        base = ndcg_from_scores(c, g, v, ig)
        cf = torch.where(eye[None], n_[:, None, :].expand(-1, N, -1), c[:, None, :].expand(-1, N, -1))
        nd = ndcg_from_scores(
            cf.reshape(-1, N), g[:, None, :].expand(-1, N, -1).reshape(-1, N),
            v[:, None, :].expand(-1, N, -1).reshape(-1, N), ig.repeat_interleave(N))
        out[a:b] = torch.where(
            cand[a:b], nd.reshape(b - a, N) - base[:, None], torch.zeros(b - a, N))
    return out


def features_at(cur, prev, active):
    """Reduced scalar feature set, only already-paid-for information."""
    valid_f = torch.where(active, cur, torch.zeros_like(cur))
    cnt = active.float().sum(1, keepdim=True).clamp(min=1)
    mean = valid_f.sum(1, keepdim=True) / cnt
    var = ((valid_f - mean).masked_fill(~active, 0) ** 2).sum(1, keepdim=True) / cnt
    z = (cur - mean) / var.sqrt().clamp(min=1e-6)
    order = cur.masked_fill(~active, -1e9).argsort(1, descending=True).argsort(1).float()
    rank_frac = order / cur.shape[1]
    return torch.stack([cur, rank_frac, z, prev], dim=-1)


def rollout_sim(S, valid, r, prio_fn, collect=None):
    Q, N, C = S.shape
    cur = S[..., 0].clone()
    active = valid.clone()
    depth = torch.zeros(Q, N, dtype=torch.long)
    for s in range(C - 1):
        if not bool(active.any()):
            break
        prev = S[..., s - 1] if s > 0 else cur
        if collect is not None:
            collect(s, active, cur, prev)
        if r >= 1.0:
            sel = active
        else:
            prio = prio_fn(s, active, cur, prev)
            prio = prio.masked_fill(~active, -1e9)
            counts = active.sum(1, keepdim=True)
            k = (r * counts).round().clamp(min=1)
            rank = prio.argsort(dim=1, descending=True).argsort(dim=1)
            sel = active & (rank < k)
        depth = torch.where(sel, torch.full_like(depth, s + 1), depth)
        cur = torch.where(sel, S[..., s + 1], cur)
        active = sel
    return cur, depth


def train_exit_head(X, y, mask, seed):
    """Train an exit head with a QUERY-LEVEL split (post-review fix).

    The previous version flattened all candidates and split randomly, so
    candidates from the same query could appear in both train and validation,
    inflating validation performance (query leakage). Now the split is over
    queries: validation queries are entirely unseen during training."""
    seed_everything(seed)
    head = MLP(X.shape[-1])
    opt = torch.optim.AdamW(head.parameters(), lr=EXIT_LR, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    n_q = X.shape[0]
    qperm = torch.randperm(n_q, generator=g)
    n_val_q = max(1, int(0.1 * n_q))
    val_q, tr_q = qperm[:n_val_q], qperm[n_val_q:]
    tr_rows = torch.nonzero(mask[tr_q], as_tuple=True)
    Xf, yf = X[tr_q][tr_rows].float(), y[tr_q][tr_rows].float()
    val_rows = torch.nonzero(mask[val_q], as_tuple=True)
    Xv, yv = X[val_q][val_rows].float(), y[val_q][val_rows].float()
    if len(Xf) == 0 or len(Xv) == 0:
        raise RuntimeError("empty train or validation set after query-level split")
    best, best_state, stale = float("inf"), None, 0
    for _ in range(EXIT_EPOCHS):
        head.train()
        perm_t = torch.randperm(len(Xf), generator=g)
        for a in range(0, len(perm_t), EXIT_BATCH):
            b = perm_t[a:a + EXIT_BATCH]
            loss = F.smooth_l1_loss(head(Xf[b]), yf[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vl = F.smooth_l1_loss(head(Xv), yv, reduction="sum").item() / len(yv)
        if vl < best - 1e-7:
            best, stale = vl, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= EXIT_PATIENCE:
                break
    head.load_state_dict(best_state)
    return head.eval(), best


def collect_states(S, cls_list, valid, gains, idcg, train_q):
    """Rollouts with current-rank policy at several survival rates; store states."""
    data = {s: {"x": [], "u": [], "g": [], "i": []} for s in range(S.shape[-1] - 1)}
    for r in SURVIVALS:
        states = {s: {"active": None} for s in range(S.shape[-1] - 1)}

        def collect(s, active, cur, prev):
            states[s]["active"] = active.clone()
            states[s]["cur"] = cur.clone()
            states[s]["prev"] = prev.clone()

        rollout_sim(S[train_q], valid[train_q], r, lambda s, a, c, p: c, collect=collect)
        for s in range(S.shape[-1] - 1):
            act = states[s]["active"]
            qq, ii = torch.nonzero(act, as_tuple=True)
            if len(qq) == 0:
                continue
            cur, prev = states[s]["cur"], states[s]["prev"]
            u = counterfactual_utility(cur, S[train_q][..., -1], act, valid[train_q], gains[train_q], idcg[train_q])
            x = features_at(cur, prev, act)
            data[s]["x"].append(x[qq, ii])
            data[s]["u"].append(u[qq, ii])
            data[s]["g"].append(qq)  # group = query index within the rollout batch
            data[s]["i"].append(ii)
    return data


def train_gain_ranker(cls_s, feats, u, groups, seed):
    """Pairwise ranknet on (cls, features) with utility-difference weights."""
    seed_everything(seed)
    X = torch.cat([cls_s, feats], dim=-1)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
    Xn = (X - mu) / sd
    model = MLP(X.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=RANK_LR, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    uniq, inv = groups.unique(return_inverse=True)
    n_groups = len(uniq)
    for _ in range(RANK_EPOCHS):
        model.train()
        perm = torch.randperm(n_groups, generator=g)
        for a in range(0, n_groups, 64):
            batch_groups = perm[a:a + 64]
            sel_rows = torch.nonzero(torch.isin(inv, batch_groups), as_tuple=True)[0]
            if len(sel_rows) < 2:
                continue
            gi = inv[sel_rows]
            i = torch.randint(0, len(sel_rows), (len(sel_rows),), generator=g)
            j = torch.randint(0, len(sel_rows), (len(sel_rows),), generator=g)
            ok = (gi[i] == gi[j]) & (u[sel_rows[i]] > u[sel_rows[j]] + UTILITY_EPS)
            if not bool(ok.any()):
                continue
            si = model(Xn[sel_rows[i[ok]]])
            sj = model(Xn[sel_rows[j[ok]]])
            w = u[sel_rows[i[ok]]] - u[sel_rows[j[ok]]]
            loss = (w * F.softplus(-(si - sj))).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model, (mu, sd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/gainer_train/nfcorpus_train.npz")
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="models/gainer_heads")
    args = parser.parse_args()

    torch.set_num_threads(2)
    z = __import__("numpy").load(args.npz, allow_pickle=False)
    pools = {}
    pools_path = Path(args.npz).with_name(Path(args.npz).stem + "_pools.jsonl")
    with open(pools_path) as f:
        for line in f:
            row = json.loads(line)
            pools[row["qid"]] = row["doc_ids"]
    df = {k: z[k] for k in ["cls1", "cls2", "final", "qid", "docid"]}
    qrels = data.load_qrels(f"data/{args.dataset}/qrels/train.tsv")

    qids, cls1, cls2, final, valid = pad_per_query(df, args.top_k)
    doc_lists = {q: pools[q][: args.top_k] for q in qids}
    gains, idcg = build_gains(qrels, qids, doc_lists, args.top_k)
    print(f"{len(qids)} queries x {args.top_k} slots, valid pairs {int(valid.sum())}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(qids), generator=g)
    n_val = max(1, int(0.1 * len(qids)))
    val_q, train_q = perm[:n_val], perm[n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"trained_on": args.npz, "checkpoints_after_layer": [2, 4], "top_k": args.top_k,
            "head_hidden": HEAD_HIDDEN, "survivals": SURVIVALS}

    heads = []
    for ci, X in enumerate([cls1, cls2]):
        head, val_loss = train_exit_head(X, final, valid, args.seed + ci)
        heads.append(head)
        torch.save(head.state_dict(), out_dir / f"exit_head_{ci}.pt")
        print(f"exit head {ci}: val smooth-L1 {val_loss:.5f}", flush=True)

    with torch.no_grad():
        S = torch.stack([heads[0](cls1), heads[1](cls2), final], dim=-1)
        S = S.masked_fill(~valid[..., None], -1e4)
    data_c = collect_states(S, [cls1, cls2], valid, gains, idcg, train_q)
    for s in range(2):
        if not data_c[s]["x"]:
            print(f"stage {s}: no states collected", flush=True)
            continue
        X = torch.cat(data_c[s]["x"], dim=0)
        u = torch.cat(data_c[s]["u"], dim=0)
        gps = torch.cat(data_c[s]["g"], dim=0)
        iis = torch.cat(data_c[s]["i"], dim=0)
        cls_src = [cls1, cls2][s][train_q]
        cls_flat = cls_src[gps, iis]
        model, (mu, sd) = train_gain_ranker(cls_flat, X, u, gps, args.seed + 10 + s)
        torch.save({"state": model.state_dict(), "mu": mu, "sd": sd}, out_dir / f"gain_ranker_{s}.pt")
        print(f"gain ranker {s}: trained on {len(u)} states", flush=True)

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("saved to", out_dir, flush=True)


if __name__ == "__main__":
    main()
