"""Stage-4 load experiment v2 (post-review f0873cd4): unified cost ledger,
closed-loop arrivals, request-level traces, strong baseline.

Fixes applied from the review:
- P0-2 unified cost ledger: embed, single-layer forward_layer, score_hidden
  (pooler+classifier), exit_head (MLP), gather (index_select+index_copy) are
  measured INDEPENDENTLY in the cost grid; serial and wavefront simulations
  price every component from the same table. The layer_block/6 derivation is
  no longer used.
- P0-3 closed-loop arrivals are completion-driven: a client submits its next
  query only after its previous one completes (busy flag until notify_done).
  Submission/completion trajectories are logged.
- P0-4 unified timing boundary: tokenization and request construction happen
  ONCE outside every timed region and are reported separately; both real
  paths are timed on identical pre-built request objects.
- P0-5 request-level traces: every real request records arrival, submission,
  first-layer, completion times and a per-layer ledger (rows executed, rows
  scored, wall ms); real p50/p95/p99 are computed from these raw traces.
- P1-6 pooler path: the real experiment uses signal="pooler"; the scheduler
  parity test in scripts/test_wavefront.py now covers it.
- P1-7 strong baseline: cascade_shallow (TinyBERT-L2 prefilter -> MiniLM-L6)
  runs on the SAME queries with the same timing boundary.

Honest labeling: simulation results are labeled "sim"; real executions are
labeled "real". The sim is validated against real runs; sim p95/p99 are NOT
presented as measured serving latency.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast import data, metrics  # noqa: E402
from cpu_cast.active_set import ActiveSetRanker  # noqa: E402
from cpu_cast.logging_utils import JsonlLogger  # noqa: E402
from cpu_cast.scheduler import (Request, SerialScheduler,  # noqa: E402
                                WavefrontScheduler)


def trim_memory():
    """Release freed heap back to the OS between benchmark phases.

    The cgroup memory limit (~3.95 GiB) is tight; glibc malloc does not
    return freed arenas promptly, so a long multi-phase run accumulates RSS
    until the serial-online phase OOM-kills (observed on the first v4 run).
    """
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except OSError:
        pass


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()[:12]
    except Exception:
        return "unknown"


def git_state():
    """Provenance (review 2 P2): HEAD sha, dirty flag and a hash of the
    working-tree diff, so a run executed on a dirty tree is identifiable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()[:12]
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
        diff_sha = "clean"
        if dirty:
            diff = subprocess.check_output(["git", "diff"], text=True)
            diff_sha = hashlib.sha256(diff.encode()).hexdigest()[:12]
        return {"git_sha": sha, "git_dirty": dirty, "git_diff_sha": diff_sha}
    except Exception:
        return {"git_sha": "unknown", "git_dirty": None, "git_diff_sha": None}


# ---------------------------------------------------------------------------
# Cost table: direct measurements only (P0-2)
# ---------------------------------------------------------------------------
class CostTable:
    """All entries are direct p50 measurements from cost_grid.jsonl.

    kinds: embed, layer (single forward_layer), gather (index_select +
    index_copy, no forward), score (pooler + classifier), exit_head (MLP).
    """

    def __init__(self, path, threads=2, seq_len=256):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("event") == "cost_grid" and r.get("threads") == threads \
                        and r.get("L") == seq_len and not r["kind"].startswith("ort_"):
                    rows.append(r)
        if not rows:
            raise RuntimeError(f"no cost_grid rows for threads={threads} L={seq_len}")
        self.tables = {"embed": {}, "layer": {}, "gather": {}, "gather2": {},
                       "score": {}, "exit_head": {}}
        for r in rows:
            if r["kind"] == "gather2":
                # two-variable gather surface: (source rows N, kept rows K);
                # skip invalid K > N_src entries (randperm(N)[:K] yields only
                # N rows, so those measurements do not test what they claim)
                if r["K"] <= r["N"]:
                    self.tables["gather2"].setdefault(r["N"], {})[r["K"]] = r["p50_ms"]
            elif r["kind"] in self.tables:
                self.tables[r["kind"]][r["N"]] = r["p50_ms"]
        self.ns = sorted(self.tables["layer"])

    def _interp(self, kind, n):
        table = self.tables[kind]
        if not table or n <= 0:
            return 0.0
        ns = sorted(table)
        if n <= ns[0]:
            return table[ns[0]]
        if n >= ns[-1]:
            # Linear extrapolation beyond the measured edge: the scheduler
            # chunks merged batches at max_batch=256, so charges can exceed
            # the largest measured N; clamping there systematically
            # under-priced wavefront layer events.
            lo, hi = ns[-2], ns[-1]
            slope = (table[hi] - table[lo]) / (hi - lo)
            return table[hi] + slope * (n - hi)
        import bisect
        i = bisect.bisect_left(ns, n)
        lo, hi = ns[i - 1], ns[i]
        if hi == lo:
            return table[lo]
        w = (n - lo) / (hi - lo)
        return table[lo] * (1 - w) + table[hi] * w

    @staticmethod
    def _interp_axis(table, n):
        """1-D linear interp/extrapolation over one axis of a nested table."""
        if not table or n <= 0:
            return 0.0
        ns = sorted(table)
        if n <= ns[0]:
            return table[ns[0]]
        if n >= ns[-1]:
            lo, hi = ns[-2], ns[-1]
            slope = (table[hi] - table[lo]) / (hi - lo)
            return table[hi] + slope * (n - hi)
        import bisect
        i = bisect.bisect_left(ns, n)
        lo, hi = ns[i - 1], ns[i]
        w = (n - lo) / (hi - lo)
        return table[lo] * (1 - w) + table[hi] * w

    def layer_ms(self, n):
        return self._interp("layer", n)

    def gather_ms(self, n_from, n_keep):
        """C_gather(N, K): source rows N, kept rows K (review P0-1B).

        Bilinear over the measured gather2 surface; falls back to the
        single-variable gather table when gather2 is unavailable.
        """
        g2 = self.tables.get("gather2") or {}
        if not g2:
            return self._interp("gather", n_from)
        # interpolate on the K axis for each measured N, then on the N axis
        per_n = {n: self._interp_axis(row, n_keep) for n, row in g2.items()}
        return self._interp_axis(per_n, n_from)

    def embed_ms(self, n):
        return self._interp("embed", n)

    def score_ms(self, n):
        return self._interp("score", n)

    def exit_head_ms(self, n):
        return self._interp("exit_head", n)


# ---------------------------------------------------------------------------
# Arrival processes (P0-3: closed loop is completion-driven)
# ---------------------------------------------------------------------------
class StaticArrivals:
    def __init__(self, qids, times_ms):
        self.items = sorted(zip(qids, times_ms), key=lambda x: x[1])
        self.i = 0
        self.submitted = []

    def due(self, clock):
        out = []
        while self.i < len(self.items) and self.items[self.i][1] <= clock + 1e-9:
            out.append(self.items[self.i])
            self.submitted.append(self.items[self.i])
            self.i += 1
        return out

    def next_time(self):
        return self.items[self.i][1] if self.i < len(self.items) else None

    def notify_done(self, qid, t):
        pass


class ClosedLoopArrivals:
    """n_clients closed-loop clients. A client submits its next query ONLY
    after its previous one completes: busy[c] is set on submission and
    cleared in notify_done. All times in ms."""

    def __init__(self, qids, n_clients):
        self.client_q = [list(qids[c::n_clients]) for c in range(n_clients)]
        self.ptr = [0] * n_clients
        self.client_free = [0.0] * n_clients
        self.busy = [False] * n_clients
        self.pending = []
        self.submitted = []   # (qid, submit_ms) trajectory

    def due(self, clock):
        out = []
        for c, q_list in enumerate(self.client_q):
            if self.busy[c]:
                continue
            if self.ptr[c] < len(q_list) and self.client_free[c] <= clock + 1e-9:
                qid = q_list[self.ptr[c]]
                t = max(self.client_free[c], clock)
                out.append((qid, t))
                self.pending.append((qid, t))
                self.submitted.append((qid, t))
                self.ptr[c] += 1
                self.busy[c] = True
        return out

    def next_time(self):
        pending_t = min((t for _, t in self.pending), default=None)
        future = [self.client_free[c] for c in range(len(self.client_q))
                  if not self.busy[c] and self.ptr[c] < len(self.client_q[c])]
        cand = future + ([pending_t] if pending_t is not None else [])
        return min(cand) if cand else None

    def notify_done(self, qid, t):
        self.pending = [(q, tt) for q, tt in self.pending if q != qid]
        for c, q_list in enumerate(self.client_q):
            if qid in q_list:
                self.client_free[c] = t
                self.busy[c] = False


# ---------------------------------------------------------------------------
# Requests, traces, calibration
# ---------------------------------------------------------------------------
def build_requests(runs, queries, corpus, qids, tokenizer, max_length):
    """Tokenize ONCE; the returned requests are reused by every policy so all
    timed regions share the same preprocessing boundary."""
    reqs = []
    for qid in qids:
        doc_ids = sorted(runs[qid].keys())
        pairs = [(queries[qid], corpus[d]["title"] + " " + corpus[d]["text"])
                 for d in doc_ids]
        enc = tokenizer(pairs, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
        r = Request(qid, enc["input_ids"], enc["attention_mask"],
                    enc["token_type_ids"])
        r.doc_ids = doc_ids
        reqs.append(r)
    return reqs


def collect_trace(ranker, req, checkpoints, thresholds, compact_threshold=0.75):
    """Run one query with the fixed-threshold keep policy; record the honest
    per-layer ledger: exec_rows (rows actually forwarded), scored_rows
    (rows passed through score_hidden), compaction events."""
    L = ranker.num_layers
    n = req.n
    with torch.inference_mode():
        hidden, ext_mask = ranker.embed(req.input_ids, req.attention_mask,
                                        req.token_type_ids)
        done = torch.zeros(n, dtype=torch.bool)
        exit_layer = torch.full((n,), L, dtype=torch.long)
        scores = torch.zeros(n)
        ledger = []   # per layer: dict(layer, exec_rows, scored_rows, compacted)
        for layer in range(L):
            idx = torch.nonzero(~done, as_tuple=True)[0]
            if idx.numel() == 0:
                break
            compacted = idx.numel() < n and idx.numel() / n < compact_threshold
            if compacted:
                h = hidden.index_select(0, idx)
                m = ext_mask.index_select(0, idx)
                h = ranker.forward_layer(h, m, layer)
                hidden = hidden.index_copy(0, idx, h)
            else:
                hidden = ranker.forward_layer(hidden, ext_mask, layer)
                h = hidden
            entry = {"layer": layer, "exec_rows": int(h.shape[0]),
                     "scored_rows": 0, "compacted": bool(compacted)}
            if layer in checkpoints:
                cur_all = ranker.score_hidden(h)
                entry["scored_rows"] = int(cur_all.shape[0])
                cur_active = cur_all if compacted else cur_all[idx]
                keep_active = cur_active >= thresholds[layer]
                keep = torch.zeros(n, dtype=torch.bool)
                keep[idx] = keep_active
                cur_map = torch.zeros(n)
                cur_map[idx] = cur_active
                finished = ~keep & ~done
                exit_layer[finished] = layer + 1
                scores[finished] = cur_map[finished]
                done = done | finished
            ledger.append(entry)
        idx = torch.nonzero(~done, as_tuple=True)[0]
        if idx.numel():
            scores[idx] = ranker.score_hidden(hidden[idx])
            done[idx] = True
    return {
        "ledger": ledger,
        "exit_layer": exit_layer.tolist(),
        "scores": scores.tolist(),
        "n_final": int((exit_layer == L).sum()),
    }


def calibrate_thresholds(ranker, calib_reqs, checkpoints, keep_frac):
    """Threshold = (1 - keep_frac)-quantile of checkpoint scores over the
    calibration queries; batch-independent by construction. NOTE: the
    checkpoint-3 distribution is collected over FULL candidate sets, while at
    serving time checkpoint 3 sees checkpoint-1 survivors; realized keep rates
    are therefore reported separately (see realized_keep_rates)."""
    L = ranker.num_layers
    vals = {c: [] for c in checkpoints}
    with torch.inference_mode():
        for req in calib_reqs:
            hidden, ext_mask = ranker.embed(req.input_ids, req.attention_mask,
                                            req.token_type_ids)
            for layer in range(L):
                hidden = ranker.forward_layer(hidden, ext_mask, layer)
                if layer in checkpoints:
                    vals[layer].extend(ranker.score_hidden(hidden).tolist())
    return {c: float(np.quantile(vals[c], 1 - keep_frac)) for c in checkpoints}


def realized_keep_rates(traces, checkpoints, L):
    """Actual keep rate at each checkpoint over the SURVIVOR set (Q7)."""
    rates = {}
    for c in checkpoints:
        reached = 0
        kept = 0
        for tr in traces.values():
            # rows that reached checkpoint c: exited after c (i.e. exit_layer > c)
            reached += sum(1 for e in tr["exit_layer"] if e > c)
            # rows kept past checkpoint c: exit_layer > c+1 (rows exiting AT c
            # have exit_layer == c+1; survivors have exit_layer == L and are
            # already counted here — do NOT add n_final again)
            kept += sum(1 for e in tr["exit_layer"] if e > c + 1)
        rates[c] = {"reached": reached, "kept": kept,
                    "keep_rate": round(kept / reached, 4) if reached else None}
    return rates


# ---------------------------------------------------------------------------
# Simulation (clearly labeled; validated against real runs)
# ---------------------------------------------------------------------------
def service_time_serial(tr, cost):
    """Serial query-local service time from the unified ledger."""
    n_total = tr["ledger"][0]["exec_rows"] if tr["ledger"] else 0
    t = cost.embed_ms(n_total)
    for entry in tr["ledger"]:
        if entry["compacted"]:
            # gather priced as C_gather(source rows, kept rows): the serial
            # path gathers exec_rows rows out of the full n_total-row hidden
            # (review P0-1C — the source is the request's full tensor)
            t += cost.gather_ms(n_total, entry["exec_rows"])
        t += cost.layer_ms(entry["exec_rows"])
        if entry["scored_rows"]:
            t += cost.score_ms(entry["scored_rows"])
    t += cost.score_ms(tr["n_final"])
    return t


def simulate_serial(traces, arrivals, cost):
    """Single worker, FIFO queue, one query at a time. Times in ms."""
    completions, free = {}, 0.0
    for qid, arr in sorted(arrivals, key=lambda x: x[1]):
        start = max(arr, free)
        completions[qid] = start + service_time_serial(traces[qid], cost)
        free = completions[qid]
    return completions


def simulate_serial_closed(traces, arrivals, cost):
    """Closed-loop serial: single worker FIFO; clients resubmit on completion."""
    completions = {}
    free = 0.0
    queue = []
    total = len(set(q for q, _ in arrivals.submitted)) if arrivals.submitted else None
    while True:
        for qid, t in arrivals.due(free):
            queue.append((qid, t))
        if not queue:
            nxt = arrivals.next_time()
            if nxt is None:
                break
            free = max(free, nxt)
            continue
        queue.sort(key=lambda x: x[1])
        qid, arr = queue.pop(0)
        start = max(arr, free)
        completions[qid] = start + service_time_serial(traces[qid], cost)
        free = completions[qid]
        arrivals.notify_done(qid, completions[qid])
        if total is not None and len(completions) >= total:
            break
    return completions


def simulate_wavefront(traces, arrivals, cost, checkpoints, num_layers,
                       max_batch=256):
    """Event simulation of the wavefront scheduler loop, priced from the SAME
    ledger as the serial path. Times in ms."""
    L = num_layers
    state = {}
    completions = {}
    exits_by_layer = {}
    for qid, tr in traces.items():
        ex = {}
        for row, el in enumerate(tr["exit_layer"]):
            if el < L:
                ex.setdefault(el - 1, set()).add(row)
        exits_by_layer[qid] = ex

    clock = 0.0
    while True:
        for qid, t in arrivals.due(clock):
            n = traces[qid]["ledger"][0]["exec_rows"]
            state[qid] = {"next": [0] * n, "done": [False] * n}
            # P0-1A: the real scheduler embeds at activation INSIDE the timed
            # region; charge the same here so both paths carry embedding cost
            clock += cost.embed_ms(n)
        ready = {}
        for qid, st in state.items():
            for row, (nl, dn) in enumerate(zip(st["next"], st["done"])):
                if not dn and nl < L:
                    ready.setdefault(nl, []).append((qid, row))
        if not ready:
            nxt = arrivals.next_time()
            if nxt is None:
                break
            clock = max(clock, nxt)
            continue
        layer = min(ready, key=lambda k: (-len(ready[k]), k))
        rows = ready[layer][:max_batch]
        n_batch = len(rows)
        per_req = {}
        for q, _ in rows:
            per_req[q] = per_req.get(q, 0) + 1
        # same pricing as serial: layer cost + per-request gather
        t = cost.layer_ms(n_batch)
        # gather priced as C_gather(source rows, kept rows): each request
        # gathers k rows out of its full hidden (ledger layer-0 exec_rows = n)
        t += sum(cost.gather_ms(traces[q]["ledger"][0]["exec_rows"], k)
                 for q, k in per_req.items())
        clock += t
        by_q = {}
        for q, row in rows:
            by_q.setdefault(q, []).append(row)
        for q, rows_q in by_q.items():
            st = state[q]
            exited = exits_by_layer[q].get(layer, set()) if layer in checkpoints else set()
            for row in rows_q:
                if row in exited:
                    st["done"][row] = True
                else:
                    st["next"][row] = layer + 1
        if layer in checkpoints:
            clock += cost.score_ms(n_batch)
        if layer == L - 1:
            for q, rows_q in by_q.items():
                st = state[q]
                survivors = [row for row in rows_q if not st["done"][row]]
                if survivors:
                    clock += cost.score_ms(len(survivors))
                    for row in survivors:
                        st["done"][row] = True
        for q in list(state):
            if q not in completions and all(state[q]["done"]):
                completions[q] = clock
                arrivals.notify_done(q, clock)
    return completions


def lat_stats(completions, arrivals):
    """completions and arrival times in ms."""
    lats = [completions[q] - t for q, t in arrivals]
    starts = [t for _, t in arrivals]
    return {
        "p50_ms": round(float(np.percentile(lats, 50)), 1),
        "p95_ms": round(float(np.percentile(lats, 95)), 1),
        "p99_ms": round(float(np.percentile(lats, 99)), 1),
        "mean_ms": round(float(np.mean(lats)), 1),
        "makespan_ms": round(float(max(completions.values()) - min(starts)), 1),
    }


# ---------------------------------------------------------------------------
# Real baselines
# ---------------------------------------------------------------------------
def run_cascade_shallow(model_path, tinybert_path, reqs, tokenizer,
                        corpus, queries, keep_top=50, max_length=256):
    """cascade_shallow: TinyBERT scores all candidates -> keep top-K ->
    MiniLM-L6 full depth on survivors. Same queries, same timing boundary
    (tokenization of the full candidate set already done in reqs)."""
    from transformers import AutoModelForSequenceClassification
    tiny = AutoModelForSequenceClassification.from_pretrained(tinybert_path).eval()
    big = ActiveSetRanker(model_path)
    times, run_out = [], {}
    with torch.inference_mode():
        for r in reqs:
            t0 = time.perf_counter()
            tscores = tiny(input_ids=r.input_ids, attention_mask=r.attention_mask,
                           token_type_ids=r.token_type_ids).logits.squeeze(-1)
            k = min(keep_top, r.n)
            topk = torch.topk(tscores, k).indices
            kept_ids = [r.doc_ids[i] for i in topk.tolist()]
            enc = tokenizer(
                [(queries[r.qid], corpus[d]["title"] + " " + corpus[d]["text"])
                 for d in kept_ids],
                padding=True, truncation=True, max_length=max_length,
                return_tensors="pt")
            bscores = big.forward_full(enc["input_ids"], enc["attention_mask"],
                                       enc["token_type_ids"])
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)
            merged = {did: float(tscores[i]) for i, did in enumerate(r.doc_ids)}
            for i, did in enumerate(kept_ids):
                merged[did] = 1e3 + float(bscores[i])
            run_out[r.qid] = merged
    return times, run_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--run", default="data/candidate_runs/scifact/rrf_top100.jsonl")
    parser.add_argument("--model", default="models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--tinybert", default="models/ms-marco-TinyBERT-L-2-v2")
    parser.add_argument("--cost-grid", default="artifacts/runs/cost_grid.jsonl")
    parser.add_argument("--num-queries", type=int, default=12)
    parser.add_argument("--num-calib", type=int, default=20)
    parser.add_argument("--keep-frac", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stagger-ms", type=float, default=250.0)
    parser.add_argument("--stagger-light-ms", type=float, default=2000.0)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default="artifacts/runs/bench_stage4_load.jsonl")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:8]
    torch.set_num_threads(args.threads)
    d = data.load_split(f"data/{args.dataset}")
    d = data.restrict_to_test_queries(d)
    runs = {}
    with open(args.run, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["qid"] in d["queries"]:
                runs[row["qid"]] = row["candidates"]
    all_qids = list(runs.keys())
    qids = all_qids[: args.num_queries]
    calib_qids = all_qids[args.num_queries: args.num_queries + args.num_calib]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ranker = ActiveSetRanker(args.model)
    L = ranker.num_layers
    log = JsonlLogger(args.out, run_meta={
        "stage": "stage4", "version": "v4", "run_id": run_id,
        **git_state(),
        "dataset": args.dataset, "model": args.model,
        "num_queries": args.num_queries, "keep_frac": args.keep_frac,
        "max_length": args.max_length, "threads": args.threads,
        "cli": " ".join(sys.argv[1:]),
    })

    # ---- preprocessing OUTSIDE all timed regions (P0-4) ----
    t0 = time.perf_counter()
    load_reqs = build_requests(runs, d["queries"], d["corpus"], qids, tokenizer,
                               args.max_length)
    calib_reqs = build_requests(runs, d["queries"], d["corpus"], calib_qids,
                                tokenizer, args.max_length)
    tok_s = time.perf_counter() - t0
    log.write({"event": "preprocessing", "tokenize_build_s": round(tok_s, 3),
               "num_load": len(load_reqs), "num_calib": len(calib_reqs)})
    print(f"preprocessing (outside timed regions): {tok_s:.1f}s", flush=True)

    checkpoints = (1, 3)
    t0 = time.perf_counter()
    thresholds = calibrate_thresholds(ranker, calib_reqs, checkpoints, args.keep_frac)
    log.write({"event": "calibration", "thresholds": thresholds,
               "keep_frac": args.keep_frac, "num_calib": len(calib_qids),
               "calibration_s": round(time.perf_counter() - t0, 1)})
    print(f"thresholds: { {k: round(v, 4) for k, v in thresholds.items()} }", flush=True)

    t0 = time.perf_counter()
    traces = {}
    for r in load_reqs:
        traces[r.qid] = collect_trace(ranker, r, checkpoints, thresholds)
    trace_s = time.perf_counter() - t0
    kept = [sum(1 for e in t["exit_layer"] if e == L) for t in traces.values()]
    rates = realized_keep_rates(traces, checkpoints, L)
    log.write({"event": "traces", "survivors_per_query": kept,
               "realized_keep_rates": rates,
               "ledger": {q: t["ledger"] for q, t in traces.items()}})
    print(f"traces ({trace_s:.1f}s); survivors/query: {kept}; "
          f"realized keep rates: { {k: v['keep_rate'] for k, v in rates.items()} }",
          flush=True)

    run_out = {r.qid: {dd: s for dd, s in zip(r.doc_ids, traces[r.qid]["scores"])}
               for r in load_reqs}
    m = metrics.evaluate({q: d["qrels"][q] for q in run_out}, run_out)
    policy_ndcg = round(m["ndcg_cut_10"], 4)
    log.write({"event": "quality", "ndcg_10": policy_ndcg,
               "mrr": round(m["recip_rank"], 4)})
    print(f"active-set policy quality: nDCG@10={policy_ndcg}", flush=True)

    cost = CostTable(args.cost_grid, threads=args.threads, seq_len=args.max_length)
    log.write({"event": "cost_table", "layer100_ms": round(cost.layer_ms(100), 2),
               "layer32_ms": round(cost.layer_ms(32), 2),
               "gather100_ms": round(cost.gather_ms(100, 50), 2),
               "score100_ms": round(cost.score_ms(100), 3),
               "embed100_ms": round(cost.embed_ms(100), 2)})
    print(f"ledger: layer(100)={cost.layer_ms(100):.2f}ms "
          f"gather(100)={cost.gather_ms(100, 50):.2f}ms "
          f"score(100)={cost.score_ms(100):.3f}ms", flush=True)

    # ---- simulation (labeled sim; validated below) ----
    burst = [(q, 0.0) for q in qids]
    stag = [(q, i * args.stagger_ms) for i, q in enumerate(qids)]
    stag_light = [(q, i * args.stagger_light_ms) for i, q in enumerate(qids)]
    for name, arr in [("burst", burst), ("staggered", stag),
                      ("staggered_light", stag_light)]:
        comps = simulate_serial(traces, arr, cost)
        log.write({"event": "sim", "mode": "serial", "arrival": name,
                   **lat_stats(comps, arr)})
        comps = simulate_wavefront(traces, StaticArrivals(qids, [t for _, t in arr]),
                                   cost, checkpoints, L)
        log.write({"event": "sim", "mode": "wavefront", "arrival": name,
                   **lat_stats(comps, arr)})
    cl = ClosedLoopArrivals(qids, 2)
    comps = simulate_serial_closed(traces, cl, cost)
    log.write({"event": "sim", "mode": "serial", "arrival": "closed2",
               **lat_stats(comps, cl.submitted),
               "submit_trajectory": cl.submitted})
    cl = ClosedLoopArrivals(qids, 2)
    comps = simulate_wavefront(traces, cl, cost, checkpoints, L)
    log.write({"event": "sim", "mode": "wavefront", "arrival": "closed2",
               **lat_stats(comps, cl.submitted),
               "submit_trajectory": cl.submitted})
    print("sim done (see jsonl for stats)", flush=True)
    trim_memory()

    # ---- real validation, unified boundary (P0-4/P0-5) ----
    for rep in range(args.reps):
        t0 = time.perf_counter()
        for r in load_reqs:
            collect_trace(ranker, r, checkpoints, thresholds)
        real_serial_s = time.perf_counter() - t0
        log.write({"event": "real_serial", "rep": rep,
                   "total_s": round(real_serial_s, 3),
                   "sim_total_s": round(
                       sum(service_time_serial(traces[q], cost) for q in qids) / 1000, 3)})
        trim_memory()

    # real wavefront, offline batch, request-level traces
    for rep in range(args.reps):
        t0 = time.perf_counter()
        sched = WavefrontScheduler(ranker, checkpoints=checkpoints, signal="pooler",
                                   keep_fn=lambda s, layer: s >= thresholds[layer])
        sched.run(load_reqs)
        total_s = time.perf_counter() - t0
        lats = [(r.finish_time - t0) * 1000 for r in load_reqs]
        log.write({"event": "real_wavefront_offline", "rep": rep,
                   "total_s": round(total_s, 3),
                   "p50_ms": round(float(np.percentile(lats, 50)), 1),
                   "p95_ms": round(float(np.percentile(lats, 95)), 1),
                   "p99_ms": round(float(np.percentile(lats, 99)), 1),
                   "layer_events": sched.layer_events})
        print(f"real offline wavefront rep{rep}: p50={np.percentile(lats, 50):.0f}ms "
              f"p95={np.percentile(lats, 95):.0f}ms", flush=True)
        del sched
        trim_memory()

    # real wavefront, ONLINE staggered arrivals -> true request-level latency
    # (same deterministic arrival trace as the serial online scheduler below)
    arrival_trace_ms = [i * args.stagger_light_ms for i in range(len(load_reqs))]
    for rep in range(args.reps):
        online_reqs = build_requests(runs, d["queries"], d["corpus"], qids,
                                     tokenizer, args.max_length)
        for i, r in enumerate(online_reqs):
            r.arrival = arrival_trace_ms[i] / 1000.0
        sched = WavefrontScheduler(ranker, checkpoints=checkpoints, signal="pooler",
                                   keep_fn=lambda s, layer: s >= thresholds[layer])
        t0 = time.perf_counter()
        sched.run(online_reqs, online=True)
        total_s = time.perf_counter() - t0
        lats = [(r.finish_time - t0) * 1000 - r.arrival * 1000 for r in online_reqs]
        req_trace = [{"qid": r.qid, "arrival_ms": r.arrival * 1000,
                      "submit_ms": round(((r.submit_time or t0) - t0) * 1000, 1),
                      "first_layer_ms": round(((r.first_exec_time or t0) - t0) * 1000, 1),
                      "finish_ms": round(((r.finish_time or t0) - t0) * 1000, 1),
                      "latency_ms": round(float(lats[i]), 1),
                      "embed_ms": round(getattr(r, "embed_ms", 0.0) or 0.0, 3),
                      "layer_log": r.layer_log}
                     for i, r in enumerate(online_reqs)]
        log.write({"event": "real_wavefront_online", "rep": rep,
                   "stagger_ms": args.stagger_light_ms,
                   "arrival_trace_ms": arrival_trace_ms,
                   "total_s": round(total_s, 3),
                   "p50_ms": round(float(np.percentile(lats, 50)), 1),
                   "p95_ms": round(float(np.percentile(lats, 95)), 1),
                   "p99_ms": round(float(np.percentile(lats, 99)), 1),
                   "requests": req_trace,
                   "layer_events": sched.layer_events})
        print(f"real online wavefront rep{rep}: p50={np.percentile(lats, 50):.0f}ms "
              f"p95={np.percentile(lats, 95):.0f}ms p99={np.percentile(lats, 99):.0f}ms",
              flush=True)
        del sched, online_reqs, req_trace
        trim_memory()

    # real serial ONLINE scheduler on the SAME arrival trace (review 2 P0-2):
    # a real one-worker FIFO loop that sleeps until each arrival and executes
    # to completion — NOT a measured-service-time replay. Both schedulers see
    # identical deterministic arrival offsets, so the arrival trace matches.
    arrival_trace_ms = [i * args.stagger_light_ms for i in range(len(load_reqs))]
    for rep in range(args.reps):
        online_reqs_s = build_requests(runs, d["queries"], d["corpus"], qids,
                                       tokenizer, args.max_length)
        for i, r in enumerate(online_reqs_s):
            r.arrival = arrival_trace_ms[i] / 1000.0
        ssched = SerialScheduler(ranker, thresholds=thresholds,
                                 checkpoints=checkpoints)
        t0 = time.perf_counter()
        ssched.run(online_reqs_s, online=True)
        total_s = time.perf_counter() - t0
        lats = [(r.finish_time - t0) * 1000 - r.arrival * 1000
                for r in online_reqs_s]
        req_trace = [{"qid": r.qid, "arrival_ms": r.arrival * 1000,
                      "submit_ms": round(((r.submit_time or t0) - t0) * 1000, 1),
                      "first_exec_ms": round(((r.first_exec_time or t0) - t0) * 1000, 1),
                      "finish_ms": round(((r.finish_time or t0) - t0) * 1000, 1),
                      "latency_ms": round(float(lats[i]), 1),
                      "layer_log": r.layer_log}
                     for i, r in enumerate(online_reqs_s)]
        log.write({"event": "real_serial_online", "rep": rep,
                   "mode": "real_online_scheduler",
                   "stagger_ms": args.stagger_light_ms,
                   "total_s": round(total_s, 3),
                   "p50_ms": round(float(np.percentile(lats, 50)), 1),
                   "p95_ms": round(float(np.percentile(lats, 95)), 1),
                   "p99_ms": round(float(np.percentile(lats, 99)), 1),
                   "requests": req_trace})
        print(f"real online serial rep{rep}: p50={np.percentile(lats, 50):.0f}ms "
              f"p95={np.percentile(lats, 95):.0f}ms p99={np.percentile(lats, 99):.0f}ms",
              flush=True)
        del ssched, online_reqs_s, req_trace
        trim_memory()

    # ---- strong baseline: cascade_shallow on the same queries (P1-7) ----
    t0 = time.perf_counter()
    cas_times, cas_run = run_cascade_shallow(
        args.model, args.tinybert, load_reqs, tokenizer, d["corpus"], d["queries"],
        keep_top=50, max_length=args.max_length)
    cas_total = time.perf_counter() - t0
    cas_m = metrics.evaluate({q: d["qrels"][q] for q in cas_run}, cas_run)
    log.write({"event": "baseline_cascade_shallow", "keep_top": 50,
               "ndcg_10": round(cas_m["ndcg_cut_10"], 4),
               "mrr": round(cas_m["recip_rank"], 4),
               "total_s": round(cas_total, 3),
               "p50_ms": round(float(np.percentile(cas_times, 50)), 1),
               "p95_ms": round(float(np.percentile(cas_times, 95)), 1)})
    print(f"cascade_shallow: nDCG@10={cas_m['ndcg_cut_10']:.4f} "
          f"total={cas_total:.2f}s p50={np.percentile(cas_times, 50):.0f}ms", flush=True)

    print("stage-4 load experiment v2 complete", flush=True)


if __name__ == "__main__":
    main()
