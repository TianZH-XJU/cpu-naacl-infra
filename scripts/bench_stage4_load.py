"""Stage-4 load experiment: query-local vs cross-query wavefront scheduling.

Tests RQ3 / H3 (plan sections 10.3, 12): does cross-query candidate-layer
batching improve serving latency on a low-core CPU, and where is the
tail-latency knee from batching wait?

Method
1. Calibrate per-checkpoint score thresholds on held-out queries so the keep
   policy is batch-independent (keep = score >= threshold). This is required
   for cross-query composition: a top-keep-fraction rule would change its
   decisions depending on which queries share a batch.
2. Collect real per-query execution traces (per-layer active counts, exit
   layers) on load-test queries with a single length bucket (L=256).
3. Simulate serving under three arrival patterns (burst, staggered, 2-client
   closed loop) with a measured-cost model built from
   artifacts/runs/cost_grid.jsonl:
     - serial query-local execution (FIFO, one query at a time)
     - cross-query wavefront scheduling
4. Validate the simulator against real execution (burst arrival): real serial
   per-query service times and real wavefront makespan.

Cost model (from cost_grid.jsonl, torch FP32):
  layer1(N, L)  = layer_block(N, L) / 6                (per-layer forward)
  gather(n, k)  = compact_half(n, L) - layer1(k, L)    (index_select + index_copy)
  score(N)      ~= exit_head(N)                        (pooler+classifier proxy)
  embed(N)      = embed(N)

Limitations (documented): single length bucket, torch FP32 path only,
simulation ignores cache/frequency effects and CPU contention between the
benchmark process and other load.
"""
import argparse
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
from cpu_cast.scheduler import Request, WavefrontScheduler  # noqa: E402


# ---------------------------------------------------------------------------
# Cost table from the measured grid
# ---------------------------------------------------------------------------
class CostTable:
    def __init__(self, path, threads=2, seq_len=256):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("event") == "cost_grid" and r.get("threads") == threads \
                        and r.get("L") == seq_len:
                    rows.append(r)
        if not rows:
            raise RuntimeError(f"no cost_grid rows for threads={threads} L={seq_len}")
        self.layer1 = {}
        self.compact = {}
        self.embed = {}
        self.exit_head = {}
        for r in rows:
            n = r["N"]
            if r["kind"] == "layer_block":
                self.layer1[n] = r["p50_ms"] / 6.0
            elif r["kind"] == "compact_half":
                self.compact[n] = r["p50_ms"]
            elif r["kind"] == "embed":
                self.embed[n] = r["p50_ms"]
            elif r["kind"] == "exit_head":
                self.exit_head[n] = r["p50_ms"]
        self.ns = sorted(self.layer1)

    def _interp(self, table, n):
        if not table or n <= 0:
            return 0.0
        ns = sorted(table)
        if n <= ns[0]:
            return table[ns[0]]
        if n >= ns[-1]:
            return table[ns[-1]]
        import bisect
        i = bisect.bisect_left(ns, n)
        lo, hi = ns[i - 1], ns[i]
        if hi == lo:
            return table[lo]
        w = (n - lo) / (hi - lo)
        return table[lo] * (1 - w) + table[hi] * w

    def layer_ms(self, n):
        return self._interp(self.layer1, n)

    def gather_ms(self, n_from, n_keep):
        base = self._interp(self.compact, n_from)
        layer_at_keep = self._interp(self.layer1, n_keep)
        return max(0.0, base - layer_at_keep)

    def embed_ms(self, n):
        return self._interp(self.embed, n)

    def score_ms(self, n):
        return self._interp(self.exit_head, n)


# ---------------------------------------------------------------------------
# Arrival processes
# ---------------------------------------------------------------------------
class StaticArrivals:
    def __init__(self, qids, times):
        self.items = sorted(zip(qids, times), key=lambda x: x[1])
        self.i = 0

    def due(self, clock):
        out = []
        while self.i < len(self.items) and self.items[self.i][1] <= clock + 1e-12:
            out.append(self.items[self.i])
            self.i += 1
        return out

    def next_time(self):
        return self.items[self.i][1] if self.i < len(self.items) else None

    def notify_done(self, qid, t):
        pass


class ClosedLoopArrivals:
    """n_clients closed-loop clients; each submits its next query as soon as
    its previous one completes (client_free = completion time)."""

    def __init__(self, qids, n_clients):
        self.client_q = [[qids[i] for i in range(c, len(qids), n_clients)]
                         for c in range(n_clients)]
        self.ptr = [0] * n_clients
        self.client_free = [0.0] * n_clients
        self.pending = []

    def due(self, clock):
        out = []
        for c, q_list in enumerate(self.client_q):
            while self.ptr[c] < len(q_list) and self.client_free[c] <= clock + 1e-12:
                qid = q_list[self.ptr[c]]
                t = max(self.client_free[c], clock)
                out.append((qid, t))
                self.pending.append((qid, t))
                self.ptr[c] += 1
        return out

    def next_time(self):
        pending_t = min((t for _, t in self.pending), default=None)
        future = [self.client_free[c] for c in range(len(self.client_q))
                  if self.ptr[c] < len(self.client_q[c])]
        cand = future + ([pending_t] if pending_t is not None else [])
        return min(cand) if cand else None

    def notify_done(self, qid, t):
        self.pending = [(q, tt) for q, tt in self.pending if q != qid]
        for c, q_list in enumerate(self.client_q):
            if qid in q_list:
                self.client_free[c] = t


# ---------------------------------------------------------------------------
# Requests, traces, calibration
# ---------------------------------------------------------------------------
def build_requests(runs, queries, corpus, qids, tokenizer, max_length):
    reqs = []
    for qid in qids:
        doc_ids = sorted(runs[qid].keys())
        pairs = [(queries[qid], corpus[d]["title"] + " " + corpus[d]["text"])
                 for d in doc_ids]
        enc = tokenizer(pairs, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
        r = Request(qid, enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
        r.doc_ids = doc_ids
        reqs.append(r)
    return reqs


def collect_trace(ranker, req, checkpoints, thresholds, compact_threshold=0.75):
    """Run one query with the fixed-threshold keep policy; record the trace
    used by both the simulator and the real wavefront run."""
    L = ranker.num_layers
    n = req.n
    with torch.inference_mode():
        hidden, ext_mask = ranker.embed(req.input_ids, req.attention_mask,
                                        req.token_type_ids)
        done = torch.zeros(n, dtype=torch.bool)
        exit_layer = torch.full((n,), L, dtype=torch.long)
        scores = torch.zeros(n)
        n_by_layer, compactions = [], []
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
            n_by_layer.append(idx.numel())
            if compacted:
                compactions.append((layer, n, idx.numel()))
            if layer in checkpoints:
                cur_all = ranker.score_hidden(h)
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
        idx = torch.nonzero(~done, as_tuple=True)[0]
        if idx.numel():
            scores[idx] = ranker.score_hidden(hidden[idx])
            done[idx] = True
    return {
        "n_by_layer": n_by_layer,
        "compactions": compactions,
        "exit_layer": exit_layer.tolist(),
        "scores": scores.tolist(),
        "n_final": int((exit_layer == L).sum()),
    }


def calibrate_thresholds(ranker, calib_reqs, checkpoints, keep_frac):
    """Threshold = (1 - keep_frac)-quantile of checkpoint scores over the
    calibration queries; batch-independent by construction."""
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


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def service_time_serial(tr, cost):
    n_total = tr["n_by_layer"][0] if tr["n_by_layer"] else 0
    t = cost.embed_ms(n_total)
    comp_map = {layer: (nf, nt) for layer, nf, nt in tr["compactions"]}
    for layer, n_active in enumerate(tr["n_by_layer"]):
        if layer in comp_map:
            nf, nt = comp_map[layer]
            t += cost.gather_ms(nf, nt)
        t += cost.layer_ms(n_active)
    t += cost.score_ms(tr["n_final"])
    return t


def simulate_serial(traces, arrivals, cost):
    """Single worker, FIFO queue, one query at a time."""
    completions, free = {}, 0.0
    for qid, arr in sorted(arrivals, key=lambda x: x[1]):
        start = max(arr, free)
        completions[qid] = start + service_time_serial(traces[qid], cost)
        free = completions[qid]
    return completions


def simulate_serial_closed(traces, qids, n_clients, cost):
    """Closed-loop serial: single worker FIFO; clients resubmit on completion."""
    client_q = [[qids[i] for i in range(c, len(qids), n_clients)]
                for c in range(n_clients)]
    ptr = [0] * n_clients
    client_free = [0.0] * n_clients
    queue = []
    completions = {}
    clock = 0.0
    free = 0.0
    total = len(qids)
    while len(completions) < total:
        for c in range(n_clients):
            while ptr[c] < len(client_q[c]) and client_free[c] <= clock + 1e-12:
                queue.append((client_q[c][ptr[c]], max(client_free[c], clock)))
                ptr[c] += 1
        if not queue:
            pending_clients = [client_free[c] for c in range(n_clients)
                               if ptr[c] < len(client_q[c])]
            if not pending_clients:
                break
            clock = max(clock, min(pending_clients))
            continue
        queue.sort(key=lambda x: x[1])
        qid, arr = queue.pop(0)
        start = max(arr, free)
        completions[qid] = start + service_time_serial(traces[qid], cost)
        free = completions[qid]
        clock = free
        for c in range(n_clients):
            if qid in client_q[c]:
                client_free[c] = free
    return completions


def simulate_wavefront(traces, arrivals, cost, checkpoints, num_layers,
                       max_batch=256):
    """Event simulation of the wavefront scheduler loop with measured costs."""
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
            n = traces[qid]["n_by_layer"][0]
            state[qid] = {"next": [0] * n, "done": [False] * n}
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
        t = cost.layer_ms(n_batch)
        if len(per_req) > 1:
            t += sum(cost.gather_ms(traces[q]["n_by_layer"][0], k)
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


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def lat_stats(completions, arrivals):
    """completions and arrival times are in milliseconds (cost-table units)."""
    lats = [completions[q] - t for q, t in arrivals]
    starts = [t for _, t in arrivals]
    return {
        "p50_ms": round(float(np.percentile(lats, 50)), 1),
        "p95_ms": round(float(np.percentile(lats, 95)), 1),
        "p99_ms": round(float(np.percentile(lats, 99)), 1),
        "mean_ms": round(float(np.mean(lats)), 1),
        "makespan_ms": round(float(max(completions.values()) - min(starts)), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--run", default="data/candidate_runs/scifact/rrf_top100.jsonl")
    parser.add_argument("--model", default="models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cost-grid", default="artifacts/runs/cost_grid.jsonl")
    parser.add_argument("--num-queries", type=int, default=12)
    parser.add_argument("--num-calib", type=int, default=20)
    parser.add_argument("--keep-frac", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stagger-ms", type=float, default=250.0)
    parser.add_argument("--stagger-light-ms", type=float, default=2000.0)
    parser.add_argument("--out", default="artifacts/runs/bench_stage4_load.jsonl")
    args = parser.parse_args()

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
    log = JsonlLogger(args.out, run_meta={
        "stage": "stage4", "dataset": args.dataset, "model": args.model,
        "num_queries": args.num_queries, "keep_frac": args.keep_frac,
        "max_length": args.max_length, "threads": args.threads,
    })

    load_reqs = build_requests(runs, d["queries"], d["corpus"], qids, tokenizer,
                               args.max_length)
    calib_reqs = build_requests(runs, d["queries"], d["corpus"], calib_qids,
                                tokenizer, args.max_length)

    checkpoints = (1, 3)
    L = ranker.num_layers
    t0 = time.perf_counter()
    thresholds = calibrate_thresholds(ranker, calib_reqs, checkpoints, args.keep_frac)
    print(f"calibrated thresholds: { {k: round(v, 4) for k, v in thresholds.items()} } "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)
    log.write({"event": "calibration", "thresholds": thresholds,
               "keep_frac": args.keep_frac, "num_calib": len(calib_qids)})

    t0 = time.perf_counter()
    traces = {}
    for r in load_reqs:
        traces[r.qid] = collect_trace(ranker, r, checkpoints, thresholds)
    print(f"traces collected ({time.perf_counter() - t0:.1f}s)", flush=True)
    kept = [sum(1 for e in t["exit_layer"] if e == L) for t in traces.values()]
    log.write({"event": "traces", "survivors_per_query": kept,
               "compactions": {q: t["compactions"] for q, t in traces.items()}})
    print(f"survivors per query (of 100): {kept}", flush=True)

    run_out = {r.qid: {dd: s for dd, s in zip(r.doc_ids, traces[r.qid]["scores"])}
               for r in load_reqs}
    m = metrics.evaluate({q: d["qrels"][q] for q in run_out}, run_out)
    log.write({"event": "quality", "ndcg_10": round(m["ndcg_cut_10"], 4),
               "mrr": round(m["recip_rank"], 4)})
    print(f"policy quality: nDCG@10={m['ndcg_cut_10']:.4f} MRR={m['recip_rank']:.4f}",
          flush=True)

    cost = CostTable(args.cost_grid, threads=args.threads, seq_len=args.max_length)
    print(f"cost table: layer1(100)={cost.layer_ms(100):.2f}ms "
          f"layer1(32)={cost.layer_ms(32):.2f}ms "
          f"gather(100,50)={cost.gather_ms(100, 50):.2f}ms", flush=True)

    results = {}
    burst = [(q, 0.0) for q in qids]
    stag = [(q, i * args.stagger_ms) for i, q in enumerate(qids)]
    stag_light = [(q, i * args.stagger_light_ms) for i, q in enumerate(qids)]

    for name, arr in [("burst", burst), ("staggered", stag),
                      ("staggered_light", stag_light)]:
        comps = simulate_serial(traces, arr, cost)
        results[("serial", name)] = lat_stats(comps, arr)
        log.write({"event": "sim_serving", "mode": "serial", "arrival": name,
                   **results[("serial", name)]})
        comps = simulate_wavefront(traces, StaticArrivals(qids, [t for _, t in arr]),
                                   cost, checkpoints, L)
        results[("wavefront", name)] = lat_stats(comps, arr)
        log.write({"event": "sim_serving", "mode": "wavefront", "arrival": name,
                   **results[("wavefront", name)]})

    comps = simulate_serial_closed(traces, qids, 2, cost)
    results[("serial", "closed2")] = lat_stats(comps, [(q, 0.0) for q in comps])
    log.write({"event": "sim_serving", "mode": "serial", "arrival": "closed2",
               **results[("serial", "closed2")]})
    comps = simulate_wavefront(traces, ClosedLoopArrivals(qids, 2), cost,
                               checkpoints, L)
    results[("wavefront", "closed2")] = lat_stats(comps, [(q, 0.0) for q in comps])
    log.write({"event": "sim_serving", "mode": "wavefront", "arrival": "closed2",
               **results[("wavefront", "closed2")]})

    for key, v in sorted(results.items()):
        print(f"sim {key[0]:>9} {key[1]:>10}: p50={v['p50_ms']:>8.1f} "
              f"p95={v['p95_ms']:>8.1f} p99={v['p99_ms']:>8.1f} "
              f"makespan={v['makespan_ms']:>8.1f}", flush=True)

    # ---- real-execution validation (burst arrival) ----
    t0 = time.perf_counter()
    for r in load_reqs:
        collect_trace(ranker, r, checkpoints, thresholds)
    real_serial_total = time.perf_counter() - t0
    sim_serial_total = sum(service_time_serial(traces[q], cost) for q in qids) / 1000.0
    log.write({"event": "validation_serial", "real_total_s": round(real_serial_total, 3),
               "sim_total_s": round(sim_serial_total, 3),
               "ratio": round(sim_serial_total / real_serial_total, 3)})
    print(f"validation serial: real={real_serial_total:.2f}s "
          f"sim={sim_serial_total:.2f}s (ratio {sim_serial_total / real_serial_total:.2f})",
          flush=True)

    t0 = time.perf_counter()
    sched = WavefrontScheduler(ranker, checkpoints=checkpoints, signal="pooler",
                               keep_fn=lambda s, layer: s >= thresholds[layer])
    sched.run(build_requests(runs, d["queries"], d["corpus"], qids, tokenizer,
                             args.max_length))
    real_wf_total = time.perf_counter() - t0
    sim_wf_ms = results[("wavefront", "burst")]["makespan_ms"]
    log.write({"event": "validation_wavefront", "real_total_s": round(real_wf_total, 3),
               "sim_makespan_s": round(sim_wf_ms / 1000, 3),
               "layer_events": sched.layer_events})
    print(f"validation wavefront: real={real_wf_total:.2f}s "
          f"sim_makespan={sim_wf_ms / 1000:.2f}s", flush=True)
    print(f"wavefront layer events (layer, rows, requests): {sched.layer_events}",
          flush=True)
    print("stage-4 load experiment complete", flush=True)


if __name__ == "__main__":
    main()
