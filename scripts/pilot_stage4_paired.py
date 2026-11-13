"""Stage-4 paired pilot: Serial vs Wavefront on the FROZEN protocol.

Run BEFORE the formal replacement rerun. Verifies, on the real model:

1. Offline policy parity — per-query reference (collect_trace),
   SerialScheduler and single-request WavefrontScheduler produce identical
   exit_layer (exact) and scores (< 1e-4) under the frozen thresholds.
2. ONLINE cross-query parity — the actual serving path: WavefrontScheduler
   run online on ALL pilot requests with the real arrival trace. After the
   run, every request's exit_layer and scores are compared against the
   per-query reference, and each checkpoint's decision margin
   (|score - threshold|) is reported so any flipped exit decision can be
   attributed to batch-composition float noise (tiny margin) or to a genuine
   semantic divergence (large margin).
3. Paired online serving with a four-point timeline per request:
   T_activation = activation - arrival (scheduling wait + embedding)
   T_ready_queue = first_layer_event - activation (waiting to join a batch)
   T_end_to_end = finish - arrival
   plus ranking quality (nDCG@10 from the SAME run's scores) and the compute
   ledger (row-layer executions, scored rows, embed time) for both arms.

Pilot qids are drawn with an independent seed from the FROZEN load_qids
(never re-sorted); the arrival order is the sampled (random) order.

Usage:
    python scripts/pilot_stage4_paired.py --stagger-ms 500 [--reps 2]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_cast import data, metrics  # noqa: E402
from cpu_cast.active_set import ActiveSetRanker  # noqa: E402
from cpu_cast.logging_utils import JsonlLogger  # noqa: E402
from cpu_cast.scheduler import Request, SerialScheduler, WavefrontScheduler  # noqa: E402
from bench_stage4_load import build_requests, calibrate_thresholds, \
    collect_trace, git_state, trim_memory  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "manifests" / "stage4_protocol.json"
MODEL = "models/ms-marco-MiniLM-L-6-v2"

THRESHOLDS = None
CHECKPOINTS = (1, 3)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen():
    if not PROTOCOL.exists():
        raise SystemExit("protocol not frozen — run freeze_stage4_protocol.py first")
    p = json.loads(PROTOCOL.read_text())
    run_path = REPO / p["candidate_run"]
    if sha256_file(run_path) != p["candidate_run_sha256"]:
        raise SystemExit("candidate run hash mismatch — data changed since freeze")
    qrels_path = REPO / "data" / p["dataset"] / "qrels" / "test.tsv"
    if sha256_file(qrels_path) != p["qrels_sha256"]:
        raise SystemExit("qrels hash mismatch — data changed since freeze")
    return p


def reference_with_margins(ranker, r, thresholds, checkpoints):
    """Per-query reference execution (always-compacted, fixed thresholds)
    that also records, per row, its distance to the threshold at the
    checkpoint that decided it."""
    L = ranker.num_layers
    n = r.n
    with torch.inference_mode():
        hidden, ext_mask = ranker.embed(r.input_ids, r.attention_mask,
                                        r.token_type_ids)
        done = torch.zeros(n, dtype=torch.bool)
        exit_layer = torch.full((n,), L, dtype=torch.long)
        scores = torch.zeros(n)
        row_margin = torch.full((n,), float("inf"))
        for layer in range(L):
            idx = torch.nonzero(~done, as_tuple=True)[0]
            if idx.numel() == 0:
                break
            h = hidden.index_select(0, idx)
            m = ext_mask.index_select(0, idx)
            h = ranker.forward_layer(h, m, layer)
            hidden = hidden.index_copy(0, idx, h)
            if layer in checkpoints and thresholds is not None:
                cur = ranker.score_hidden(h).reshape(-1)
                keep = cur >= thresholds[layer]
                x_idx = idx[~keep]
                exit_layer[x_idx] = layer + 1
                scores[x_idx] = cur[~keep]
                row_margin[idx] = torch.minimum(
                    row_margin[idx], (cur - thresholds[layer]).abs())
                done[x_idx] = True
        idx = torch.nonzero(~done, as_tuple=True)[0]
        if idx.numel():
            scores[idx] = ranker.score_hidden(hidden[idx])
    return scores, exit_layer, row_margin


def offline_parity(ranker, load_reqs, thresholds, checkpoints):
    """Per-query reference vs Serial offline vs single-request Wavefront."""
    max_s, exit_mm = 0.0, 0
    with torch.inference_mode():
        for r in load_reqs:
            ref = collect_trace(ranker, r, checkpoints, thresholds)
            sr = Request(r.qid, r.input_ids, r.attention_mask, r.token_type_ids)
            SerialScheduler(ranker, thresholds=thresholds,
                            checkpoints=checkpoints).run([sr], online=False)
            wr = Request(r.qid, r.input_ids, r.attention_mask, r.token_type_ids)
            WavefrontScheduler(ranker, checkpoints=checkpoints, signal="pooler",
                               keep_fn=lambda s, layer: s >= thresholds[layer],
                               ).run([wr])
            max_s = max(max_s,
                        float(np.max(np.abs(np.array(ref["scores"])
                                            - sr.scores.numpy()))),
                        float(np.max(np.abs(sr.scores.numpy()
                                            - wr.scores.numpy()))))
            exit_mm += sum(1 for a, b, c in zip(ref["exit_layer"],
                                                sr.exit_layer.tolist(),
                                                wr.exit_layer.tolist())
                           if a != b or a != c)
    return max_s, exit_mm


def online_cross_query_parity(ranker, load_reqs, thresholds, checkpoints,
                              stagger_ms):
    """The REAL serving path: online Wavefront over all requests with the
    actual arrival trace, then per-query/per-candidate comparison against
    the per-query reference, with decision margins."""
    reqs = [Request(r.qid, r.input_ids, r.attention_mask, r.token_type_ids)
            for r in load_reqs]
    for i, r in enumerate(reqs):
        r.arrival = i * stagger_ms / 1000.0
    sched = WavefrontScheduler(ranker, checkpoints=checkpoints, signal="pooler",
                               keep_fn=lambda s, layer: s >= thresholds[layer])
    t0 = time.perf_counter()
    sched.run(reqs, online=True)
    makespan_s = time.perf_counter() - t0

    flips, max_diff = 0, 0.0
    flip_margins, min_margin = [], {}
    by_qid = {r.qid: r for r in reqs}
    refs = {}
    with torch.inference_mode():
        for r in load_reqs:
            refs[r.qid] = reference_with_margins(ranker, r, thresholds,
                                                 checkpoints)
        for r in load_reqs:
            ref_s, ref_e, ref_m = refs[r.qid]
            wr = by_qid[r.qid]
            max_diff = max(max_diff, (ref_s - wr.scores).abs().max().item())
            mismatch = ref_e != wr.exit_layer
            flips += int(mismatch.sum())
            if mismatch.any():
                flip_margins.extend(ref_m[mismatch].tolist())
        for layer in checkpoints:
            ms_ = []
            for r in load_reqs:
                _, ref_e, ref_m = refs[r.qid]
                decided = torch.nonzero(ref_e == layer + 1, as_tuple=True)[0]
                if decided.numel():
                    ms_.append(float(ref_m[decided].min()))
            if ms_:
                min_margin[layer] = round(min(ms_), 6)
    rows_executed = sum(int(e["batch_rows"]) for e in sched.layer_events)
    rows_scored = sum(int(e["scored_rows"]) for e in sched.layer_events)
    return {
        "exit_mismatches": flips,
        "max_score_diff": round(max_diff, 8),
        "min_decision_margin_by_checkpoint": min_margin,
        "flip_margins": [round(x, 8) for x in flip_margins[:20]],
        "rows_executed": rows_executed,
        "rows_scored": rows_scored,
        "layer_events": len(sched.layer_events),
        "makespan_s": round(makespan_s, 3),
        "scores": {r.qid: r.scores.clone() for r in reqs},
    }


def serve_once(ranker, tokenizer, runs, queries, corpus, qrels, qids,
               max_length, stagger_ms, which):
    reqs = build_requests(runs, queries, corpus, qids, tokenizer, max_length)
    for i, r in enumerate(reqs):
        r.arrival = i * stagger_ms / 1000.0
    if which == "serial":
        sched = SerialScheduler(ranker, thresholds=THRESHOLDS,
                                checkpoints=CHECKPOINTS)
    else:
        sched = WavefrontScheduler(ranker, checkpoints=CHECKPOINTS,
                                   signal="pooler",
                                   keep_fn=lambda s, layer: s >= THRESHOLDS[layer])
    t0 = time.perf_counter()
    sched.run(reqs, online=True)
    total_s = time.perf_counter() - t0
    lats = np.array([(r.finish_time - t0) * 1000 - r.arrival * 1000
                     for r in reqs])
    t_act = np.array([(r.activation_time - t0) * 1000 - r.arrival * 1000
                      for r in reqs])
    t_rq = np.array([(r.first_exec_time - r.activation_time) * 1000
                     for r in reqs])
    trace = [{"qid": r.qid,
              "arrival_ms": round(r.arrival * 1000, 1),
              "submit_ms": round((r.submit_time - t0) * 1000, 1),
              "activation_ms": round((r.activation_time - t0) * 1000, 1),
              "first_exec_ms": round((r.first_exec_time - t0) * 1000, 1),
              "finish_ms": round((r.finish_time - t0) * 1000, 1),
              "latency_ms": round(float(lats[i]), 1)}
             for i, r in enumerate(reqs)]
    run_out = {r.qid: {dd: float(s) for dd, s in zip(r.doc_ids, r.scores)}
               for r in reqs}
    m = metrics.evaluate({q: qrels[q] for q in run_out}, run_out)
    return {
        "p50_ms": round(float(np.percentile(lats, 50)), 1),
        "p90_ms": round(float(np.percentile(lats, 90)), 1),
        "p95_ms": round(float(np.percentile(lats, 95)), 1),
        "mean_ms": round(float(lats.mean()), 1),
        "t_activation_mean_ms": round(float(t_act.mean()), 1),
        "t_activation_max_ms": round(float(t_act.max()), 1),
        "t_ready_queue_mean_ms": round(float(t_rq.mean()), 1),
        "t_ready_queue_max_ms": round(float(t_rq.max()), 1),
        "ndcg_10": round(m["ndcg_cut_10"], 4),
        "mrr": round(m["recip_rank"], 4),
        "rows_executed": sum(int(e["batch_rows"]) for e in sched.layer_events),
        "rows_scored": sum(int(e["scored_rows"]) for e in sched.layer_events),
        "embed_ms_total": round(sum(getattr(r, "embed_ms", 0.0) or 0.0
                                    for r in reqs), 1),
        "throughput_qps": round(len(reqs) / total_s, 3),
        "makespan_s": round(total_s, 3),
        "requests": trace,
    }


def main():
    global THRESHOLDS, CHECKPOINTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--num-load", type=int, default=20,
                        help="pilot draws N qids from the frozen load set")
    parser.add_argument("--pilot-seed", type=int, default=None,
                        help="independent seed for the pilot draw "
                             "(default: protocol seed + 1)")
    parser.add_argument("--parity-limit", type=int, default=20,
                        help="offline parity checked on the first N reqs")
    parser.add_argument("--stagger-ms", type=float, default=2000.0)
    parser.add_argument("--out", default="artifacts/runs/stage4_pilot.jsonl")
    args = parser.parse_args()

    p = load_frozen()
    torch.set_num_threads(p["threads"])
    d = data.load_split(f"data/{p['dataset']}")
    d = data.restrict_to_test_queries(d)
    runs = {}
    with (REPO / p["candidate_run"]).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["qid"] in d["queries"]:
                runs[row["qid"]] = row["candidates"]

    pilot_seed = args.pilot_seed if args.pilot_seed is not None \
        else p["seed"] + 1
    rng = np.random.default_rng(pilot_seed)
    sampled = [p["load_qids"][i] for i in
               rng.permutation(len(p["load_qids"]))[: args.num_load]]

    CHECKPOINTS = tuple(p["checkpoints"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ranker = ActiveSetRanker(MODEL)
    log = JsonlLogger(args.out, run_meta={
        "stage": "stage4", "version": "pilot-v2", **git_state(),
        "protocol_version": p["protocol_version"],
        "protocol_sha256": sha256_file(PROTOCOL),
        "pilot_seed": pilot_seed,
        "pilot_qids_in_arrival_order": sampled,
        "dataset": p["dataset"], "model": MODEL,
        "num_load": len(sampled), "reps": args.reps,
        "stagger_ms": args.stagger_ms, "threads": p["threads"],
        "cli": " ".join(sys.argv[1:]),
    })
    print(f"pilot qids (arrival order, seed {pilot_seed}): {sampled}",
          flush=True)

    calib_reqs = build_requests(runs, d["queries"], d["corpus"],
                                p["calib_qids"], tokenizer, p["max_length"])
    t0 = time.perf_counter()
    THRESHOLDS = calibrate_thresholds(ranker, calib_reqs, CHECKPOINTS,
                                      p["keep_frac"])
    log.write({"event": "calibration", "thresholds": THRESHOLDS,
               "calibration_s": round(time.perf_counter() - t0, 1)})
    print(f"thresholds: { {k: round(v, 4) for k, v in THRESHOLDS.items()} }",
          flush=True)

    load_reqs = build_requests(runs, d["queries"], d["corpus"], sampled,
                               tokenizer, p["max_length"])

    max_s, exit_mm = offline_parity(ranker, load_reqs[: args.parity_limit],
                                    THRESHOLDS, CHECKPOINTS)
    log.write({"event": "offline_parity", "max_score_diff": round(max_s, 8),
               "exit_mismatches": exit_mm,
               "queries": min(args.parity_limit, len(load_reqs))})
    print(f"offline parity: max_score_diff={max_s:.2e} "
          f"exit_mismatches={exit_mm}", flush=True)
    assert exit_mm == 0 and max_s < 1e-4, "offline parity FAILED"

    online = online_cross_query_parity(ranker, load_reqs, THRESHOLDS,
                                       CHECKPOINTS, args.stagger_ms)
    online_scores = online.pop("scores")
    log.write({"event": "online_cross_query_parity", **online})
    print(f"online cross-query parity: exit_mismatches="
          f"{online['exit_mismatches']} max_score_diff="
          f"{online['max_score_diff']:.2e} margins="
          f"{online['min_decision_margin_by_checkpoint']} "
          f"rows_executed={online['rows_executed']}", flush=True)
    if online["exit_mismatches"] > 0:
        print(f"WARNING: online wavefront exit decisions DIVERGE from the "
              f"per-query reference; flipped-row margins: "
              f"{online['flip_margins']}", flush=True)

    run_out = {r.qid: {dd: float(s) for dd, s in zip(r.doc_ids,
                                                     online_scores[r.qid])}
               for r in load_reqs}
    m = metrics.evaluate({q: d["qrels"][q] for q in run_out}, run_out)
    log.write({"event": "online_wavefront_quality",
               "ndcg_10": round(m["ndcg_cut_10"], 4),
               "mrr": round(m["recip_rank"], 4)})
    print(f"online wavefront quality: nDCG@10={m['ndcg_cut_10']:.4f}",
          flush=True)

    for rep in range(args.reps):
        for which in ("serial", "wavefront"):
            stats = serve_once(ranker, tokenizer, runs, d["queries"],
                               d["corpus"], d["qrels"], sampled,
                               p["max_length"], args.stagger_ms, which)
            log.write({"event": "real_online", "scheduler": which,
                       "rep": rep, "stagger_ms": args.stagger_ms,
                       "n_requests": len(sampled),
                       "p99_reported": False,
                       "p99_reason": "below frozen completion threshold",
                       **stats})
            print(f"rep{rep} {which}: p50={stats['p50_ms']}ms "
                  f"p95={stats['p95_ms']}ms "
                  f"act={stats['t_activation_mean_ms']}ms "
                  f"rq={stats['t_ready_queue_mean_ms']}ms "
                  f"ndcg={stats['ndcg_10']} "
                  f"rows={stats['rows_executed']} "
                  f"qps={stats['throughput_qps']}", flush=True)
            trim_memory()

    print("PILOT DONE", flush=True)


if __name__ == "__main__":
    main()
