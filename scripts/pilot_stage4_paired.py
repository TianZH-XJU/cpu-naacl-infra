"""Stage-4 paired pilot: Serial vs Wavefront on the FROZEN protocol.

Runs BEFORE the formal replacement rerun. Verifies, on the real model:

1. Policy parity — per-query reference (collect_trace), SerialScheduler and
   WavefrontScheduler produce identical exit_layer (exact) and scores
   (< 1e-4) for every load query under the frozen thresholds.
2. Paired online serving — both schedulers serve the SAME qids, candidates,
   thresholds and arrival offsets (stagger level from the frozen protocol)
   for a small number of reps. Reports p50/p90/p95 (NOT p99: the pilot does
   not reach the completion count the frozen reporting rule requires),
   throughput, and mean queue time (first_exec - submit).

Usage:
    python scripts/pilot_stage4_paired.py [--reps 2] [--stagger-ms 2000]
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
from cpu_cast import data  # noqa: E402
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


def load_frozen():
    if not PROTOCOL.exists():
        raise SystemExit("protocol not frozen — run freeze_stage4_protocol.py first")
    p = json.loads(PROTOCOL.read_text())
    run_path = REPO / p["candidate_run"]
    h = hashlib.sha256(run_path.read_bytes()).hexdigest()
    if h != p["candidate_run_sha256"]:
        raise SystemExit("candidate run hash mismatch — data changed since freeze")
    return p


def parity_check(ranker, load_reqs, thresholds, checkpoints):
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


def serve_once(ranker, tokenizer, runs, queries, corpus, qids, max_length,
               stagger_ms, which):
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
    queue_ms = np.array([((r.first_exec_time or t0) - (r.submit_time or t0))
                         * 1000 for r in reqs])
    trace = [{"qid": r.qid,
              "arrival_ms": round(r.arrival * 1000, 1),
              "submit_ms": round(((r.submit_time or t0) - t0) * 1000, 1),
              "first_exec_ms": round(((r.first_exec_time or t0) - t0) * 1000, 1),
              "finish_ms": round(((r.finish_time or t0) - t0) * 1000, 1),
              "latency_ms": round(float(lats[i]), 1)}
             for i, r in enumerate(reqs)]
    return {
        "p50_ms": round(float(np.percentile(lats, 50)), 1),
        "p90_ms": round(float(np.percentile(lats, 90)), 1),
        "p95_ms": round(float(np.percentile(lats, 95)), 1),
        "mean_ms": round(float(lats.mean()), 1),
        "mean_queue_ms": round(float(queue_ms.mean()), 1),
        "max_queue_ms": round(float(queue_ms.max()), 1),
        "throughput_qps": round(len(reqs) / total_s, 3),
        "makespan_s": round(total_s, 3),
        "requests": trace,
    }


def main():
    global THRESHOLDS, CHECKPOINTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--num-load", type=int, default=20,
                        help="pilot uses the first N frozen load qids")
    parser.add_argument("--parity-limit", type=int, default=20,
                        help="policy parity checked on the first N load reqs")
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

    CHECKPOINTS = tuple(p["checkpoints"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ranker = ActiveSetRanker(MODEL)
    log = JsonlLogger(args.out, run_meta={
        "stage": "stage4", "version": "pilot", **git_state(),
        "protocol_version": p["protocol_version"],
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "dataset": p["dataset"], "model": MODEL,
        "num_load": len(p["load_qids"]), "reps": args.reps,
        "stagger_ms": args.stagger_ms, "threads": p["threads"],
        "cli": " ".join(sys.argv[1:]),
    })

    calib_reqs = build_requests(runs, d["queries"], d["corpus"],
                                p["calib_qids"], tokenizer, p["max_length"])
    t0 = time.perf_counter()
    THRESHOLDS = calibrate_thresholds(ranker, calib_reqs, CHECKPOINTS,
                                      p["keep_frac"])
    log.write({"event": "calibration", "thresholds": THRESHOLDS,
               "calibration_s": round(time.perf_counter() - t0, 1)})
    print(f"thresholds: { {k: round(v, 4) for k, v in THRESHOLDS.items()} }",
          flush=True)

    load_qids = p["load_qids"][: args.num_load]
    load_reqs = build_requests(runs, d["queries"], d["corpus"], load_qids,
                               tokenizer, p["max_length"])
    max_s, exit_mm = parity_check(ranker, load_reqs[: args.parity_limit],
                                  THRESHOLDS, CHECKPOINTS)
    log.write({"event": "parity", "max_score_diff": round(max_s, 8),
               "exit_mismatches": exit_mm,
               "rows": sum(r.n for r in load_reqs)})
    print(f"parity: max_score_diff={max_s:.2e} exit_mismatches={exit_mm}",
          flush=True)
    assert exit_mm == 0 and max_s < 1e-4, "policy parity FAILED — do not rerun"

    for rep in range(args.reps):
        for which in ("serial", "wavefront"):
            stats = serve_once(ranker, tokenizer, runs, d["queries"],
                               d["corpus"], load_qids, p["max_length"],
                               args.stagger_ms, which)
            log.write({"event": "real_online", "scheduler": which,
                       "rep": rep, "stagger_ms": args.stagger_ms,
                       "n_requests": len(load_qids),
                       "p99_reported": False,
                       "p99_reason": "pilot below frozen completion threshold",
                       **stats})
            print(f"rep{rep} {which}: p50={stats['p50_ms']}ms "
                  f"p95={stats['p95_ms']}ms q_mean={stats['mean_queue_ms']}ms "
                  f"qps={stats['throughput_qps']}", flush=True)
            trim_memory()

    print("PILOT DONE", flush=True)


if __name__ == "__main__":
    main()
