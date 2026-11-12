"""Freeze the Stage-4 replacement-run protocol (D007 remediation).

Produces manifests/stage4_protocol.json: a seeded, hashed, idempotent
manifest fixing every input the paired Serial-vs-Wavefront rerun needs:

- dataset + candidate run file with sha256
- frozen load and calibration qid sets (seeded shuffle, NOT file order)
- checkpoints, keep_frac, max_length, threads
- arrival trace (stagger levels = offered-load points) and repetitions
- reporting rule: p99 only when a load point has enough completions

Re-running with the same seed reproduces the identical manifest; the file is
written once and never silently overwritten (--force to re-freeze).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast import data  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "manifests" / "stage4_protocol.json"

MIN_P99_COMPLETIONS = 10_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--run", default="data/candidate_runs/scifact/rrf_top100.jsonl")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--num-load", type=int, default=100)
    parser.add_argument("--num-calib", type=int, default=50)
    parser.add_argument("--stagger-levels-ms", type=float, nargs="+",
                        default=[500.0, 1000.0, 2000.0, 4000.0],
                        help="offered-load points: per-request arrival gap")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--keep-frac", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        print(f"protocol already frozen: {OUT} (use --force to re-freeze)")
        return

    run_path = REPO / args.run
    d = data.load_split(f"data/{args.dataset}")
    d = data.restrict_to_test_queries(d)
    with run_path.open(encoding="utf-8") as f:
        run_qids = sorted({json.loads(line)["qid"] for line in f
                           if json.loads(line)["qid"] in d["queries"]})

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(run_qids))
    ordered = [run_qids[i] for i in perm]
    if args.num_load + args.num_calib > len(ordered):
        raise SystemExit(
            f"requested {args.num_load}+{args.num_calib} queries but only "
            f"{len(ordered)} available in the candidate run")
    load_qids = sorted(ordered[: args.num_load])
    calib_qids = sorted(ordered[args.num_load: args.num_load + args.num_calib])

    protocol = {
        "protocol_version": "stage4-rerun-1",
        "seed": args.seed,
        "dataset": args.dataset,
        "candidate_run": args.run,
        "candidate_run_sha256": sha256(run_path),
        "qrels_sha256": sha256(REPO / f"data/{args.dataset}/qrels/test.tsv"),
        "load_qids": load_qids,
        "calib_qids": calib_qids,
        "checkpoints": sorted(args.checkpoints),
        "keep_frac": args.keep_frac,
        "max_length": args.max_length,
        "threads": args.threads,
        "reps": args.reps,
        "stagger_levels_ms": sorted(args.stagger_levels_ms),
        "arrival_trace_rule": "qid in load_qids order, arrival = index * stagger_ms",
        "reporting_rules": {
            "p99_min_completions": MIN_P99_COMPLETIONS,
            "p99_rule": ("report p99 only if a load point completes >= "
                         f"{MIN_P99_COMPLETIONS} requests; otherwise report "
                         "p50/p90 and label the sample size"),
            "paired_design": ("serial and wavefront see identical qids, "
                              "candidates, thresholds and arrival offsets"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(protocol, indent=2))
    print(f"frozen protocol -> {OUT}")
    print(f"  load qids: {len(load_qids)}, calib qids: {len(calib_qids)}, "
          f"seed: {args.seed}")
    print(f"  stagger levels: {protocol['stagger_levels_ms']} ms, "
          f"reps: {args.reps}")


if __name__ == "__main__":
    main()
