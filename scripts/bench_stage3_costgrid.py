"""Stage-3 cost grid: shape-to-latency table for the measured CPU cost model.

Measures MiniLM-L6 execution cost as a function of
  N (rows = candidates/batch), L (padded sequence length),
  T (threads), precision (FP32 torch / INT8 ORT dynamic).

Per (T, L, N) shape, records:
  torch_fp32   full 6-layer forward (torch)
  ort_fp32     full forward via ORT fp32 session
  ort_int8     full forward via ORT dynamic-INT8 session
  compact      index_select + forward + index_copy on half the rows
  exit_head    trained exit-head MLP on N rows

N includes 200 to test the cross-query batching basis (RQ3):
t(200xL) vs 2 x t(100xL).

Results: artifacts/runs/cost_grid.jsonl

Run torch and ORT as separate processes (--backends torch, then --backends ort)
to keep only one inference framework resident; the script resumes from
already-recorded measurements.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast.active_set import ActiveSetRanker  # noqa: E402
from cpu_cast.logging_utils import JsonlLogger  # noqa: E402
from scripts.train_gainer_heads import MLP  # noqa: E402
from scripts.bench_reranker import OrtBackend  # noqa: E402


def make_inputs(tokenizer, N, L, vocab):
    ids = torch.randint(1000, vocab, (N, L))
    ids[:, 0] = tokenizer.cls_token_id
    ids[:, -1] = tokenizer.sep_token_id
    tt = torch.zeros(N, L, dtype=torch.long)
    tt[:, L // 2:] = 1
    am = torch.ones(N, L, dtype=torch.long)
    return ids, am, tt


def bench(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "p50_ms": round(float(np.percentile(times, 50)) * 1000, 3),
        "p95_ms": round(float(np.percentile(times, 95)) * 1000, 3),
        "reps": reps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--heads", default="models/gainer_heads")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 100, 200])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--min-reps", type=int, default=8)
    parser.add_argument("--backends", choices=["torch", "ort", "both"], default="both",
                        help="run torch and ORT in separate processes to stay "
                             "under the memory cgroup limit")
    parser.add_argument("--out", default="artifacts/runs/cost_grid.jsonl")
    args = parser.parse_args()

    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "cost_grid":
                    done.add((r["threads"], r["N"], r["L"], r["kind"]))
    print(f"resuming: {len(done)} measurements already recorded", flush=True)

    log = JsonlLogger(args.out, run_meta={
        "stage": "stage3", "event_type": "cost_grid", "model": args.model,
    })
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    run_torch = args.backends in ("torch", "both")
    run_ort = args.backends in ("ort", "both")

    ranker = None
    heads = []
    if run_torch:
        ranker = ActiveSetRanker(args.model)
        for i in range(2):
            h = MLP(384)
            h.load_state_dict(torch.load(Path(args.heads) / f"exit_head_{i}.pt", weights_only=True))
            heads.append(h.eval())

    # NOTE: ORT sessions are created per thread-count inside the T loop so the
    # recorded thread label matches the actual session configuration.
    ort = {}

    vocab = tokenizer.vocab_size
    for T in args.threads:
        torch.set_num_threads(T)
        if run_ort:
            for quantize in (False, True):
                try:
                    ort["int8" if quantize else "fp32"] = OrtBackend(
                        args.model, max_length=max(args.seq_lens), threads=T,
                        quantize=quantize)
                except Exception as e:
                    print(f"ORT quantize={quantize} unavailable: {e}", flush=True)
        for L in args.seq_lens:
            for N in args.batch_sizes:
                rec_base = {"event": "cost_grid", "threads": T, "N": N, "L": L,
                            "torch_num_threads": torch.get_num_threads()}
                todo_torch = run_torch and any(
                    (T, N, L, k) not in done
                    for k in ("embed", "layer", "layer_block", "gather",
                              "score", "compact_half", "exit_head"))
                todo_ort = run_ort and any(
                    (T, N, L, f"ort_{t}") not in done for t in ort)
                if not todo_torch and not todo_ort:
                    continue

                ids, am, tt = make_inputs(tokenizer, N, L, vocab)
                reps = args.reps
                if todo_torch:
                    with torch.inference_mode():
                        hidden, ext_mask = ranker.embed(ids, am, tt)
                        est = bench(lambda: ranker.forward_layer(hidden, ext_mask, 0), 3, 5)["p50_ms"]
                    if est * reps > 60:
                        reps = max(args.min_reps, int(60 / est))

                    with torch.inference_mode():
                        if (T, N, L, "embed") not in done:
                            r = bench(lambda: ranker.embed(ids, am, tt), args.warmup, reps)
                            log.write({**rec_base, "kind": "embed", **r})
                        # direct single-layer cost (the honest ledger entry)
                        if (T, N, L, "layer") not in done:
                            r = bench(lambda: ranker.forward_layer(hidden, ext_mask, 0),
                                      args.warmup, reps)
                            log.write({**rec_base, "kind": "layer", **r})
                        if (T, N, L, "layer_block") not in done:
                            r = bench(lambda: ranker.forward_full(ids, am, tt), args.warmup, reps)
                            log.write({**rec_base, "kind": "layer_block", **r})

                        idx = torch.arange(0, N, 2)
                        # gather/scatter only: index_select + index_copy, no forward
                        if (T, N, L, "gather") not in done:
                            def gather_only():
                                h = hidden.index_select(0, idx)
                                return hidden.index_copy(0, idx, h)
                            r = bench(gather_only, args.warmup, reps)
                            log.write({**rec_base, "kind": "gather", **r})
                        # pooler + classifier: the real checkpoint/final scoring path
                        if (T, N, L, "score") not in done:
                            r = bench(lambda: ranker.score_hidden(hidden), args.warmup, reps)
                            log.write({**rec_base, "kind": "score", **r})

                        def compact():
                            h = hidden.index_select(0, idx)
                            m = ext_mask.index_select(0, idx)
                            h = ranker.forward_layer(h, m, 0)
                            return hidden.index_copy(0, idx, h)
                        if (T, N, L, "compact_half") not in done:
                            r = bench(compact, args.warmup, reps)
                            log.write({**rec_base, "kind": "compact_half", **r})

                        cls = hidden[:, 0, :]
                        if (T, N, L, "exit_head") not in done:
                            r = bench(lambda: heads[0](cls), args.warmup, reps)
                            log.write({**rec_base, "kind": "exit_head", **r})
                    del hidden, ext_mask

                for tag, sess in ort.items():
                    if (T, N, L, f"ort_{tag}") in done:
                        continue
                    feeds = {
                        "input_ids": ids.numpy(),
                        "attention_mask": am.numpy(),
                        "token_type_ids": tt.numpy(),
                    }
                    r = bench(lambda: sess.session.run(["logits"], feeds), args.warmup, reps)
                    log.write({**rec_base, "kind": f"ort_{tag}", **r,
                               "session_threads": sess.session.get_session_options().intra_op_num_threads if hasattr(sess.session, 'get_session_options') else T})
                print(f"T={T} N={N} L={L} done (reps={reps})", flush=True)

    print("cost grid complete", flush=True)


if __name__ == "__main__":
    main()
