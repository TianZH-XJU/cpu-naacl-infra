"""Stage-1 reranker baseline benchmark (plan section 7.2 / Stage 1).

Backends: PyTorch FP32, ONNX Runtime FP32, ONNX Runtime INT8 (dynamic).
Models: MiniLM-L6 (main), MiniLM-L4/L2, TinyBERT-L2 (shallow baselines).

For each (model, backend, threads):
  - quality: candidate-level nDCG@10 / MRR on the first N test queries,
    reranking the fixed RRF top-100 candidate run
  - latency: batch-1 (10 candidates, per-pair loop) and batch-K (100 candidates
    in one batch) per-query times, p50/p95
  - peak RSS
Results: artifacts/runs/bench_reranker.jsonl
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast import data, metrics  # noqa: E402
from cpu_cast.logging_utils import JsonlLogger  # noqa: E402


def peak_rss_kib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def percentile(xs, p):
    return float(np.percentile(xs, p))


class TorchBackend:
    name = "torch_fp32"

    def __init__(self, path, max_length):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.max_length = max_length

    def score(self, pairs, batch_size):
        scores = []
        with torch.inference_mode():
            for i in range(0, len(pairs), batch_size):
                enc = self.tokenizer(
                    pairs[i : i + batch_size], padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                )
                logits = self.model(**enc).logits
                scores.extend(logits.squeeze(-1).tolist())
        return scores


class OrtBackend:
    def __init__(self, path, max_length, threads, quantize):
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.max_length = max_length
        tag = "int8" if quantize else "fp32"
        onnx_path = Path(path) / f"model.{tag}.onnx"
        if not onnx_path.exists():
            base = Path(path) / "model.fp32.onnx"
            if not base.exists():
                self._export(path, base)
            if quantize:
                quantize_dynamic(str(base), str(onnx_path), weight_type=QuantType.QInt8)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.name = f"ort_{tag}"

    @staticmethod
    def _export(path, out):
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        dummy = dict(
            input_ids=torch.ones(1, 16, dtype=torch.int64),
            attention_mask=torch.ones(1, 16, dtype=torch.int64),
            token_type_ids=torch.ones(1, 16, dtype=torch.int64),
        )
        torch.onnx.export(
            model, (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            str(out), opset_version=17, dynamo=False,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
        )
        print(f"  exported {out}")

    def score(self, pairs, batch_size):
        scores = []
        for i in range(0, len(pairs), batch_size):
            enc = self.tokenizer(
                pairs[i : i + batch_size], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="np",
            )
            feeds = {k: v for k, v in {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
                "token_type_ids": enc.get("token_type_ids", np.zeros_like(enc["input_ids"])).astype(np.int64),
            }.items() if k in self.input_names}
            logits = self.session.run(["logits"], feeds)[0]
            scores.extend(logits.squeeze(-1).tolist())
        return scores


def rerank_queries(model, queries, corpus, runs, qrels, batch_size, log, meta, quality_limit):
    run_out = {}
    lat = []
    for qid, cands in runs.items():
        doc_ids = sorted(cands.keys())
        pairs = [(queries[qid], corpus[d]["title"] + " " + corpus[d]["text"]) for d in doc_ids]
        t0 = time.perf_counter()
        scores = model.score(pairs, batch_size)
        dt = time.perf_counter() - t0
        lat.append(dt)
        run_out[qid] = {d: float(s) for d, s in zip(doc_ids, scores)}
        if len(run_out) == quality_limit:
            break
    m = metrics.evaluate({q: qrels[q] for q in run_out}, run_out)
    record = dict(meta)
    record.update({
        "event": "bench_reranker",
        "candidate_ndcg_10": round(m["ndcg_cut_10"], 4),
        "candidate_mrr": round(m["recip_rank"], 4),
        "num_queries": len(run_out),
        "p50_ms_per_query": round(percentile(lat, 50) * 1000, 1),
        "p95_ms_per_query": round(percentile(lat, 95) * 1000, 1),
        "mean_ms_per_query": round(float(np.mean(lat)) * 1000, 1),
        "peak_rss_kib": peak_rss_kib(),
    })
    log.write(record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--run", default="data/candidate_runs/scifact/rrf_top100.jsonl")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--quality-queries", type=int, default=100)
    parser.add_argument("--latency-queries", type=int, default=20)
    parser.add_argument("--models", nargs="+", default=[
        "models/ms-marco-MiniLM-L-6-v2",
        "models/ms-marco-MiniLM-L-4-v2",
        "models/ms-marco-MiniLM-L-2-v2",
        "models/ms-marco-TinyBERT-L-2-v2",
    ])
    parser.add_argument("--backends", nargs="+", default=["torch", "ort_fp32", "ort_int8"])
    parser.add_argument("--out", default="artifacts/runs/bench_reranker.jsonl")
    args = parser.parse_args()

    d = data.load_split(f"data/{args.dataset}")
    d = data.restrict_to_test_queries(d)
    runs = {}
    with open(args.run, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["qid"] in d["queries"]:
                runs[row["qid"]] = row["candidates"]
    runs_latency = dict(list(runs.items())[: args.latency_queries])
    log = JsonlLogger(args.out, run_meta={"stage": "stage1", "dataset": args.dataset, "run": args.run})
    results = []

    for model_path in args.models:
        for threads in args.threads:
            torch.set_num_threads(threads)
            for backend in args.backends:
                if backend == "torch":
                    model = TorchBackend(model_path, max_length=256)
                elif backend.startswith("ort"):
                    model = OrtBackend(model_path, max_length=256, threads=threads, quantize=backend.endswith("int8"))
                meta = {"model": Path(model_path).name, "backend": model.name, "threads": threads}
                print(f"[{meta['model']} / {model.name} / {threads}T]", flush=True)
                q = rerank_queries(model, d["queries"], d["corpus"], runs, d["qrels"], 32, log, meta, args.quality_queries)
                print(f"  quality({args.quality_queries}q): nDCG@10={q['candidate_ndcg_10']}", flush=True)
                l1 = rerank_queries(model, d["queries"], d["corpus"], runs_latency, d["qrels"], 1, log,
                                    {**meta, "mode": "batch1_k10"}, 10**9)
                lk = rerank_queries(model, d["queries"], d["corpus"], runs_latency, d["qrels"], 10**9, log,
                                    {**meta, "mode": "batchK_k100"}, 10**9)
                print(f"  batch-1 (K=10): p50={l1['p50_ms_per_query']}ms p95={l1['p95_ms_per_query']}ms | "
                      f"batch-K (K=100): p50={lk['p50_ms_per_query']}ms p95={lk['p95_ms_per_query']}ms", flush=True)
                results.append((meta, q, l1, lk))
                del model

    print("\n=== Summary (quality on first 100 queries, latency on 20 queries) ===")
    print(f"{'model':24s} {'backend':10s} {'T':>2s} {'nDCG@10':>8s} {'b1_p50':>9s} {'b1_p95':>9s} {'bK_p50':>9s} {'bK_p95':>9s}")
    for meta, q, l1, lk in results:
        print(f"{meta['model']:24s} {meta['backend']:10s} {meta['threads']:2d} {q['candidate_ndcg_10']:8.4f} "
              f"{l1['p50_ms_per_query']:9.1f} {l1['p95_ms_per_query']:9.1f} "
              f"{lk['p50_ms_per_query']:9.1f} {lk['p95_ms_per_query']:9.1f}")


if __name__ == "__main__":
    main()
