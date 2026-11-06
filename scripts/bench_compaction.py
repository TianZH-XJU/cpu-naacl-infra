"""Gate A microbenchmark: active-set compaction vs keep-batch (plan section 9.2).

Runs MiniLM-L6 layer-by-layer on real tokenized (query, doc) pairs and measures,
for each candidate batch size / sequence length / thread count / active ratio:

  C_keep    = layer forward on the full padded batch (inactive rows included)
  C_compact = index_select (gather) + layer forward on active rows only + scatter back
  C_gather / C_scatter are measured separately.

The break-even active ratio f* is where C_compact < C_keep starts to hold.
This is the evidence base for CPU-CAST Gate A: does active-set compaction ever
pay off on a low-core CPU, and under which shapes?

Results: artifacts/runs/bench_compaction.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_cast import data  # noqa: E402
from cpu_cast.logging_utils import JsonlLogger  # noqa: E402


class LayerWiseRanker:
    def __init__(self, path):
        full = AutoModelForSequenceClassification.from_pretrained(path)
        self.bert = full.bert
        self.classifier = full.classifier
        self.dropout = full.dropout
        self.num_layers = len(self.bert.encoder.layer)
        self.embeddings = self.bert.embeddings
        self.pooler = self.bert.pooler

    def embed(self, input_ids, attention_mask, token_type_ids):
        ext_mask = attention_mask[:, None, None, :].to(dtype=torch.float32)
        ext_mask = (1.0 - ext_mask) * torch.finfo(torch.float32).min
        return self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids), ext_mask

    def forward_layer(self, layer, hidden, ext_mask):
        out = layer(hidden, attention_mask=ext_mask)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def score_from_hidden(self, hidden, attention_mask):
        seq_out = hidden
        pooled = self.pooler(seq_out)
        logits = self.classifier(self.dropout(pooled))
        return logits.squeeze(-1)


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize if torch.cuda.is_available() else None
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--run", default="data/candidate_runs/scifact/rrf_top100.jsonl")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64, 100])
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.1, 0.05])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--layer", type=int, default=2, help="which layer to benchmark (0-indexed)")
    parser.add_argument("--out", default="artifacts/runs/bench_compaction.jsonl")
    args = parser.parse_args()

    d = data.load_split(f"data/{args.dataset}")
    queries = list(d["queries"].values())
    docs = [d["corpus"][k]["title"] + " " + d["corpus"][k]["text"] for k in list(d["corpus"])[:200]]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ranker = LayerWiseRanker(args.model)
    layer = ranker.bert.encoder.layer[args.layer]
    log = JsonlLogger(args.out, run_meta={
        "stage": "gate_a", "model": Path(args.model).name, "layer": args.layer,
    })

    for n_threads in args.threads:
        torch.set_num_threads(n_threads)
        for seq in args.seq_lengths:
            enc = tokenizer(
                [(queries[i % len(queries)], docs[i % len(docs)]) for i in range(max(args.batch_sizes))],
                padding="max_length", truncation=True, max_length=seq, return_tensors="pt",
            )
            input_ids, attention_mask, token_type_ids = (
                enc["input_ids"], enc["attention_mask"], enc["token_type_ids"],
            )
            with torch.inference_mode():
                hidden0, ext_mask = ranker.embed(input_ids, attention_mask, token_type_ids)
                for n in args.batch_sizes:
                    h_full = hidden0[:n]
                    m_full = ext_mask[:n]
                    for f in args.fractions:
                        k = max(1, int(round(n * f)))
                        idx = torch.randperm(n)[:k]

                        def keep():
                            ranker.forward_layer(layer, h_full, m_full)

                        h_active = h_full.index_select(0, idx)
                        m_active = m_full.index_select(0, idx)

                        def compact():
                            h_new = ranker.forward_layer(layer, h_active, m_active)
                            h_full.index_copy_(0, idx, h_new)

                        def gather():
                            return h_full.index_select(0, idx)

                        t_keep = time_fn(keep, args.warmup, args.iters)
                        t_compact = time_fn(compact, args.warmup, args.iters)
                        t_gather = time_fn(gather, args.warmup, args.iters)
                        record = {
                            "event": "compaction",
                            "threads": n_threads,
                            "seq": seq,
                            "batch": n,
                            "active": k,
                            "fraction": f,
                            "keep_ms": round(float(np.mean(t_keep)) * 1000, 3),
                            "keep_p95_ms": round(float(np.percentile(t_keep, 95)) * 1000, 3),
                            "compact_ms": round(float(np.mean(t_compact)) * 1000, 3),
                            "gather_ms": round(float(np.mean(t_gather)) * 1000, 3),
                            "compact_wins": bool(np.mean(t_compact) < np.mean(t_keep)),
                            "gain_pct": round((1 - np.mean(t_compact) / np.mean(t_keep)) * 100, 1),
                        }
                        log.write(record)
                        print(json.dumps({k2: record[k2] for k2 in
                              ("threads", "seq", "batch", "active", "fraction", "keep_ms", "compact_ms", "gain_pct")}))
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
