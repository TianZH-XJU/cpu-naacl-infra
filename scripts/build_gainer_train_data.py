"""GAINER training data: trajectory extraction on nfcorpus train queries.

Follows the GAINER (baseline repo, GAINER_v3.ipynb cell 12) recipe, adapted:
- leave-one-dataset-out: train pools from nfcorpus train split, evaluation on
  scifact (GAINER trains on all other BEIR datasets; we only have nfcorpus
  locally, so the training set is smaller -- documented limitation)
- BM25 top-50 pools (GAINER TOP_K=50)
- CLS hidden state at checkpoints after layers 2 and 4 of MiniLM-L6
  (1/3 and 2/3 of depth, matching GAINER's checkpoint proportions)
- max_length 256 (our evaluation setting, not GAINER's 384)

Output: data/gainer_train/nfcorpus_train.npz + pools.jsonl
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

CHECKPOINT_LAYERS = [1, 3]  # 0-indexed; CLS after layers 2 and 4


def bm25_pool(corpus, queries, budget):
    from rank_bm25 import BM25Okapi

    doc_ids = list(corpus.keys())
    texts = [(corpus[d]["title"] + " " + corpus[d]["text"]).lower().split() for d in doc_ids]
    bm25 = BM25Okapi(texts)
    run = {}
    for qid, qtext in queries.items():
        scores = bm25.get_scores(qtext.lower().split())
        top = np.argsort(-scores)[:budget]
        run[qid] = [doc_ids[i] for i in top if scores[i] > 0]
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-queries", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--model", default="models/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/gainer_train")
    args = parser.parse_args()

    torch.set_num_threads(2)
    rng = np.random.RandomState(args.seed)
    corpus = data.load_corpus(f"data/{args.dataset}/corpus.jsonl")
    queries_all = data.load_queries(f"data/{args.dataset}/queries.jsonl")
    qrels = data.load_qrels(f"data/{args.dataset}/qrels/{args.split}.tsv")
    qids = sorted(set(qrels.keys()) & set(queries_all.keys()))
    if len(qids) > args.num_queries:
        qids = list(rng.choice(qids, args.num_queries, replace=False))
    queries = {q: queries_all[q] for q in qids}
    print(f"{len(queries)} {args.split} queries, {len(corpus)} docs", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / f"{args.dataset}_{args.split}_pools.jsonl"
    if pool_path.exists():
        pools = {}
        with pool_path.open() as f:
            for line in f:
                row = json.loads(line)
                pools[row["qid"]] = row["doc_ids"]
    else:
        pools = bm25_pool(corpus, queries, args.top_k)
        with pool_path.open("w") as f:
            for qid in sorted(pools):
                f.write(json.dumps({"qid": qid, "doc_ids": pools[qid]}) + "\n")
    print(f"pools: {sum(len(v) for v in pools.values())} pairs", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    hidden_size = model.config.hidden_size

    pairs, meta = [], []
    for qid in sorted(pools):
        for did in pools[qid]:
            pairs.append((queries[qid], corpus[did]["title"] + " " + corpus[did]["text"]))
            meta.append((qid, did))
    n = len(pairs)
    cls = {c: np.zeros((n, hidden_size), dtype=np.float16) for c in CHECKPOINT_LAYERS}
    final = np.zeros(n, dtype=np.float32)

    # sort by length for batching efficiency, keep mapping
    order = np.argsort([len(q) + len(d) for q, d in pairs], kind="stable")
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            enc = tokenizer(
                [pairs[i] for i in idx], padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt")
            out = model(**enc, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states
            final[idx] = out.logits[:, 0].float().numpy()
            for c in CHECKPOINT_LAYERS:
                cls[c][idx] = hs[c + 1][:, 0, :].float().numpy().astype(np.float16)
            if (start // args.batch_size) % 50 == 0:
                el = time.perf_counter() - t0
                print(f"  {start}/{n} pairs, {el:.0f}s elapsed", flush=True)
    print(f"extraction done in {time.perf_counter() - t0:.0f}s", flush=True)

    np.savez_compressed(
        out_dir / f"{args.dataset}_{args.split}.npz",
        cls1=cls[CHECKPOINT_LAYERS[0]],
        cls2=cls[CHECKPOINT_LAYERS[1]],
        final=final,
        qid=np.array([m[0] for m in meta]),
        docid=np.array([m[1] for m in meta]),
    )
    print("saved", out_dir / f"{args.dataset}_{args.split}.npz", flush=True)


if __name__ == "__main__":
    main()
