"""BEIR-format data loading (corpus.jsonl / queries.jsonl / qrels/*.tsv)."""
import csv
import json
from pathlib import Path


def load_corpus(path):
    corpus = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            corpus[row["_id"]] = {
                "title": row.get("title", "") or "",
                "text": row.get("text", "") or "",
            }
    return corpus


def load_queries(path):
    queries = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries[row["_id"]] = row.get("text", "") or ""
    return queries


def load_qrels(path):
    qrels = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            qid, did, rel = row[0], row[1], int(row[2])
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def load_split(dataset_dir, split="test"):
    d = Path(dataset_dir)
    return {
        "corpus": load_corpus(d / "corpus.jsonl"),
        "queries": load_queries(d / "queries.jsonl"),
        "qrels": load_qrels(d / "qrels" / f"{split}.tsv"),
    }


def restrict_to_test_queries(data):
    """Keep only queries that appear in qrels (the official test split)."""
    qids = set(data["qrels"].keys())
    data = dict(data)
    data["queries"] = {q: t for q, t in data["queries"].items() if q in qids}
    return data
