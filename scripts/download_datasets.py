"""Download BEIR datasets from the official BeIR HuggingFace repos and
materialize the classic BEIR folder layout per dataset:

    data/<name>/corpus.jsonl
    data/<name>/queries.jsonl
    data/<name>/qrels/{train,dev,test}.tsv

Datasets: scifact, nfcorpus, arguana, fiqa (CPU-CAST Stage-1 set).
Writes a manifest with counts, sizes, and sha256 of each raw file.
"""
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MANIFESTS = REPO / "manifests"
HF = "https://huggingface.co/datasets"

DATASETS = ["scifact", "nfcorpus", "arguana", "fiqa"]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.relative_to(REPO)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main() -> None:
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(), "source": "official BeIR HuggingFace repos", "datasets": {}}
    for name in DATASETS:
        print(f"[{name}]")
        d = DATA / name
        raw = d / "raw"
        fetch(f"{HF}/BeIR/{name}/resolve/main/corpus/corpus-00000-of-00001.parquet", raw / "corpus.parquet")
        fetch(f"{HF}/BeIR/{name}/resolve/main/queries/queries-00000-of-00001.parquet", raw / "queries.parquet")
        qrels_files = {}
        for split in ("train", "dev", "test"):
            dest = raw / f"qrels-{split}.tsv"
            url = f"{HF}/BeIR/{name}-qrels/resolve/main/{split}.tsv"
            try:
                fetch(url, dest)
                qrels_files[split] = dest
            except Exception as e:
                print(f"  no {split} qrels ({e})")

        corpus = pd.read_parquet(raw / "corpus.parquet")
        queries = pd.read_parquet(raw / "queries.parquet")
        corpus.to_json(d / "corpus.jsonl", orient="records", lines=True, force_ascii=False)
        queries.to_json(d / "queries.jsonl", orient="records", lines=True, force_ascii=False)
        (d / "qrels").mkdir(exist_ok=True)
        qrels_rows = {}
        for split, path in qrels_files.items():
            out = d / "qrels" / f"{split}.tsv"
            out.write_text(path.read_text())
            qrels_rows[split] = max(0, len(out.read_text().strip().splitlines()) - 1)

        entry = {
            "corpus_parquet_sha256": sha256(raw / "corpus.parquet"),
            "queries_parquet_sha256": sha256(raw / "queries.parquet"),
            "corpus_parquet_mib": round((raw / "corpus.parquet").stat().st_size / 2**20, 2),
            "num_corpus_docs": int(len(corpus)),
            "num_queries": int(len(queries)),
            "qrels_rows": qrels_rows,
        }
        manifest["datasets"][name] = entry
        print(f"  docs={entry['num_corpus_docs']} queries={entry['num_queries']} qrels={qrels_rows}")

    out_path = MANIFESTS / "datasets_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
