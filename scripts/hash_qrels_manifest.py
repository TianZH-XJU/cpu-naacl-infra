"""Add qrels file SHA-256 hashes to manifests/datasets_manifest.json.

Second-review P2 fix: the manifest had parquet hashes and qrels row counts
but no hashes for the qrels/*.tsv files that GAINER training actually reads
(e.g. data/nfcorpus/qrels/train.tsv). Computes hashes from local files.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "manifests" / "datasets_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    for name, entry in manifest["datasets"].items():
        qrels_dir = REPO / "data" / name / "qrels"
        if not qrels_dir.is_dir():
            continue
        entry["qrels_sha256"] = {
            f.name: sha256_file(f)
            for f in sorted(qrels_dir.glob("*.tsv"))
        }
        print(f"[{name}] {len(entry['qrels_sha256'])} qrels files hashed")
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["note"] = (manifest.get("note", "") +
                        " | qrels_sha256 added for training-data traceability").strip(" |")
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
