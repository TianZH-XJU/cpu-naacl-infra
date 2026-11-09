"""Download CPU-CAST Stage-1 model weights from HuggingFace into models/.

Downloads only the files needed for CPU inference (config, tokenizer, weights).
Writes a manifest with repo id, revision, and local size.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
MANIFESTS = REPO / "manifests"

ALLOW = [
    "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
    "special_tokens_map.json", "modules.json", "sentence_bert_config.json",
    "1_Pooling/config.json", "*.safetensors", "pytorch_model.bin",
]

MODELS_TO_FETCH = [
    # (repo_id, pinned_revision) — immutable HF commit SHAs (P2, second review);
    # re-running the download reproduces byte-identical weights.
    ("BAAI/bge-small-en-v1.5",
     "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
    ("sentence-transformers/all-MiniLM-L6-v2",
     "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    ("cross-encoder/ms-marco-TinyBERT-L-2-v2",
     "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc"),
    ("cross-encoder/ms-marco-MiniLM-L-2-v2",
     "1b5cd67b15209f24824c50370e0397743aa9b787"),
    ("cross-encoder/ms-marco-MiniLM-L-4-v2",
     "777b2f369bc1c2f850df8bd367ed1654bda4497b"),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2",
     "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
]


def dir_size_mib(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20, 2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(), "models": {}}
    for repo_id, revision in MODELS_TO_FETCH:
        print(f"[{repo_id}@{revision}]")
        local = MODELS / repo_id.split("/")[-1]
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local,
            allow_patterns=ALLOW,
        )
        entry = {"repo_id": repo_id, "revision": revision,
                 "local_path": str(Path(path).relative_to(REPO)), "size_mib": dir_size_mib(local),
                 "files": {f.name: sha256_file(f) for f in sorted(local.rglob("*")) if f.is_file()}}
        manifest["models"][repo_id] = entry
        print(f"  -> {entry['local_path']} ({entry['size_mib']} MiB)")
    out = MANIFESTS / "models_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
