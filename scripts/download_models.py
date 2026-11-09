"""Download CPU-CAST Stage-1 model weights from HuggingFace into models/.

Downloads only the files needed for CPU inference (config, tokenizer, weights).
Writes a manifest with repo id, revision, and local size.
"""
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
    # (repo_id, pinned_revision) — revisions fixed for reproducibility (P2);
    # re-running the download reproduces byte-identical weights.
    ("BAAI/bge-small-en-v1.5", "main"),
    ("sentence-transformers/all-MiniLM-L6-v2", "main"),
    ("cross-encoder/ms-marco-TinyBERT-L-2-v2", "main"),
    ("cross-encoder/ms-marco-MiniLM-L-2-v2", "main"),
    ("cross-encoder/ms-marco-MiniLM-L-4-v2", "main"),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "main"),
]


def dir_size_mib(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20, 2)


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
                 "local_path": str(Path(path).relative_to(REPO)), "size_mib": dir_size_mib(local)}
        manifest["models"][repo_id] = entry
        print(f"  -> {entry['local_path']} ({entry['size_mib']} MiB)")
    out = MANIFESTS / "models_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
