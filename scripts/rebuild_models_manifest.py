"""Rebuild manifests/models_manifest.json from local weights WITHOUT downloading.

Second-review P2 fix: the committed manifest (Nov 5) predates revision pinning
and carries no file hashes. This script records, for each model in
MODELS_TO_FETCH, the pinned HF commit SHA plus the SHA-256 of every local
weight/config file. Run with the weights already present; never touches the
network.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from download_models import MODELS_TO_FETCH  # noqa: E402

MODELS = REPO / "models"
MANIFESTS = REPO / "manifests"


SKIP_PARTS = {".gitignore", "CACHEDIR.TAG"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_files(local: Path) -> dict:
    """SHA-256 of the real weight/config artifacts, skipping HF cache clutter
    (lock/metadata files, snapshot pointers)."""
    files = {}
    for f in sorted(local.rglob("*")):
        if not f.is_file() or f.name in SKIP_PARTS or f.name.endswith((".lock", ".metadata")):
            continue
        files[f.name] = sha256_file(f)
    return files


def main() -> None:
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "note": "revisions are immutable HF commit SHAs; file hashes are SHA-256",
                "models": {}}
    missing = []
    for repo_id, revision in MODELS_TO_FETCH:
        local = MODELS / repo_id.split("/")[-1]
        if not local.is_dir():
            missing.append(repo_id)
            continue
        files = artifact_files(local)
        manifest["models"][repo_id] = {
            "repo_id": repo_id, "revision": revision,
            "local_path": str(local.relative_to(REPO)),
            "size_mib": round(sum(f.stat().st_size for f in local.rglob("*")
                                  if f.is_file()) / 2**20, 2),
            "files": files,
        }
        print(f"[{repo_id}@{revision}] {len(files)} files hashed")
    if missing:
        raise SystemExit(f"missing local weights (run download_models.py first): {missing}")
    out = MANIFESTS / "models_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
