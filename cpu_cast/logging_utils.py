"""Unified JSONL logger for all CPU-CAST experiments (plan section Stage 0)."""
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path


class JsonlLogger:
    """Append-only JSONL records with shared run metadata."""

    def __init__(self, path, run_meta=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_meta = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }
        if run_meta:
            self.run_meta.update(run_meta)

    def write(self, record):
        row = dict(self.run_meta)
        row["record_utc"] = datetime.now(timezone.utc).isoformat()
        row.update(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_all(self, records):
        for r in records:
            self.write(r)
