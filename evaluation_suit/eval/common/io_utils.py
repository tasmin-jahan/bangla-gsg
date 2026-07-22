"""
I/O utilities for the eval suite.

Provides crash-safe JSONL read/write/append operations.
Every task run appends one JSON line per (model, task, seed) so
aggregation is trivial and partial runs aren't lost.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    """Write a list of records to a JSONL file (overwrites)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file. Returns empty list if file missing."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_result(path: str, record: Dict[str, Any]) -> None:
    """
    Append a single result record to a JSONL file.

    Crash-safe: uses append mode so partial runs aren't lost.
    Automatically adds a timestamp if not present.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if "timestamp" not in record:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_completed_runs(path: str) -> List[tuple]:
    """
    Get list of completed (model, seed) pairs from an existing results file.

    Used by run_all.py for resumability — skip re-running completed seeds.
    """
    records = read_jsonl(path)
    return [(r.get("model"), r.get("seed")) for r in records if "model" in r and "seed" in r]


def write_json(path: str, data: Any) -> None:
    """Write a JSON file (for summary outputs)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Optional[Any]:
    """Read a JSON file. Returns None if file missing."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
