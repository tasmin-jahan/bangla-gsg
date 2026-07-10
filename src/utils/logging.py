"""
Metric logging utilities for BanglaGSG.

Simple CSV-based logging for training metrics. Lightweight, no external
dependencies (no wandb/tensorboard required, though can be added later).

Resume-safe: appends to existing CSVs and continues elapsed_s from
the checkpoint-backed wall clock (with CSV fallback for backward
compatibility with checkpoints that don't have wall_clock yet).
"""

import csv
import time
from pathlib import Path
from typing import Dict, Optional


class MetricLogger:
    """
    Logs training metrics to CSV.

    Creates a CSV file with columns for step, tokens_seen, loss, lr,
    gradient norms, etc. Safe to resume: appends to existing CSVs and
    continues elapsed_s from the checkpoint-backed wall clock.
    """

    def __init__(self, log_dir: str, run_name: str = "default"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"
        self._csv_initialized = False
        self._fieldnames: list = []

        # Wall-clock tracking: the Trainer sets _resumed_wall_clock from
        # the checkpoint. If the checkpoint is old (no wall_clock key),
        # the Trainer falls back to _read_last_elapsed() from the CSV.
        self._resumed_wall_clock = 0.0
        self._session_start = time.time()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_last_elapsed(self, path: Path) -> float:
        """Return the last elapsed_s value in an existing CSV, or 0.

        Used as a fallback for backward compatibility with checkpoints
        that don't have wall_clock saved yet. Once the first new-format
        checkpoint is saved, this fallback is never needed again.
        """
        if not path.exists() or path.stat().st_size == 0:
            return 0.0
        try:
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows and "elapsed_s" in rows[-1]:
                return float(rows[-1]["elapsed_s"])
        except Exception:
            pass
        return 0.0

    def _elapsed(self) -> float:
        """Wall-clock seconds since first run began (survives restarts).

        Uses checkpoint-backed _resumed_wall_clock + current session time.
        """
        return round(self._resumed_wall_clock + (time.time() - self._session_start), 1)

    def _init_csv(self, fieldnames: list):
        """
        Prepare CSV for appending.

        If the file already exists and has a header, read fieldnames from
        it so we stay schema-compatible. If it is absent or empty, write
        the header now.
        """
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        if file_exists:
            # Read existing header so we don't duplicate or misorder columns.
            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                existing = reader.fieldnames or []
            # Merge: keep existing order, append any new columns at the end.
            merged = list(existing)
            for col in fieldnames:
                if col not in merged:
                    merged.append(col)
            self._fieldnames = merged
        else:
            self._fieldnames = fieldnames
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
        self._csv_initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_eval(self, metrics: Dict[str, float]):
        """Log evaluation metrics to a separate CSV (safe to resume)."""
        eval_csv = self.log_dir / "eval_metrics.csv"
        metrics["elapsed_s"] = self._elapsed()

        file_exists = eval_csv.exists() and eval_csv.stat().st_size > 0
        with open(eval_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(metrics.keys()),
                extrasaction="ignore",
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def log(self, metrics: Dict[str, float]):
        """
        Append one row of metrics to the CSV.

        Args:
            metrics: Dict of metric name → value. None values are written
                     as empty cells (not 0) so downstream tools can
                     distinguish "not measured" from "zero".
        """
        metrics["elapsed_s"] = self._elapsed()

        if not self._csv_initialized:
            self._init_csv(list(metrics.keys()))

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
                restval="",          # missing keys → empty cell, not error
            )
            writer.writerow(metrics)
