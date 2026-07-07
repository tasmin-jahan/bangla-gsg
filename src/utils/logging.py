"""
Metric logging utilities for BanglaGSG.

Simple CSV-based logging for training metrics. Lightweight, no external
dependencies (no wandb/tensorboard required, though can be added later).
"""

import csv
import time
from pathlib import Path
from typing import Dict, Optional


class MetricLogger:
    """
    Logs training metrics to CSV and stdout.

    Creates a CSV file with columns for step, tokens_seen, loss, lr,
    gradient norms, etc.
    """

    def __init__(self, log_dir: str, run_name: str = "default"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"
        self.start_time = time.time()
        self._csv_initialized = False

    def _init_csv(self, fieldnames: list):
        """Initialize CSV file with headers."""
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        mode = "a" if file_exists else "w"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
        self._csv_initialized = True
        self._fieldnames = fieldnames

    def log_eval(self, metrics: Dict[str, float]):
        """Log evaluation metrics to a separate CSV."""
        eval_csv = self.log_dir / "eval_metrics.csv"
        metrics["elapsed_s"] = round(time.time() - self.start_time, 1)
        
        file_exists = eval_csv.exists()
        with open(eval_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def log(self, metrics: Dict[str, float]):
        """
        Log one row of metrics.

        Args:
            metrics: Dict of metric name → value.
        """
        # Add elapsed time
        metrics["elapsed_s"] = round(time.time() - self.start_time, 1)

        if not self._csv_initialized:
            self._init_csv(list(metrics.keys()))

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writerow(metrics)

    def log_stdout(
        self,
        step: int,
        total_steps: int,
        loss: float,
        lr_muon: float,
        lr_adamw: float,
        tokens_per_sec: Optional[float] = None,
        grad_norm_muon: Optional[float] = None,
        grad_norm_adamw: Optional[float] = None,
    ):
        """Print a formatted training progress line."""
        parts = [
            f"step {step:>6d}/{total_steps}",
            f"loss={loss:.4f}",
            f"lr_muon={lr_muon:.2e}",
            f"lr_adamw={lr_adamw:.2e}",
        ]
        if tokens_per_sec is not None:
            parts.append(f"tok/s={tokens_per_sec:.0f}")
        if grad_norm_muon is not None:
            parts.append(f"gnorm_muon={grad_norm_muon:.3f}")
        if grad_norm_adamw is not None:
            parts.append(f"gnorm_adamw={grad_norm_adamw:.3f}")

        print(f"[Train] {' | '.join(parts)}")
