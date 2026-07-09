"""
Delete all __pycache__ directories in the project.
Ignores venvs, dotfiles, and any folder starting with '.'.

Usage:
  python utils/clean_pycache.py
  python utils/clean_pycache.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Remove __pycache__ dirs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without deleting.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    removed = 0
    skipped = 0

    for pycache in root.rglob("__pycache__"):
        # Skip hidden dirs (dotfiles)
        if any(part.startswith(".") for part in pycache.parts):
            skipped += 1
            continue
        # Skip venvs
        if "venv" in pycache.parts or ".venv" in pycache.parts:
            skipped += 1
            continue

        if args.dry_run:
            print(f"  would delete: {pycache}")
        else:
            shutil.rmtree(pycache)
            print(f"  deleted: {pycache}")
        removed += 1

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{action} {removed} __pycache__ dirs ({skipped} skipped)")


if __name__ == "__main__":
    main()
