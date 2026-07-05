#!/usr/bin/env python3
"""
Download only the parquet shards from
https://huggingface.co/datasets/ahmed-farhanur-rashid/bn-foundational-pretrain-corpus

- Skips per-folder LICENSE / ATTRIBUTION / README / .gitattributes files
- Preserves the folder structure (bangla_corpus/, fineweb_edu/, nllb_nmt/, sangraha/)
- Resumable: uses hf_hub_download's built-in caching, so re-running skips
  files that were already downloaded successfully.

Usage:
    pip install huggingface_hub --break-system-packages

    python download_bn__pretraincorpus.py
        --out ./saved/data/

Optional:
    --workers N       parallel download threads (default 4)
    --dry-run         list what would be downloaded without downloading
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "ahmed-farhanur-rashid/bn-foundational-pretrain-corpus"
REPO_TYPE = "dataset"

# File names/suffixes to exclude anywhere in the repo (top-level or per-folder)
EXCLUDE_NAMES = {
    "license",
    "attribution.md",
    "readme.md",
    ".gitattributes",
}


def is_excluded(path: str) -> bool:
    basename = os.path.basename(path).lower()
    if basename in EXCLUDE_NAMES:
        return True
    # Only keep parquet shards
    if not path.lower().endswith(".parquet"):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Download parquet shards only from bn-foundational-pretrain-corpus")
    parser.add_argument("--out", default="./bn-foundational-pretrain-corpus", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download threads")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    parser.add_argument("--revision", default="main", help="Repo revision/branch/commit")
    args = parser.parse_args()

    api = HfApi()

    print(f"Listing files in {REPO_ID} ...")
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE, revision=args.revision)

    parquet_files = sorted(f for f in all_files if not is_excluded(f))
    skipped_files = sorted(f for f in all_files if is_excluded(f))

    print(f"\nFound {len(all_files)} total files.")
    print(f"  -> {len(parquet_files)} parquet shards to download")
    print(f"  -> {len(skipped_files)} files skipped (license/readme/non-parquet):")
    for f in skipped_files:
        print(f"       skip: {f}")

    if not parquet_files:
        print("No parquet files found. Exiting.")
        sys.exit(0)

    if args.dry_run:
        print("\n--dry-run set, not downloading. Files that would be downloaded:")
        for f in parquet_files:
            print(f"  {f}")
        return

    os.makedirs(args.out, exist_ok=True)

    def download_one(rel_path: str):
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=rel_path,
                revision=args.revision,
                local_dir=args.out,
            )
            return rel_path, local_path, None
        except HfHubHTTPError as e:
            return rel_path, None, str(e)
        except Exception as e:  # noqa: BLE001
            return rel_path, None, str(e)

    print(f"\nDownloading {len(parquet_files)} parquet files into '{args.out}' with {args.workers} workers...\n")

    failed = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, f): f for f in parquet_files}
        for fut in as_completed(futures):
            rel_path, local_path, err = fut.result()
            completed += 1
            if err:
                failed.append((rel_path, err))
                print(f"[{completed}/{len(parquet_files)}] FAILED  {rel_path}  ({err})")
            else:
                print(f"[{completed}/{len(parquet_files)}] OK      {rel_path}")

    print("\nDone.")
    if failed:
        print(f"\n{len(failed)} file(s) failed to download:")
        for rel_path, err in failed:
            print(f"  {rel_path}: {err}")
        sys.exit(1)
    else:
        print(f"All {len(parquet_files)} parquet shards downloaded successfully into '{args.out}'.")


if __name__ == "__main__":
    main()
