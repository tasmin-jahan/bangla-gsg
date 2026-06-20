#!/usr/bin/env python3
"""
BanglaGSG — Parallel Prothom Alo Scraper
============================================
Scrapes Prothom Alo articles in parallel using multiple worker threads.
Output: saved/data/raw/prothomalo_raw.jsonl

Usage:
  python scripts/parallel_prothomalo.py --workers 8 --max-articles 100000
  python scripts/parallel_prothomalo.py --dry-run              # Count URLs only
"""

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
import trafilatura

SITEMAP_INDEX = "https://www.prothomalo.com/sitemap_index.xml"
OUTPUT = Path("saved/data/raw/prothomalo_raw.jsonl")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BanglaGSG-Research/1.0; "
                  "+https://github.com/ahmed-farhanur-rashid/BanglaGSG)"
}
CRAWL_DELAY = 1.5


def fetch_all_sitemap_urls() -> list[str]:
    """Fetch all article URLs from the Prothom Alo sitemap index."""
    urls = []
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    resp = requests.get(SITEMAP_INDEX, headers=HEADERS, timeout=30)
    root = ET.fromstring(resp.content)
    sitemap_locs = [l.text for l in root.findall("sm:sitemap/sm:loc", ns)]
    print(f"  Found {len(sitemap_locs)} sub-sitemaps")

    for i, loc in enumerate(sitemap_locs):
        time.sleep(0.3)
        try:
            r = requests.get(loc, headers=HEADERS, timeout=30)
            content = gzip.decompress(r.content) if loc.endswith(".gz") else r.content
            sub_root = ET.fromstring(content)
            page_urls = [l.text for l in sub_root.findall("sm:url/sm:loc", ns)]
            urls.extend(page_urls)
        except Exception as e:
            print(f"  Sitemap error [{i}/{len(sitemap_locs)}]: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Sitemaps: {i+1}/{len(sitemap_locs)} | URLs so far: {len(urls):,}")

    return urls


def scrape_one(url: str) -> dict | None:
    """Scrape a single Prothom Alo article."""
    try:
        downloaded = trafilatura.fetch_url(url, decode=True)
        if not downloaded:
            return None

        result = trafilatura.extract(
            downloaded, output_format="json", include_metadata=True,
            include_comments=False, favor_precision=True,
        )
        if not result:
            return None

        data = json.loads(result)
        text = data.get("text", "")
        if len(text.split()) < 30:
            return None

        return {
            "url": url,
            "domain": "prothomalo.com",
            "source_type": "formal_news",
            "language_region": "BD",
            "title": data.get("title", ""),
            "date": data.get("date", ""),
            "text": text,
            "word_count": len(text.split()),
        }
    except Exception:
        return None


def scrape_batch(urls: list[str], max_articles: int, workers: int):
    """Scrape articles using a thread pool."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    scraped_urls = set()
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for line in f:
                try:
                    scraped_urls.add(json.loads(line)["url"])
                except Exception:
                    pass

    pending = [u for u in urls if u not in scraped_urls]
    if max_articles:
        pending = pending[:max_articles]

    print(f"\n  Already scraped: {len(scraped_urls):,}")
    print(f"  Pending: {len(pending):,}")
    if not pending:
        return

    with ThreadPoolExecutor(max_workers=workers) as pool, \
         open(OUTPUT, "a", encoding="utf-8") as f:

        fut_map = {pool.submit(scrape_one, url): url for url in pending}
        done = 0
        t0 = time.time()

        for fut in as_completed(fut_map):
            result = fut.result()
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            done += 1

            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1)
                print(f"    Progress: {done:,}/{len(pending):,} "
                      f"({rate:.0f} docs/sec)")

    total = sum(1 for _ in open(OUTPUT))
    print(f"\n  Done. Total articles: {total:,}")


def main():
    parser = argparse.ArgumentParser(description="Parallel Prothom Alo scraper")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only count available URLs, don't scrape")
    args = parser.parse_args()

    print("Fetching sitemap URLs...")
    all_urls = fetch_all_sitemap_urls()
    print(f"  Total URLs in sitemap: {len(all_urls):,}")

    if args.dry_run:
        return

    scrape_batch(all_urls, args.max_articles, args.workers)


if __name__ == "__main__":
    main()
