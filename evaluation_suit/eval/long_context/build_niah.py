"""
Bangla Needle-in-a-Haystack (NIAH) — Haystack Construction.

Builds haystacks from Bangla Wikipedia articles with a synthetic
"needle" (a factoid sentence) inserted at controlled depths.

The haystack is trimmed to fit within the model's context window
(hard cap at 2048 tokens).

Usage:
    python -m evaluation_suit.eval.05_long_context.build_niah
"""

import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from evaluation_suit.eval.common.io_utils import write_json, write_jsonl
from evaluation_suit.eval.common.seeding import set_seed


# ── Needles (Bangla factoid sentences) ───────────────────────────────────────

# These are simple factoid sentences that can be verified by exact string match
NEEDLES = [
    ("পৃথিবীর সবচেয়ে উঁচু পর্বত এভারেস্ট।", "এভারেস্ট"),
    ("বাংলাদেশের রাজধানী ঢাকা।", "ঢাকা"),
    ("সূর্য পূর্ব দিকে ওঠে।", "পূর্ব"),
    ("পানির রাসায়নিক সংকেত H2O।", "H2O"),
    ("চাঁদ পৃথিবীর একমাত্র প্রাকৃতিক উপগ্রহ।", "চাঁদ"),
    ("বাংলা ভাষার বর্ণমালায় ৫০টি বর্ণ আছে।", "৫০"),
    ("রবীন্দ্রনাথ ঠাকুর নোবেল পুরস্কার পেয়েছিলেন।", "রবীন্দ্রনাথ"),
    ("পদ্মা বাংলাদেশের প্রধান নদী।", "পদ্মা"),
    ("বাংলাদেশের জাতীয় ফুল শাপলা।", "শাপলা"),
    ("মুক্তিযুদ্ধ ১৯৭১ সালে হয়েছিল।", "১৯৭১"),
]


def load_wikipedia_haystack(
    max_articles: int = 200,
    min_article_len: int = 500,
) -> List[str]:
    """
    Load Bangla Wikipedia articles for haystack construction.

    Returns a list of article texts (cleaned paragraphs).
    """
    print("[NIAH] Loading Bangla Wikipedia...")
    try:
        wiki = load_dataset(
            "wikimedia/wikipedia",
            "20231101.bn",
            split="train",
            streaming=True,
        )
    except Exception:
        # Fallback to older config
        wiki = load_dataset(
            "wikipedia",
            "20220301.bn",
            split="train",
            streaming=True,
        )

    articles = []
    for article in wiki:
        text = article.get("text", "")
        # Clean: remove very short articles, headers, etc.
        text = text.strip()
        if len(text) >= min_article_len:
            # Remove Wikipedia markup artifacts
            lines = [
                line.strip() for line in text.split("\n")
                if line.strip()
                and not line.strip().startswith("==")
                and not line.strip().startswith("{{")
                and not line.strip().startswith("|")
                and len(line.strip()) > 20
            ]
            clean_text = " ".join(lines)
            if len(clean_text) >= min_article_len:
                articles.append(clean_text)

        if len(articles) >= max_articles:
            break

    print(f"[NIAH] Loaded {len(articles)} articles")
    return articles


def build_haystack(
    articles: List[str],
    tokenizer,
    target_tokens: int,
    needle_text: str,
    needle_depth: float,
) -> Tuple[str, str]:
    """
    Build a single haystack with a needle inserted at a specific depth.

    Args:
        articles: Pool of Wikipedia article texts.
        tokenizer: Tokenizer for counting tokens.
        target_tokens: Target haystack length in tokens.
        needle_text: The needle sentence to insert.
        needle_depth: Where to insert (0.0=start, 1.0=end).

    Returns:
        (haystack_text, retrieval_prompt)
    """
    # Build haystack by concatenating random article chunks
    random.shuffle(articles)
    haystack_parts = []
    total_tokens = 0

    for article in articles:
        article_tokens = len(tokenizer.encode(article, add_special_tokens=False))
        if total_tokens + article_tokens > target_tokens * 2:
            # Take a portion
            words = article.split()
            portion = words[:max(len(words) // 2, 50)]
            haystack_parts.append(" ".join(portion))
            total_tokens += len(tokenizer.encode(" ".join(portion), add_special_tokens=False))
        else:
            haystack_parts.append(article)
            total_tokens += article_tokens

        if total_tokens >= target_tokens:
            break

    # Join and tokenize to exact length
    full_text = " ".join(haystack_parts)
    tokens = tokenizer.encode(full_text, add_special_tokens=False)

    # Reserve space for needle + prompt
    needle_tokens = len(tokenizer.encode(needle_text, add_special_tokens=False))
    prompt_template = "উপরের লেখা থেকে নিম্নলিখিত প্রশ্নের উত্তর দিন: লুকানো তথ্যটি কী ছিল?"
    prompt_tokens = len(tokenizer.encode(prompt_template, add_special_tokens=False))

    # Available tokens for haystack content
    available = target_tokens - needle_tokens - prompt_tokens - 10  # margin
    available = max(available, 50)

    haystack_tokens = tokens[:available]

    # Insert needle at depth
    insert_pos = int(len(haystack_tokens) * needle_depth)
    insert_pos = max(0, min(insert_pos, len(haystack_tokens)))

    # Find a good insertion point (between words)
    final_tokens = (
        haystack_tokens[:insert_pos]
        + tokenizer.encode(f" {needle_text} ", add_special_tokens=False)
        + haystack_tokens[insert_pos:]
    )

    haystack_text = tokenizer.decode(final_tokens, skip_special_tokens=True)

    # Build full prompt
    full_prompt = f"{haystack_text}\n\n{prompt_template}"

    return full_prompt, needle_text


def build_niah_dataset(
    tokenizer,
    context_lengths: List[int] = None,
    needle_depths: List[float] = None,
    samples_per_cell: int = 20,
    output_dir: str = "evaluation_suit/results/05_long_context/niah_data",
) -> str:
    """
    Build the full NIAH evaluation dataset.

    Args:
        tokenizer: Tokenizer for token counting.
        context_lengths: List of target context lengths in tokens.
        needle_depths: List of needle insertion depths (0.0-1.0).
        samples_per_cell: Number of samples per (length, depth) cell.
        output_dir: Where to save the dataset.

    Returns:
        Path to the saved JSONL file.
    """
    if context_lengths is None:
        context_lengths = [256, 512, 1024, 1536, 2048]
    if needle_depths is None:
        needle_depths = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Hard cap
    context_lengths = [min(cl, 2048) for cl in context_lengths]

    set_seed(42)
    articles = load_wikipedia_haystack()

    all_samples = []
    sample_id = 0

    for ctx_len in context_lengths:
        for depth in needle_depths:
            for i in range(samples_per_cell):
                needle_text, needle_answer = random.choice(NEEDLES)

                prompt, needle = build_haystack(
                    articles, tokenizer, ctx_len, needle_text, depth,
                )

                sample = {
                    "id": sample_id,
                    "context_length": ctx_len,
                    "needle_depth": depth,
                    "sample_idx": i,
                    "prompt": prompt,
                    "needle": needle,
                    "expected_answer": needle_answer,
                    "actual_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                }
                all_samples.append(sample)
                sample_id += 1

    output_path = f"{output_dir}/niah_samples.jsonl"
    write_jsonl(output_path, all_samples)
    print(f"[NIAH] Built {len(all_samples)} samples → {output_path}")

    # Summary
    summary = {
        "total_samples": len(all_samples),
        "context_lengths": context_lengths,
        "needle_depths": needle_depths,
        "samples_per_cell": samples_per_cell,
        "grid_size": f"{len(context_lengths)}×{len(needle_depths)}",
    }
    write_json(f"{output_dir}/build_summary.json", summary)

    return output_path


if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Use gamba tokenizer for building (both models share the same vocab)
    print("[NIAH] Loading tokenizer for dataset construction...")
    tokenizer = AutoTokenizer.from_pretrained(
        "ahmed-farhanur-rashid/bangla-gamba",
        trust_remote_code=True,
    )
    build_niah_dataset(tokenizer)
