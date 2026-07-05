"""
BanglaGSG Tokenizer Evaluation
=================================

Compares BanglaGSG against reference tokenizers on fertility, compression,
UNK rate, and speed. Also runs quick sanity checks (--sanity).

Usage:
  python tests/test_tokenizer.py
  python tests/test_tokenizer.py --sanity --skip-references
  python tests/test_tokenizer.py --sample-size 5000
  python tests/test_tokenizer.py --categories bangla_formal english
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BANGLA_GSG_TOKENIZER = PROJECT_ROOT / "saved" / "tokenizer"
TOKENIZER_CORPUS = PROJECT_ROOT / "saved" / "data" / "tokenizer_corpus" / "corpus.jsonl"
REPORT_DIR     = PROJECT_ROOT / "saved" / "reports"

REFERENCE_MODELS = {
    "mBART-50":   "facebook/mbart-large-50-many-to-many-mmt",
    "NLLB-200":   "facebook/nllb-200-distilled-600M",
    "BanglaBERT": "sagorsarker/bangla-bert-base",
    "GPT-2":      "gpt2",
}

TEST_SENTENCES = {
    "bangla_formal": [
        "আমি বাংলাদেশের মানুষ। আমি বাংলায় কথা বলি।",
        "প্রধানমন্ত্রী আজ জাতীয় সংসদে ভাষণ দিয়েছেন।",
        "বাংলাদেশ একটি সুন্দর দেশ। এখানে নদী, পাহাড় এবং সমুদ্র রয়েছে।",
        "ছাত্ররা পরীক্ষার জন্য প্রস্তুতি নিচ্ছে। তারা প্রতিদিন অধ্যয়ন করে।",
        "বাংলা ভাষা আমাদের জাতীয় ভাষা। এটি বিশ্বের সবচেয়ে সুন্দর ভাষাগুলোর একটি।",
    ],
    "bangla_news": [
        "রাজধানীতে আজ সকালে একটি বিশাল অগ্নিকাণ্ড ঘটেছে।",
        "বিশ্বকাপ ফুটবলে বাংলাদেশ প্রথমবার যোগ্যতা অর্জন করেছে।",
        "চট্টগ্রাম বন্দরে নতুন টার্মিনাল উদ্বোধন করা হয়েছে।",
        "দেশে করোনা ভাইরাসের নতুন প্রকৃতি চিহ্নিত হয়েছে।",
        "সরকার নতুন শিক্ষানীতি ঘোষণা করেছে।",
    ],
    "english": [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is a subfield of artificial intelligence.",
        "The transformer architecture revolutionized deep learning for text.",
        "Bangladesh is a country in South Asia with a rich cultural heritage.",
        "The model was trained on a corpus of one billion words.",
    ],
    "banglish": [
        "ami tomake bhalobashi, tumi kemon acho?",
        "ajke khub bhalo din, amra park e ber korlam",
        "tumi ki kaj koro? ami ekta software engineer",
        "amar nam farhan. ami bangladeshi",
        "bhai eta ki hoise? khub bhalo lagtese",
    ],
    "code_mixed": [
        "এই product টা really ভালো, must buy করো।",
        "আমি একটি Python script লিখেছি যেটা data process করবে।",
        "তুমি কি GitHub এ code দেখেছো? সেখানে অনেক ভালো project আছে।",
        "আমাদের team এ ৫ জন developer আছে। আমরা React এ কাজ করি।",
        "বইটা খুব ভালো লেখা। তুমি কি এটা pad এ পড়েছো?",
    ],
    "python_code": [
        "def hello(): print('Hello, World!')",
        "class Model(nn.Module): def __init__(self): super().__init__()",
        "import torch; x = torch.randn(3, 3)",
        "for i in range(10): print(i ** 2)",
        "with open('data.json') as f: data = json.load(f)",
    ],
    "bangla_edge_cases": [
        "রাষ্ট্রবিজ্ঞান, স্বাতন্ত্র্য, ঐচ্ছিক, দ্ব্যর্থহীন, জলোচ্ছ্বাস",
        "😊 👍 🇧🇩 ✨ 🚀",
        "২০২৪ সালে ১,৫০০ টাকা দাম ছিল।",
        "Check out https://bengali.ai and email test@example.com!",
        "র‍্যাব (RAB) এর অভিযান।",
    ],
}

# Sanity: quick per-sentence decode + special token checks (moved from wrapper)
SANITY_SENTENCES = [
    ("Bangla",     "আমি বাংলাদেশের মানুষ।"),
    ("English",    "Hello, how are you?"),
    ("Banglish",   "ami tomake bhalobashi"),
    ("Code-mixed", "এই product টা really ভালো।"),
]

SANITY_SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<s>", "</s>", "<|im_start|>",
    "<|lang_bn|>", "<|reserved_0|>", "<|reserved_99|>",
]


# ── Tokenizer loading ──────────────────────────────────────────────────────

def load_tokenizers(skip_references: bool = False) -> dict:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizers = {}

    print("Loading BanglaGSG tokenizer...")
    if BANGLA_GSG_TOKENIZER.exists():
        tokenizers["BanglaGSG"] = PreTrainedTokenizerFast.from_pretrained(str(BANGLA_GSG_TOKENIZER))
        print(f"  [OK] BanglaGSG (vocab={tokenizers['BanglaGSG'].vocab_size})")
    else:
        print(f"  [SKIP] Not found at {BANGLA_GSG_TOKENIZER}")
        print("  Run: python -m src.tokenizer.wrapper --spm-model saved/tokenizer/model/banglagsg_tokenizer.model")

    if skip_references:
        return tokenizers

    for name, model_id in REFERENCE_MODELS.items():
        try:
            print(f"Loading {name} ({model_id})...")
            tokenizers[name] = AutoTokenizer.from_pretrained(model_id)
            print(f"  [OK] {name} (vocab={tokenizers[name].vocab_size})")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return tokenizers


# ── Corpus sampling ────────────────────────────────────────────────────────

def sample_corpus(path: Path, n: int = 10_000, seed: int = 42) -> list[str]:
    """Reservoir-sample n docs from a JSONL file."""
    if not path.exists():
        print(f"  [SKIP] Corpus not found: {path}")
        return []

    random.seed(seed)
    samples, total = [], 0

    with open(path) as f:
        for line in f:
            total += 1
            try:
                text = json.loads(line).get("text", "").strip()
            except json.JSONDecodeError:
                continue
            if not text:
                continue
            if len(samples) < n:
                samples.append(text)
            else:
                j = random.randint(0, total - 1)
                if j < n:
                    samples[j] = text

    print(f"  Sampled {len(samples):,} docs from {path.name} ({total:,} total)")
    return samples


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(tokenizer, texts: list[str]) -> dict:
    total_tokens = total_words = total_chars = total_unk = 0
    round_trip_failures = 0
    hyper_fragmented_words = 0
    total_words_measured = 0
    unk_id = tokenizer.unk_token_id or 0

    t0 = time.time()
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(ids)
        total_unk    += sum(1 for t in ids if t == unk_id)
        
        # Round trip accuracy (allow normal spacing differences)
        decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
        
        # Strip any special tokens present in the original text before comparing,
        # since skip_special_tokens=True removes them from `decoded` but not from `text`.
        text_for_comparison = text
        for special in getattr(tokenizer, "all_special_tokens", []):
            text_for_comparison = text_for_comparison.replace(special, "")

        if "".join(decoded.split()) != "".join(text_for_comparison.split()):
            round_trip_failures += 1
            
        # Fragmentation analysis
        words = text.split()
        total_words += max(len(words), 1)
        for w in words:
            w_ids = tokenizer.encode(w, add_special_tokens=False)
            if len(w_ids) > 3:  # Hyper fragmented if a single word takes >3 tokens
                hyper_fragmented_words += 1
            total_words_measured += 1
            
        total_chars  += len(text)
    elapsed = time.time() - t0

    n = max(len(texts), 1)
    return {
        "docs":               n,
        "total_tokens":       total_tokens,
        "avg_tokens_per_doc": round(total_tokens / n, 1),
        "fertility":          round(total_tokens / max(total_words, 1), 3),
        "compression":        round(total_chars  / max(total_tokens, 1), 2),
        "unk_rate_pct":       round(total_unk    / max(total_tokens, 1) * 100, 4),
        "round_trip_fail_pct":round(round_trip_failures / n * 100, 2),
        "hyper_frag_pct":     round(hyper_fragmented_words / max(total_words_measured, 1) * 100, 2),
        "docs_per_sec":       round(n / max(elapsed, 1e-6), 1),
    }


# ── Category / corpus runners ──────────────────────────────────────────────

def run_category_tests(tokenizers: dict, categories: list[str] | None = None) -> dict:
    cats = {k: TEST_SENTENCES[k] for k in (categories or TEST_SENTENCES) if k in TEST_SENTENCES}
    return {
        cat: {name: compute_metrics(tok, sents) for name, tok in tokenizers.items()}
        for cat, sents in cats.items()
    }


def run_corpus_tests(tokenizers: dict, sample_size: int = 10_000) -> dict:
    print(f"\nSampling {sample_size:,} docs from tokenizer corpus...")
    corpora = {
        "tokenizer_corpus": sample_corpus(TOKENIZER_CORPUS, n=sample_size),
    }
    return {
        label: {name: compute_metrics(tok, samples) for name, tok in tokenizers.items()}
        for label, samples in corpora.items()
        if samples
    }


# ── Sanity tests (absorbed from wrapper._test_tokenizer) ──────────────────

def run_sanity_tests(tokenizer) -> None:
    print(f"\n{'=' * 60}\n  Sanity Tests\n{'=' * 60}\n")

    for label, text in SANITY_SENTENCES:
        ids = tokenizer.encode(text, add_special_tokens=False)
        words = len(text.split())
        print(f"  [{label}]")
        print(f"    Input:   {text}")
        print(f"    Tokens:  {len(ids)} tok / {words} words = {len(ids)/max(words,1):.2f} fertility")
        print(f"    IDs:     {ids[:10]}{'...' if len(ids) > 10 else ''}")
        print(f"    Decoded: {tokenizer.decode(ids, skip_special_tokens=True)}\n")

    print("  Special token spot-check:")
    for tok in SANITY_SPECIAL_TOKENS:
        tid  = tokenizer.convert_tokens_to_ids(tok)
        fail = tid == tokenizer.unk_token_id and tok != tokenizer.unk_token
        print(f"    {'[FAIL]' if fail else '[OK]'} {tok:30s} -> ID {tid}")

    print()
    try:
        chat_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "তুমি একটি সহায়ক বাংলা ভাষার মডেল।"},
                {"role": "user",   "content": "আমাকে সাহায্য করো।"},
            ],
            tokenize=False, add_generation_prompt=True,
        )
        assert "<|im_start|>" in chat_text
        print("  Chat template output:")
        for line in chat_text.splitlines():
            print(f"    {line}")
        print("\n  [OK] Chat template works correctly.")
    except Exception as e:
        print(f"  [FAIL] Chat template: {e}")


# ── Output ─────────────────────────────────────────────────────────────────

# ponytail: one helper drives all four metric sections instead of copy-pasted loops
def _metric_rows(results: dict, tok_names: list[str], metric: str, fmt: str) -> None:
    for cat, cat_res in results.items():
        row = f"  {cat:<18}"
        for name in tok_names:
            val = cat_res.get(name, {}).get(metric, "—")
            row += f" {val:>12}" if isinstance(val, str) else f" {val:>{12}{fmt}}"
        print(row)


def print_comparison_table(results: dict, title: str) -> None:
    tok_names = sorted({n for cr in results.values() for n in cr})

    print(f"\n{'=' * 90}\n  {title}\n{'=' * 90}\n")
    header = f"{'Category':<20}" + "".join(f" {n:>12}" for n in tok_names)
    print(header)
    print("-" * len(header))

    for label, metric, fmt in [
        ("Fertility (tokens/word, lower = better)",    "fertility",    ".3f"),
        ("Compression (chars/token, higher = better)", "compression",  ".2f"),
        ("UNK rate % (lower = better)",                "unk_rate_pct", ".4f"),
        ("Round-Trip Failure % (lower = better)",      "round_trip_fail_pct", ".2f"),
        ("Hyper-Fragmentation % (words > 3 tokens)",   "hyper_frag_pct", ".2f"),
        ("Speed (docs/sec)",                           "docs_per_sec", ".1f"),
    ]:
        print(f"\n  {label}")
        _metric_rows(results, tok_names, metric, fmt)


def save_report(category_results: dict, corpus_results: dict, all_tokenizers: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "tokenizer_evaluation.md"

    def md_table(results: dict, metric: str, fmt: str, tok_names: list[str]) -> list[str]:
        header = "| Category |" + "".join(f" {n} |" for n in tok_names)
        sep    = "|---|" + "---|" * len(tok_names)
        rows   = []
        for cat, cr in results.items():
            row = f"| {cat} |"
            for name in tok_names:
                val = cr.get(name, {}).get(metric, "—")
                row += f" {val:{fmt}} |" if isinstance(val, (int, float)) else f" {val} |"
            rows.append(row)
        return [header, sep] + rows + [""]

    lines = [
        "# BanglaGSG Tokenizer Evaluation\n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "",
        "## Tokenizer Summary\n",
        "| Tokenizer | Vocab Size | Type |",
        "|---|---|---|",
        *[f"| {n} | {tok.vocab_size:,} | {'Unigram (SP)' if n == 'BanglaGSG' else 'Auto'} |"
          for n, tok in all_tokenizers.items()],
        "",
    ]

    for section_label, results in [
        ("## Curated Sentence Tests", category_results),
        ("## Corpus Tests (sampled from cleaned data)", corpus_results),
    ]:
        if not results:
            continue
        tok_names = sorted({n for cr in results.values() for n in cr})
        lines.append(f"{section_label}\n")
        for title, metric, fmt in [
            ("Fertility (tokens/word, lower = better)",    "fertility",    ".3f"),
            ("Compression (chars/token, higher = better)", "compression",  ".2f"),
            ("UNK Rate % (lower = better)",                "unk_rate_pct", ".4f"),
            ("Round-Trip Failure % (lower = better)",      "round_trip_fail_pct", ".2f"),
            ("Hyper-Fragmentation % (words > 3 tokens)",   "hyper_frag_pct", ".2f"),
        ]:
            lines += [f"### {title}\n"] + md_table(results, metric, fmt, tok_names)

    # Detailed metrics
    lines.append("## Detailed Metrics\n")
    for results in [category_results, corpus_results]:
        for cat, cr in results.items():
            tok_names = sorted(cr)
            lines += [
                f"### {cat}\n",
                "| Metric |" + "".join(f" {n} |" for n in tok_names),
                "|---|" + "---|" * len(tok_names),
                *[
                    "| {} |".format(m) + "".join(f" {cr[n].get(m, '—')} |" for n in tok_names)
                    for m in ["fertility", "compression", "unk_rate_pct", "round_trip_fail_pct", "hyper_frag_pct", "avg_tokens_per_doc", "docs_per_sec"]
                ],
                "",
            ]

    report_path.write_text("\n".join(lines))
    print(f"\n[OK] Report saved to: {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate BanglaGSG tokenizer against references.")
    parser.add_argument("--sample-size",     type=int, default=10_000)
    parser.add_argument("--skip-references", action="store_true")
    parser.add_argument("--categories",      nargs="*")
    parser.add_argument("--no-corpus",       action="store_true")
    parser.add_argument("--sanity",          action="store_true",
                        help="Run quick decode/special-token/chat-template checks only.")
    args = parser.parse_args()

    print("=" * 60)
    print("  BanglaGSG Tokenizer Evaluation")
    print("=" * 60)

    tokenizers = load_tokenizers(skip_references=args.skip_references)
    if not tokenizers:
        sys.exit("ERROR: No tokenizers loaded. Run wrapper first.")

    if args.sanity:
        tok = tokenizers.get("BanglaGSG")
        if not tok:
            sys.exit("ERROR: BanglaGSG tokenizer not loaded.")
        run_sanity_tests(tok)
        return

    category_results = run_category_tests(tokenizers, args.categories)
    corpus_results   = {} if args.no_corpus else run_corpus_tests(tokenizers, args.sample_size)

    print_comparison_table(category_results, "Curated Sentence Tests")
    if corpus_results:
        print_comparison_table(corpus_results, "Corpus Tests")

    save_report(category_results, corpus_results, tokenizers)
    print("\n[OK] Evaluation complete!")


if __name__ == "__main__":
    main()
