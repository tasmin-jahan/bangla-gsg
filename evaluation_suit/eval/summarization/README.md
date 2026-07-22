# 06_summarization — XL-Sum Bengali

## Dataset

**XL-Sum** (`csebuetnlp/xlsum`, `"bengali"` config, test split)

XL-Sum is a large-scale multilingual summarization dataset containing
professionally annotated article-summary pairs from BBC Bengali.

## Task

Prompt-based abstractive summarization. The source article is truncated
to `max_src_len` tokens (default 1024) and the model generates a summary
using greedy decoding.

## Models

- **gamba** ✓
- **gsg** ✓
- **banglabert** ✗ (not generative — same exclusion as tasks 04 and 05)

## Metrics

- **ROUGE-L** (required) — primary summarization metric
- **BERTScore** (attempted, with `try/except` fallback)
  - If BERTScore fails to load, falls back to ROUGE-L only with a
    logged warning. A metric library failure should not kill the whole run.

## Usage

```bash
python -m evaluation_suit.eval.06_summarization.generate --model gamba
python -m evaluation_suit.eval.06_summarization.generate --model gsg

# Quick test with limited examples
python -m evaluation_suit.eval.06_summarization.generate --model gamba --max_examples 10
```

## Results

- `evaluation_suit/results/06_summarization/seeds.jsonl` — combined scores
- `evaluation_suit/results/06_summarization/scores_<model>.json` — detailed + samples
