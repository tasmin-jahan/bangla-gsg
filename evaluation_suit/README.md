# Bangla LM Evaluation Suite

This directory contains a complete, self-contained evaluation suite for
Bangla language models. It handles dataset loading, model inference, scoring,
and aggregation for 6 distinct tasks.

## Tasks

| ID | Task | Dataset | Type | Metric |
|---|---|---|---|---|
| `01_sentiment` | Sentiment Analysis | SentNoB | Sentence Classification | Macro-F1 |
| `02_ner` | Named Entity Recognition | ANCHOLIK, WikiAnn | Token Classification | Entity-F1 |
| `03_nli` | Natural Language Inference | XNLI, BanglaParaphrase | Sentence-Pair | Accuracy |
| `04_mt` | Machine Translation | FLORES-200 | Generation (bn↔en) | BLEU, chrF |
| `05_long_context`| Needle-in-a-Haystack | Custom (Wikipedia) | Retrieval | Accuracy |
| `06_summarization`| Abstractive Summary | XL-Sum Bengali | Generation | ROUGE-L |

**Note on BanglaBERT**: Tasks 04, 05, and 06 require generative models.
BanglaBERT is excluded from these tasks (and is marked as such, not just
left as a blank cell).

## Setup

```bash
# Install evaluation dependencies
pip install -r evaluation_suit/requirements.txt
```

## Running the Suite

The `run_all.py` orchestrator handles running the entire suite. It is
resumable — if a run fails or is interrupted, it will skip already completed
(model, seed) pairs on the next run.

```bash
# Run the entire suite (all tasks, all models, 3 seeds)
python -m evaluation_suit.eval.run_all

# Run a specific task
python -m evaluation_suit.eval.run_all --tasks 01 02

# Run specific models
python -m evaluation_suit.eval.run_all --models gamba gsg

# Dry run to see what will execute
python -m evaluation_suit.eval.run_all --dry-run
```

## Results Aggregation

Once tasks are completed, aggregate the results into the final summary tables:

```bash
python -m evaluation_suit.scripts.aggregate_results
```

This generates:
- `evaluation_suit/results/aggregated_results.json`
- `evaluation_suit/results/summary_tables.md` (the main paper results table)

## Contamination Check (Task 04)

Task 04 (MT) is gated by a contamination check because FLORES-200 is often
derived alongside NLLB data (which is in our pretraining corpus).

The orchestrator runs `04_mt/check_contamination.py` automatically. If the
overlap rate exceeds the threshold (2%), the MT evaluation will be skipped
to prevent reporting compromised numbers.
