# 01_sentiment — SentNoB Sentiment Classification

## Dataset

**SentNoB** — a 3-class (Positive, Negative, Neutral) Bangla sentiment
classification dataset from the BLP 2023 Shared Task.

- **Source**: https://github.com/KhondokerIslam/SentNoB
- **License**: Check the original repository for license terms.
- **Classes**: Positive (0), Negative (1), Neutral (2)
- **Loading**: The data loader tries HF dataset hub first, then falls back to
  cloning the GitHub repository.

## Task

Sentence-level sentiment classification. A linear classification head is
trained on top of frozen base model representations.

### Pooling Strategy

| Model Type | Pooling |
|---|---|
| Causal LM (gamba, gsg) | Last non-pad token hidden state |
| Masked LM (banglabert) | [CLS] token (first token) |

## Metrics

- **Primary**: Macro-F1 (3-class)
- **Secondary**: Accuracy

## Sanity Check

BanglaBERT's fine-tuned macro-F1 should be in the neighborhood of 72.89%
(the reference point from BanglaGamba's README). If it deviates by >5 points,
the script warns loudly — debug before trusting downstream comparisons.

## Usage

```bash
# Single run
python -m evaluation_suit.eval.01_sentiment.run --model gamba --seed 0

# Full sweep (3 models × 3 seeds)
for model in gamba gsg banglabert; do
    for seed in 0 1 2; do
        python -m evaluation_suit.eval.01_sentiment.run --model $model --seed $seed
    done
done
```

## Results

Written to `evaluation_suit/results/01_sentiment/seeds.jsonl` (one JSON line
per run, crash-safe append).
