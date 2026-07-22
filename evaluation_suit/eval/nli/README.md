# 03_nli — Natural Language Inference

## Datasets

### XNLI-bn
- **Source**: `load_dataset("xnli", "bn")` from HF
- **Labels**: entailment (0), neutral (1), contradiction (2)
- **⚠ Known caveat**: XNLI-bn is machine-translated and known to be noisy
  for Bangla. This must be noted explicitly in the paper. The translations
  contain artifacts that may affect model performance differently depending
  on whether the model was pretrained on translated text.

### BanglaParaphrase
- **Source**: https://github.com/csebuetnlp/banglaparaphrase (not on HF)
- **Labels**: paraphrase (1), not-paraphrase (0)
- **Purpose**: Clean native-Bangla sentence-pair reasoning counterpart to
  XNLI's translation artifacts. Reported as a separate row.

## Task

Sentence-pair classification. Same pooling/head logic as 01_sentiment,
but with two input sequences.

### Sentence-Pair Encoding

| Model Type | Encoding Strategy |
|---|---|
| Masked LM (BanglaBERT) | Standard `text_pair` via tokenizer → `[CLS] premise [SEP] hypothesis [SEP]` |
| Causal LM (gamba, gsg) | Explicit separator: `premise । hypothesis` (Bangla danda) |

The custom GSG/Gamba tokenizers have no defined pair-encoding convention,
so we use plain string concatenation with a Bangla danda separator. This
choice is documented here and should be noted in the paper.

## Metric

**Accuracy** — standard for NLI.
Macro-F1 is also computed as a secondary metric.

## Usage

```bash
# Single run
python -m evaluation_suit.eval.03_nli.run --model gamba --dataset xnli --seed 0

# Full sweep (3 models × 3 seeds × 2 datasets)
for model in gamba gsg banglabert; do
    for dataset in xnli paraphrase; do
        for seed in 0 1 2; do
            python -m evaluation_suit.eval.03_nli.run \
                --model $model --dataset $dataset --seed $seed
        done
    done
done
```

## Results

Written to `evaluation_suit/results/03_nli/seeds.jsonl`.
XNLI and BanglaParaphrase results are separate rows.
