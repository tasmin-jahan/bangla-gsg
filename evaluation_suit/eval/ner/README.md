# 02_ner — Named Entity Recognition

## Datasets

### ANCHOLIK-NER
- **Source**: https://github.com/AridHasan/ancholik-ner
- **Not on HF** — cloned from GitHub and converted to HF datasets format.
- **License**: Check the original repository for license terms and usage
  restrictions.
- **Label schema**: Auto-detected (BIO or BILOU) from the data files.

### WikiAnn-bn
- **Source**: `load_dataset("wikiann", "bn")` from HF
- **Labels**: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC (BIO schema)
- **Purpose**: Comparable to published BanglaBERT NER numbers.
  Kept separate from ANCHOLIK in results — two rows, not merged.

## Task

Token-level NER classification. A per-token linear classifier is trained on
top of frozen base model hidden states.

**Key difference from sentence classification (01_sentiment):**
- No pooling — every token position gets classified independently.
- Hidden states shape `[B, T, H]` → logits shape `[B, T, num_labels]`.

### Label Alignment (Critical)

NER labels are word-level, but tokenizers are subword-level. Standard
alignment:
- First subword of each word → gets the word's NER tag
- Continuation subwords → get `-100` (ignored in loss and evaluation)
- Special tokens (BOS, EOS, PAD) → get `-100`

This is the most common source of silent NER bugs. The implementation
includes explicit alignment logic and should be verified on a few examples.

## Metric

**Entity-level F1** via `seqeval` — this is the standard NER metric and
what's comparable to published numbers. Not token-level accuracy.

## Usage

```bash
# Single run
python -m evaluation_suit.eval.02_ner.run --model gamba --dataset ancholik --seed 0
python -m evaluation_suit.eval.02_ner.run --model gamba --dataset wikiann --seed 0

# Full sweep (3 models × 3 seeds × 2 datasets)
for model in gamba gsg banglabert; do
    for dataset in ancholik wikiann; do
        for seed in 0 1 2; do
            python -m evaluation_suit.eval.02_ner.run \
                --model $model --dataset $dataset --seed $seed
        done
    done
done
```

## Results

Written to `evaluation_suit/results/02_ner/seeds.jsonl`.
ANCHOLIK and WikiAnn results are separate rows — do NOT average across datasets.
