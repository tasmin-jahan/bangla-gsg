# 04_mt — Machine Translation (FLORES-200 bn↔en)

## ⚠ Contamination Warning

The pretraining corpus (`ahmed-farhanur-rashid/bn-foundational-pretrain-corpus`)
includes NLLB and BanglaNMT parallel data. FLORES-200 is commonly derived
alongside NLLB releases, creating a real contamination risk.

**`check_contamination.py` MUST be run before `generate.py`.**

The contamination check:
1. SHA-256 hash-compares FLORES devtest sentences against the NLLB config
2. Reports overlap count, rate, and samples
3. Gates the generation script — it refuses to run without a clean report

### Known Limitation
This is an exact-match check only. Near-duplicate and paraphrase
contamination is NOT detected. This limitation must be noted in the paper.

## Dataset

**FLORES-200** (`facebook/flores`, `ben_Beng-eng_Latn` config, devtest split)

## Task

Prompt-based translation in both directions:
- **bn→en**: Bangla source → English target
- **en→bn**: English source → Bangla target

Greedy decoding (`do_sample=False`).

## Models

- **gamba** ✓
- **gsg** ✓
- **banglabert** ✗ (not generative — explicitly excluded, not a blank cell)

## Metrics

- **BLEU** (sacrebleu, 0-100 scale)
- **chrF** (sacrebleu)

## Usage

```bash
# Step 1: Run contamination check (REQUIRED)
python -m evaluation_suit.eval.04_mt.check_contamination

# Step 2: Run translation (only if contamination check passes)
python -m evaluation_suit.eval.04_mt.generate --model gamba
python -m evaluation_suit.eval.04_mt.generate --model gsg
```

## Results

- `evaluation_suit/results/04_mt/contamination_report.json` — contamination check
- `evaluation_suit/results/04_mt/scores_<model>_<direction>.json` — per-direction scores
- `evaluation_suit/results/04_mt/seeds.jsonl` — combined results
