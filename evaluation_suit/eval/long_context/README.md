# 05_long_context — Bangla Needle-in-a-Haystack (NIAH)

## Framing

This is NOT a "long-context capability" test — both models have a fixed
`seq_len=2048`. This task measures **retrieval accuracy under a fixed,
modest context budget**.

## Task

A "needle" (Bangla factoid sentence) is inserted into a "haystack"
(Bangla Wikipedia context) at a controlled depth. The model must generate
a response that contains the needle's key information.

### Grid

| Parameter | Values |
|---|---|
| Context lengths (tokens) | 256, 512, 1024, 1536, 2048 |
| Needle depths | 0.1, 0.3, 0.5, 0.7, 0.9 |
| Samples per cell | 20 |
| **Total per model** | **500** |

**Hard cap: 2048 tokens. Do not exceed.**

### Needles

Bangla factoid sentences (e.g., "বাংলাদেশের রাজধানী ঢাকা।") with
expected answers that can be verified by substring match.

### Haystack Source

Bangla Wikipedia (`wikimedia/wikipedia`, `20231101.bn` config).

## Models

- **gamba** ✓
- **gsg** ✓
- **banglabert** ✗ (not generative, not built for retrieval prompting)

## Output

- Raw JSONL: per-sample results with generated text and correctness
- Summary JSON: accuracy heatmap (context_length × depth → accuracy)
  suitable for direct plotting

## Usage

```bash
# Step 1: Build NIAH dataset (only needed once)
python -m evaluation_suit.eval.05_long_context.build_niah

# Step 2: Run evaluation
python -m evaluation_suit.eval.05_long_context.run --model gamba
python -m evaluation_suit.eval.05_long_context.run --model gsg
```

## Results

- `evaluation_suit/results/05_long_context/niah_data/` — constructed samples
- `evaluation_suit/results/05_long_context/raw_<model>.jsonl` — per-sample results
- `evaluation_suit/results/05_long_context/summary_<model>.json` — heatmap data
