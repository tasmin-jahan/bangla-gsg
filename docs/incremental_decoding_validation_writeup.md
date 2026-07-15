# Incremental Decoding for a Hybrid GDN–SWA–GQA Architecture: Implementation and Empirical Validation

## 1. Motivation

BanglaGSG is a 186M-parameter decoder-only language model trained from scratch on approximately 9.65B tokens (~75-80% Bangla-equivalent, including bidirectional Bangla-English translation pairs), using a hybrid layer architecture that interleaves three distinct mixer types in a 1:1:1 ratio across 12 layers: Gated DeltaNet (GDN), Sliding Window Attention (SWA), and Grouped Query Attention (GQA). GDN is a linear-time recurrent layer based on the delta rule (Yang et al., 2025), implemented via the `flash-linear-attention` library; SWA and GQA are standard attention variants implemented via FlashAttention 2.

The base implementation supported only full-sequence forward passes: generating N new tokens autoregressively required re-processing the entire growing sequence from scratch at each step, incurring O(N²) computational cost. This is prohibitive for any generation-based evaluation (e.g., machine translation) at scale. We therefore implemented incremental, cache-based decoding across all three layer types, with the constraint that the existing full-sequence (training and non-generative evaluation) code path remain numerically and behaviorally unchanged.

This document reports the implementation approach and, more centrally, the empirical validation methodology and results, since correctness of a caching mechanism for a novel hybrid recurrent-attention architecture cannot be assumed and must be demonstrated.

## 2. Implementation Overview

### 2.1 GDN caching

The `flash-linear-attention` library's `GatedDeltaNet` module (v0.5.1) natively supports incremental decoding via a `past_key_values: Cache | None` argument and a `use_cache: bool` flag, returning an updated `Cache` object alongside the layer output. The `Cache` class dynamically allocates per-layer state on first use and requires no explicit multi-layer initialization, allowing a single shared `Cache()` instance to be threaded across all GDN layers in the stack, with per-layer routing handled internally via each layer's `layer_idx`.

Integrating this required threading `past_key_values` and `use_cache` through the model's layer stack (`GDNBlock.forward()` → `BanglaGSGBlock.forward()` → `BanglaGSGModel.forward()`), with the no-cache path preserved as the default and left structurally unmodified.

### 2.2 SWA and GQA caching

SWA and GQA use the standard `flash_attn_func` primitive, which has no built-in caching mechanism analogous to GDN's. We evaluated FlashAttention's dedicated `flash_attn_with_kvcache` function as an alternative but rejected it in favor of manual key/value concatenation, for two reasons: (i) `flash_attn_with_kvcache` requires pre-allocated, fixed-size cache buffers indexed via `cache_seqlens`, which is unnecessary infrastructure for evaluation-scale generation; and (ii) it expects to apply rotary position embeddings internally via raw `rotary_cos`/`rotary_sin` tensors under its own layout convention (`rotary_interleaved=True` by default), which would have required bypassing the model's existing, independently-tested `RotaryEmbedding` module and introduced an unverified risk of positional misalignment. Manual concatenation — applying RoPE via the existing module prior to attention, then concatenating newly computed key/value tensors onto cached history — avoided this risk entirely while remaining functionally sufficient for single-sequence, evaluation-scale generation.

For SWA specifically, the key/value cache is truncated to the most recent `window_size` tokens after each step, consistent with the layer's bounded attention window; GQA's cache grows unboundedly across the full sequence, consistent with its role as the architecture's full-context layer type.

### 2.3 Scope

The implementation supports single-sequence (batch size 1) greedy or sampled generation. Batched incremental decoding was left unimplemented, as the underlying library's support for this case could not be confirmed from available documentation and was judged out of scope for evaluation-scale usage (generating outputs for a bounded evaluation set, not high-throughput serving).

## 3. Validation Methodology

Given that GDN is a stateful recurrent layer with no straightforward analytical guarantee of cache correctness under a novel wrapper, and that the architecture combines three structurally distinct layer types, we adopted a staged validation strategy: each layer type's caching was independently verified in isolation before combined end-to-end validation was attempted. This staging was deliberate, so that any discrepancy could be attributed to a specific component rather than requiring diagnosis across the full stack simultaneously.

For each stage, correctness was assessed by comparing the output of incremental (cached) generation against the pre-existing, unmodified full-sequence forward pass — the latter serving as ground truth, since it was already validated by successful model training and prior evaluation.

### 3.1 Isolated layer-type validation

Using single-layer-type toy configurations (e.g., all-GDN, all-SWA, all-GQA, small hidden dimension, small vocabulary), we compared full-sequence and token-by-token incremental outputs under `torch.allclose` with tolerances appropriate to `bfloat16` precision (`atol=1e-3`, `rtol=1e-2`).

An initial GDN-only test reported near-exact agreement (max absolute difference ≈ 1×10⁻⁶); this result was later found to be an artifact of an unintentional `float32` execution path in that specific test script, rather than representative of the model's actual `bfloat16` inference behavior (see Section 4.3).

SWA and GQA isolated tests, run correctly in `bfloat16`, showed bounded discrepancies on the order of 10⁻⁴ to 10⁻² — consistent with expected floating-point accumulation differences between batched and sequential reduction orderings, not indicative of logical errors.

### 3.2 Diagnosing an apparent GQA/SWA asymmetry

An initial single-run comparison showed GQA's isolated discrepancy (1.6×10⁻²) approximately two orders of magnitude larger than SWA's (1.2×10⁻⁴) under nominally identical conditions. Rather than accept a post hoc explanation, we required this asymmetry to be diagnosed with layer-level evidence:

- A layer-by-layer intermediate-tensor comparison confirmed that query, key, value, and attention-output tensors matched exactly (difference = 0.0) at the point of key/value concatenation, ruling out ordering or positional-encoding errors as the cause.
- Bypassing all `RMSNorm` layers in a controlled diagnostic reduced the divergence from 1.6×10⁻² to below 1×10⁻³, directly implicating the normalization layers' sequence-length-dependent reduction order (operating on a length-1 versus length-T tensor) as the mechanism, rather than the caching logic itself.
- A five-seed repetition of both SWA and GQA isolated tests showed both layer types exhibiting the same characteristic magnitudes (2⁻⁷ ≈ 0.0078 and 2⁻⁶ ≈ 0.0156, consistent with single-bit `bfloat16` rounding boundaries), with no consistent asymmetry between them. The original single-run comparison was determined to be within-distribution sampling variation rather than a systematic difference between the two layer types.

### 3.3 Combined end-to-end validation and discovery of a caching defect

Following independent validation of all three layer types, we tested the fully integrated `generate()` method against the naive full-reforward baseline using the real 12-layer heterogeneous architecture (GDN/SWA/GQA interleaved as in the production configuration), first on a randomly initialized model.

This test produced token-identical output across three random seeds (12/12 greedy decoding steps matched in each case), but with a substantially larger final-logit discrepancy (peak ≈ 0.69–0.86) than any isolated-layer test had shown. Given the magnitude of this increase relative to the isolated-component results, we treated the initially offered explanation ("expected bfloat16 accumulation variance") as unverified and required further diagnosis before acceptance — applying the same evidentiary standard used for the GQA/SWA asymmetry in Section 3.2.

A layer-by-layer breakdown of the combined 12-layer stack revealed that GDN layers contributed disproportionately to the accumulated divergence relative to SWA and GQA layers (approximately 10× larger per-layer contribution). Investigation of this finding revealed that the original isolated GDN test (Section 3.1) had inadvertently run in `float32` rather than `bfloat16`, meaning the near-exact result obtained there was not representative of the model's actual mixed-precision inference behavior; GDN's recurrent state updates and convolutional operations were confirmed to accumulate substantially more `bfloat16` rounding error per layer than a single attention operation.

Subsequently, direct evaluation against the model's actual trained checkpoint (rather than a randomly initialized model) revealed a further, more serious defect: the model's fallback logic for initializing an empty cache (`past_key_values = None` when no cache was yet provided) failed to construct a valid `Cache` object for the GDN branch, causing the underlying library to silently bypass caching and re-initialize recurrent state at every generation step rather than carrying it forward. This defect produced a maximum logit divergence of 14.5 and an 80% argmax mismatch rate against the baseline — an unambiguous failure that had not been detected by any prior isolated or randomly-initialized test, because small-magnitude random weights did not produce outputs sensitive enough to expose the missing state propagation.

This defect was not observable using randomly initialized weights and was only surfaced through direct validation against trained model parameters, underscoring that toy-configuration testing, while useful for isolating mechanism-level correctness, is not a substitute for validation against the actual deployed model.

The defect was corrected by explicitly constructing an empty `Cache()` instance whenever no prior cache was supplied, rather than passing `None` through to the underlying library (which interprets `None` as "caching disabled" rather than "cache not yet initialized"). Following this correction, the isolated GDN-only test (re-run correctly in `bfloat16`) produced a maximum absolute difference of 4.9×10⁻⁴, consistent in magnitude with the SWA and GQA isolated results and confirming the fix restored expected behavior at the component level.

## 4. Final Empirical Results (Trained Checkpoint)

Following the fix described in Section 3.3, we validated the complete incremental generation pipeline against the model's actual trained checkpoint (training step 25,797), using the production 12-layer configuration (d_model = 1024, vocabulary size = 48,000, GDN/SWA/GQA interleaved 1:1:1). Two independent evaluation runs were conducted:

| Run | Prompts | Generation steps | Close calls (top-2 margin < 1.0) | Argmax mismatches | Max logit divergence |
|---|---|---|---|---|---|
| 1 | 7 | 280 | 45.7% (128) | 1.79% (5) | 0.1250 |
| 2 | 12 | 480 | 45.6% (219) | 2.71% (13) | 0.1250 |

Two findings are notable. First, the maximum logit divergence between incremental and full-sequence generation was identical (0.1250) across both runs despite different prompts and step counts, indicating a stable, bounded precision ceiling rather than unbounded error accumulation. Second, manual inspection of all mismatch cases confirmed that argmax divergences occurred exclusively when the baseline's top-2 candidate logits were separated by a negligible margin (observed margin = 0.0000 in inspected examples) — i.e., cases in which the model's own prediction was effectively tied between two candidates, such that `bfloat16` rounding noise, rather than any structural error, determined which candidate the argmax operation selected. In such cases, both the baseline and cached outputs corresponded to plausible, semantically valid continuations (e.g., "independent" vs. "country" following an identical Bangla context).

## 5. Discussion and Limitations

We characterize the validated behavior as follows: incremental and full-sequence generation produce identical output except in a small, bounded subset of cases (~2-3% of generation steps in our evaluation) where the underlying model's own logit distribution is already near-tied between two candidates. This divergence is attributable to standard `bfloat16` floating-point non-associativity between differently-ordered (batched versus sequential) reduction operations, principally within the GDN and RMSNorm components, and is consistent with known behavior in other `bfloat16` inference settings rather than specific to this architecture or implementation.

We note three limitations of the present validation. First, generation lengths tested were bounded (30-50 tokens); whether the argmax mismatch rate remains stable over substantially longer generations (e.g., for summarization-length outputs) was not evaluated and is left to future work. Second, batched incremental decoding was not implemented or validated. Third, evaluation prompts, while drawn from real Bangla and English text, were not systematically sampled from a held-out benchmark; the reported margin distribution and mismatch rate should be read as indicative rather than as a precise population estimate.

We consider this a methodologically important finding for a broader audience: a caching implementation that passes extensive component-level and randomly-initialized end-to-end testing may still harbor defects that are only observable against trained model parameters, because small-magnitude random weights can fail to exercise the numerical sensitivity that real, trained parameters exhibit. We recommend that validation of inference-time modifications to trained language models include direct testing against the deployed checkpoint, not solely against synthetic or randomly initialized configurations.

## 6. Summary

Incremental decoding was implemented for a hybrid GDN-SWA-GQA architecture by (i) leveraging native cache support in the `flash-linear-attention` library for GDN, and (ii) implementing manual key/value caching for SWA and GQA, with RoPE application preserved through the existing, independently-validated rotary embedding module. Validation proceeded through isolated per-layer-type testing, diagnosis of an initially unexplained magnitude asymmetry (traced to normalization-layer reduction order), and combined end-to-end testing against both randomly initialized and trained model parameters — the latter of which revealed and enabled correction of a cache-initialization defect invisible to synthetic testing. The final validated implementation reproduces full-sequence generation behavior exactly except in a bounded ~2-3% of cases involving genuine model-internal uncertainty, with a stable maximum logit divergence of 0.125 attributable to standard mixed-precision floating-point behavior.
