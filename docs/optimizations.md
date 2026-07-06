# BanglaGSG Optimization Tricks & Rationale

This document outlines the performance and memory optimization techniques implemented in the BanglaGSG training pipeline to maximize training speed on consumer hardware (12GB VRAM).

## 1. Mixed Precision (BF16 Autocast)
**Implementation**: `torch.autocast("cuda", dtype=torch.bfloat16)`
**Rationale**: 
- **Speed**: Executes all heavy matrix multiplications (Linear layers, attention projections, FFNs) using ultra-fast BF16 Tensor Cores, offering up to a 3x speedup over FP32.
- **Precision**: We keep the master weights and optimizer momentum states in high-precision `float32`. This prevents small gradients from being lost or truncated during optimizer updates, which is a common failure mode in "Pure BF16" training.

## 2. FlashAttention-2
**Implementation**: `flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), causal=True)`
**Rationale**:
- Replaces standard PyTorch attention with a highly optimized fused CUDA kernel.
- Scales linearly in memory with sequence length instead of quadratically, completely eliminating the VRAM bottleneck of the attention mechanism.
- **Explicit Casting Requirement**: Because we maintain FP32 master weights to prevent gradient underflow, we explicitly cast the Q, K, and V tensors down to `bfloat16` directly before passing them into `flash_attn_func` in both our GQA and SWA layers. FlashAttention is a raw hardware wrapper for Tensor Cores and completely rejects FP32 inputs.

## 3. PyTorch 2.0 Compiler (`torch.compile`) — **[CURRENTLY DISABLED]**
**Implementation**: `compile_model: false` (in `configs/default_training.yaml`)
**Rationale**:
- The PyTorch compiler usually provides massive kernel fusion speedups without graph breaks.
- **Why it is currently disabled:** During testing, the underlying NVIDIA PTX Assembler (`ptxas`) suffered a fatal crash (Error Code -2) when attempting to compile the incredibly complex Triton kernels underlying the Gated Delta Networks (GDN). This is a known register-allocation failure on RTX 40-series (sm_89) GPUs. 
- We bypass this by running in standard eager mode, falling back to the standard JIT-compiled Triton kernels which still offer phenomenal performance.

## 4. TensorFloat-32 (TF32) Math
**Implementation**: `torch.backends.cuda.matmul.allow_tf32 = True`
**Rationale**:
- For any internal operations that do not natively trigger the `bfloat16` autocast, TF32 allows the Nvidia Ampere+ GPU (RTX 4070 SUPER) to execute FP32 matrix multiplications using Tensor Cores. It operates at the speed of FP16 but maintains the numerical range of FP32.

## 5. Fused AdamW Kernel
**Implementation**: `torch.optim.AdamW(..., fused=True)`
**Rationale**:
- Fuses the element-wise optimizer update loop into a single CUDA kernel. Instead of doing a separate VRAM read/write for every single parameter's gradient update, it processes them concurrently, significantly speeding up the `optimizer.step()` phase.

## 6. Gradient Accumulation
**Implementation**: `batch_size: 2`, `accumulation_steps: 126` (Effective Batch Size = 252)
**Rationale**:
- A batch size of 252 is required for stable training, but the massive `[252, 2047, 65024]` tensors would require hundreds of gigabytes of VRAM. 
- Gradient accumulation allows us to process the batch sequentially in micro-batches of 2, summing the gradients iteratively, and only stepping the optimizer once at the end. This mathematically perfectly simulates a batch size of 252 while keeping peak VRAM firmly under 11 GB.

## 7. Gradient Checkpointing (Activation Checkpointing)
**Implementation**: `model.gradient_checkpointing_enable()`
**Rationale**:
- Standard training stores all forward-pass activations in VRAM so they can be reused during the backward pass. This takes an astronomical amount of memory.
- Gradient checkpointing throws away the activations after the forward pass and simply recomputes them on the fly during the backward pass. It trades a slight (~20%) compute penalty for massive VRAM savings, which allows us to fit the model into 12GB.

## 8. Graceful Interrupts (SIGINT Handling)
**Implementation**: Signal interceptors for `Ctrl+C` in `trainer.py` and `worker_init_fn=ignore_sigint` in `collator.py`.
**Rationale**:
- Instead of instantly crashing and losing up to 999 steps of progress when interrupted, the trainer intercepts the kill signal, finishes the current operation, explicitly dumps an emergency checkpoint to disk, and cleanly exits. This guarantees 100% progress retention.
- **DataLoader Immunity:** The Python `multiprocessing` workers used by the DataLoader are explicitly told to ignore `SIGINT`. This prevents the workers from dying instantly on `Ctrl+C`, ensuring the main training loop can finish its final forward/backward pass without throwing a `RuntimeError: DataLoader worker exited unexpectedly`.
