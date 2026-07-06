# BanglaGSG Optimization Tricks & Rationale

This document outlines the performance and memory optimization techniques implemented in the BanglaGSG training pipeline to maximize training speed on consumer hardware (12GB VRAM).

## 1. Mixed Precision (BF16 Autocast)
**Implementation**: `torch.autocast("cuda", dtype=torch.bfloat16)`
**Rationale**: 
- **Speed**: Executes all heavy matrix multiplications (Linear layers, attention projections, FFNs) using ultra-fast BF16 Tensor Cores, offering up to a 3x speedup over FP32.
- **Precision**: We keep the master weights and optimizer momentum states in high-precision `float32`. This prevents small gradients from being lost or truncated during optimizer updates, which is a common failure mode in "Pure BF16" training.

## 2. FlashAttention-2
**Implementation**: `flash_attn_func(q.to(bfloat16), k.to(bfloat16), v.to(bfloat16), causal=True)`
**Rationale**:
- Replaces standard PyTorch attention with a highly optimized fused CUDA kernel.
- Scales linearly in memory with sequence length instead of quadratically, completely eliminating the VRAM bottleneck of the attention mechanism.

## 3. PyTorch 2.0 Compiler (`torch.compile`)
**Implementation**: `model = torch.compile(model)`
**Rationale**:
- Reads the entire PyTorch execution graph before training starts and fuses multiple small operations (like activation functions, layer norms, and addition) into single, unified CUDA kernels.
- **Why default mode?** We specifically avoid `mode="reduce-overhead"` because that mode utilizes CUDA Graphs. When paired with heavy gradient accumulation, CUDA Graphs cause memory buffer overwrites (and OOMs). The default mode gives massive kernel fusion speedups without the rigid CUDA Graph memory constraints.
- **Why GDN is superior to Mamba here**: Mamba relies on hand-written C++ kernels (`mamba_inner_fn`) that constantly trigger "graph breaks" and crash the PyTorch compiler. GDN (Gated Delta Networks) is built using standard PyTorch primitives and Triton kernels, allowing `torch.compile` to perfectly optimize the entire architecture without a single graph break.

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
**Implementation**: Signal interceptors for `Ctrl+C` in `trainer.py`
**Rationale**:
- Instead of instantly crashing and losing up to 999 steps of progress when interrupted, the trainer intercepts the kill signal, finishes the current operation, explicitly dumps an emergency checkpoint to disk, and cleanly exits. This guarantees 100% progress retention.
