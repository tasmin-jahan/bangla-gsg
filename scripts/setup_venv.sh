#!/bin/bash
# ============================================================
# GDN-SWA-GQA Stack — Install Script
# Python 3.12 | PyTorch 2.12.0 | CUDA 13.0 | RTX 4070 SUPER
#
# Verified version matrix (May 2026):
#   torch                    2.12.0+cu130
#   torchvision              auto (cu130)
#   torchaudio               auto (cu130)
#   causal-conv1d            latest (short-conv used by GDN)
#   flash-linear-attention   HEAD (provides Gated DeltaNet / GDN)
#   flash-attn               2.8.3 (provides Sliding Window Attention / SWA + GQA)
#   xformers                 latest (cu130)
#
# KEY FIX — rsqrt/rsqrtf noexcept conflict (CUDA 13 + glibc >= 2.42):
#   glibc 2.42+ declares rsqrt/rsqrtf with noexcept(true).
#   CUDA 13.0's crt/math_functions.h declares them without noexcept.
#   When both end up in the same translation unit, nvcc errors out.
#
#   The previous script tried -D_MATHCALLS_H (suppress glibc's mathcalls.h),
#   but this DOES NOT WORK: CUDA's own header is pulled in first without
#   noexcept, and glibc's header still reaches it via a different include path.
#
#   THE CORRECT FIX (confirmed on NVIDIA developer forums, Jan 2026):
#     Patch /usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h
#     to add noexcept(true) to the rsqrt/rsqrtf declarations so they match glibc.
#     This is a one-line sed on the CUDA header — safe, reversible, and the only
#     approach that actually works.
#
# Usage:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

# ============================================================
# Step 0: Python 3.12 + venv
# ============================================================
echo "=== Step 0: Python 3.12 + venv ==="

if ! command -v python3.12 &>/dev/null; then
  echo "Python 3.12 not found — adding deadsnakes PPA..."
  sudo apt-get update -qq
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -qq
  sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
else
  echo "Python 3.12 found: $(python3.12 --version)"
fi

if [ ! -d "$VENV_PATH" ]; then
  echo "Creating fresh venv at $VENV_PATH"
  python3.12 -m venv "$VENV_PATH"
else
  echo "Venv already exists — using it"
fi

source "$VENV_PATH/bin/activate"
echo "Active python: $(which python) — $(python --version)"
pip install --upgrade pip setuptools wheel

# ============================================================
# Step 1: GCC 13
# ============================================================
# CUDA 13.0 + GCC 15 (Ubuntu 26.04 default) = broken builds for many CUDA
# extensions. GCC 13 is the proven safe compiler for flash-attn, fla, etc.
echo ""
echo "=== Step 1: GCC 13 ==="

if ! command -v gcc-13 &>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y gcc-13 g++-13
else
  echo "GCC 13 already installed: $(gcc-13 --version | head -1)"
fi

export CC=gcc-13
export CXX=g++-13
echo "Using CC=$CC, CXX=$CXX"

# ============================================================
# Step 2: CUDA 13.0 toolkit
# ============================================================
echo ""
echo "=== Step 2: CUDA 13.0 toolkit ==="

if nvcc --version 2>/dev/null | grep -q "13\."; then
  echo "CUDA 13.x already installed — skipping apt install."
else
  echo "Installing CUDA 13.0 toolkit..."
  if [ ! -f "cuda-keyring_1.1-1_all.deb" ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  else
    echo "cuda-keyring deb already present — skipping download."
  fi
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  rm -f cuda-keyring_1.1-1_all.deb
  sudo apt-get update -qq
  sudo apt-get install -y cuda-toolkit-13-0

  grep -qxF 'export CUDA_HOME=/usr/local/cuda-13.0' ~/.bashrc || \
    echo 'export CUDA_HOME=/usr/local/cuda-13.0' >> ~/.bashrc
  grep -qxF 'export PATH=$CUDA_HOME/bin:$PATH' ~/.bashrc || \
    echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
  grep -qxF 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' ~/.bashrc || \
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

if [ -d "/usr/local/cuda-13.0" ]; then
  export CUDA_HOME=/usr/local/cuda-13.0
elif [ -d "/usr/local/cuda" ]; then
  export CUDA_HOME=/usr/local/cuda
else
  echo "ERROR: CUDA 13.0 not found at /usr/local/cuda-13.0 or /usr/local/cuda"
  exit 1
fi

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc: $(nvcc --version | grep release)"

# ============================================================
# Step 3: Patch CUDA 13 math_functions.h (rsqrt/rsqrtf noexcept fix)
# ============================================================
# This is the CORRECT fix for the build failure you're hitting.
#
# The problem: glibc 2.42+ added noexcept(true) to rsqrt/rsqrtf in
# bits/mathcalls.h. CUDA 13.0's crt/math_functions.h does not have noexcept,
# so when both headers end up in scope, nvcc emits:
#   "exception specification is incompatible with that of previous function rsqrt"
#
# The fix: add noexcept(true) to CUDA's declarations so they match glibc.
# This is a safe, targeted change that doesn't affect CUDA's semantics.
# A backup of the original file is saved as math_functions.h.bak.
#
# Source: https://forums.developer.nvidia.com/t/354510 (Dec 2025, confirmed working)
echo ""
echo "=== Step 3: Patching CUDA math_functions.h (rsqrt/rsqrtf noexcept fix) ==="

MATH_H="$CUDA_HOME/targets/x86_64-linux/include/crt/math_functions.h"

if [ ! -f "$MATH_H" ]; then
  echo "ERROR: CUDA math header not found at $MATH_H"
  echo "Check that CUDA 13.0 toolkit is fully installed."
  exit 1
fi

if grep -q 'rsqrt(double x) noexcept' "$MATH_H"; then
  echo "Patch already applied — skipping."
else
  echo "Backing up original to ${MATH_H}.bak"
  sudo cp "$MATH_H" "${MATH_H}.bak"

  echo "Applying noexcept patch..."
  # Add noexcept(true) to rsqrt (double) declaration
  sudo sed -i \
    's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double \+rsqrt(double x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x) noexcept (true);/' \
    "$MATH_H"

  # Add noexcept(true) to rsqrtf (float) declaration
  sudo sed -i \
    's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float \+rsqrtf(float x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x) noexcept (true);/' \
    "$MATH_H"

  # Verify the patch took effect
  if grep -q 'rsqrt(double x) noexcept' "$MATH_H"; then
    echo "Patch applied successfully."
  else
    echo "WARNING: sed substitution may not have matched. Checking file..."
    echo "Lines around rsqrt in $MATH_H:"
    grep -n 'rsqrt' "$MATH_H" | head -20
    echo ""
    echo "If noexcept is still missing, manually edit $MATH_H and add"
    echo "'noexcept (true)' after rsqrt(double x) and rsqrtf(float x)."
    echo "Continuing anyway — build may still fail if the patch didn't apply."
  fi
fi

# ============================================================
# Step 4: Clean any previous installs
# ============================================================
echo ""
echo "=== Step 4: Clean previous installs ==="
pip uninstall -y \
  flash-linear-attention fla causal-conv1d flash-attn \
  torch torchvision torchaudio triton \
  tilelang quack-kernels \
  nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base \
  cuda-python cuda-bindings cuda-toolkit \
  xformers 2>/dev/null || true
echo "Clean done."

# ============================================================
# Step 5: PyTorch 2.12.0 + CUDA 13.0
# ============================================================
echo ""
echo "=== Step 5: PyTorch 2.12.0 + CUDA 13.0 ==="
pip install torch==2.12.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130

python -c "
import torch
print('torch:   ', torch.__version__)
print('cuda:    ', torch.version.cuda)
print('gpu:     ', torch.cuda.get_device_name(0))
print('bf16:    ', torch.cuda.is_bf16_supported())
"

# ============================================================
# Step 6: causal-conv1d
# ============================================================
# Required by GDN's short-conv pre-mixing step.
echo ""
echo "=== Step 6: causal-conv1d ==="
CC=gcc-13 CXX=g++-13 \
  pip install "causal-conv1d>=1.4.0" --no-build-isolation

# ============================================================
# Step 7: flash-linear-attention (provides GDN — Gated DeltaNet)
# ============================================================
# fla-org/flash-linear-attention is not pushed to PyPI as a stable release;
# install directly from source/HEAD.
echo ""
echo "=== Step 7: flash-linear-attention (GDN / Gated DeltaNet) ==="

CC=gcc-13 CXX=g++-13 \
  pip install -U "git+https://github.com/fla-org/flash-linear-attention.git" \
  --no-build-isolation

python -c "from fla.layers.gated_deltanet import GatedDeltaNet; print('fla: GatedDeltaNet (GDN) OK')"

echo "Dropping page cache to free RAM before next build..."
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# ============================================================
# Step 8: Flash Attention 2.8.3 (provides SWA + GQA)
# ============================================================
# Sliding Window Attention is just flash-attn invoked with window_size=(left, right).
# Grouped Query Attention is handled natively by flash-attn's kv-head broadcasting.
echo ""
echo "=== Step 8: Flash Attention 2.8.3 (Sliding Window Attention / SWA + GQA) ==="
CC=gcc-13 CXX=g++-13 MAX_JOBS=2 \
  pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

python -c "
import flash_attn
print('flash-attn:', flash_attn.__version__)
from flash_attn import flash_attn_func
import inspect
sig = inspect.signature(flash_attn_func)
assert 'window_size' in sig.parameters, 'window_size arg missing — SWA unavailable'
print('flash-attn: window_size (SWA) arg confirmed present')
"

echo "Dropping page cache to free RAM before xformers build..."
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# ============================================================
# Step 9: xformers
# ============================================================
echo ""
echo "=== Step 9: xformers ==="
pip install xformers --no-cache-dir --index-url https://download.pytorch.org/whl/cu130

# ============================================================
# Step 10: LLM + CV stack
# ============================================================
echo ""
echo "=== Step 10: LLM + CV stack ==="

# --- LLM / Training ---
pip install \
  "transformers>=4.51.0" \
  "accelerate>=1.6.0" \
  "datasets>=3.5.0" \
  "tokenizers>=0.21.0" \
  "safetensors>=0.5.0" \
  "huggingface-hub>=0.30.0" \
  "peft>=0.15.0" \
  "trl>=0.16.0" \
  "bitsandbytes>=0.45.0" \
  "autoawq>=0.2.8" \
  "deepspeed>=0.16.0" \
  "sentencepiece>=0.2.0" \
  "tiktoken>=0.9.0" \
  "einops>=0.8.0" \
  "rotary-embedding-torch>=0.8.6"

# --- Computer Vision ---
pip install \
  "timm>=1.0.15" \
  "Pillow>=11.0.0" \
  "albumentations>=2.0.0" \
  "opencv-python-headless>=4.11.0" \
  "kornia>=0.8.0" \
  "torchmetrics>=1.6.0" \
  "supervision>=0.25.0"

# --- Utilities / Monitoring ---
pip install \
  "ninja>=1.11.0" \
  "packaging>=24.0" \
  "torch-ema>=0.3" \
  "wandb>=0.19.0" \
  "tensorboard>=2.19.0" \
  "tqdm>=4.67.0" \
  "numpy>=2.0.0" \
  "scipy>=1.15.0" \
  "matplotlib>=3.10.0" \
  "psutil>=7.0.0"

# ============================================================
# Final verification
# ============================================================
echo ""
echo "=== Final verification ==="
python - <<'EOF'
import sys, torch

print(f"python:         {sys.version.split()[0]}")
print(f"torch:          {torch.__version__}")
print(f"cuda:           {torch.version.cuda}")
print(f"gpu:            {torch.cuda.get_device_name(0)}")
print(f"bf16:           {torch.cuda.is_bf16_supported()}")

from fla.layers.gated_deltanet import GatedDeltaNet
print("fla:            GatedDeltaNet (GDN) OK")

import flash_attn
print(f"flash-attn:     {flash_attn.__version__} (SWA + GQA)")

import xformers
print(f"xformers:       {xformers.__version__}")

import transformers
print(f"transformers:   {transformers.__version__}")

import peft
print(f"peft:           {peft.__version__}")

import bitsandbytes
print(f"bitsandbytes:   {bitsandbytes.__version__}")

import timm
print(f"timm:           {timm.__version__}")

import albumentations
print(f"albumentations: {albumentations.__version__}")

import cv2
print(f"opencv:         {cv2.__version__}")

print("\n✅ All good.")
EOF

pip freeze > "$SCRIPT_DIR/requirements_lock.txt"
echo ""
echo "Lock file saved → requirements_lock.txt"
echo "To reactivate: source .venv/bin/activate"
