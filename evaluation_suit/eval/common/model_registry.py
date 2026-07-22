"""
Model Registry — unified model loading for the eval suite.

Single `load_model(key)` function used by every task script.
Reads model metadata from configs/models.yaml and returns a LoadedModel
dataclass with model, tokenizer, and metadata.

Usage:
    python -m evaluation_suit.eval.00_common.model_registry gamba
    python -m evaluation_suit.eval.00_common.model_registry gsg
    python -m evaluation_suit.eval.00_common.model_registry banglabert
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)


@dataclass
class LoadedModel:
    """Container for a loaded model + tokenizer + metadata."""
    key: str
    display_name: str
    model: Any  # nn.Module (PreTrainedModel subclass)
    tokenizer: Any  # PreTrainedTokenizer subclass
    model_type: str  # "causal_lm" or "masked_lm"
    seq_len: int
    device: torch.device


# Path to configs relative to this file
_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def _load_models_yaml() -> dict:
    """Load models.yaml config."""
    config_path = _CONFIGS_DIR / "models.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"models.yaml not found at {config_path}. "
            "Expected at evaluation_suit/configs/models.yaml"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _check_requirements(key: str, cfg: dict) -> None:
    """Hard-fail if required packages are missing (e.g. bnunicodenormalizer for GSG)."""
    requires = cfg.get("requires", [])
    for pkg in requires:
        try:
            __import__(pkg)
        except ImportError:
            raise ImportError(
                f"Model '{key}' requires package '{pkg}' but it is not installed. "
                f"Install it with: pip install {pkg}\n"
                f"This is a HARD FAILURE — '{key}' will produce degraded output "
                f"without '{pkg}' and results would be invalid."
            )


def load_model(
    key: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> LoadedModel:
    """
    Load a model by key from the registry.

    Args:
        key: One of "gamba", "gsg", "banglabert".
        device: Target device. Defaults to "cuda" if available, else "cpu".
        dtype: Model dtype. Default bfloat16 for causal LMs.

    Returns:
        LoadedModel with model, tokenizer, and metadata.

    Raises:
        ImportError: If required packages are missing (hard fail for GSG).
        ValueError: If key is not in models.yaml.
    """
    configs = _load_models_yaml()

    if key not in configs:
        valid_keys = list(configs.keys())
        raise ValueError(
            f"Unknown model key '{key}'. Valid keys: {valid_keys}"
        )

    cfg = configs[key]

    # Hard-fail on missing requirements
    _check_requirements(key, cfg)

    hf_repo = cfg["hf_repo"]
    model_type = cfg["model_type"]
    trust_remote_code = cfg.get("trust_remote_code", False)
    seq_len = cfg.get("seq_len", 2048)
    display_name = cfg.get("display_name", key)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"[ModelRegistry] Loading '{display_name}' from {hf_repo}...")
    print(f"  model_type={model_type}, trust_remote_code={trust_remote_code}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_repo,
        trust_remote_code=trust_remote_code,
    )

    # Load model based on type
    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(
            hf_repo,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
    elif model_type == "masked_lm":
        model = AutoModelForMaskedLM.from_pretrained(
            hf_repo,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype if key != "banglabert" else torch.float32,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}' for key '{key}'")

    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    claimed = cfg.get("claimed_params_m", 0) * 1e6
    actual_m = total_params / 1e6

    print(f"  Loaded: {actual_m:.1f}M params (claimed: {claimed/1e6:.0f}M)")

    if claimed > 0:
        deviation = abs(total_params - claimed) / claimed
        if deviation > 0.05:
            print(
                f"  ⚠ WARNING: Param count deviation {deviation*100:.1f}% "
                f"exceeds 5% threshold! "
                f"Actual={total_params:,}, Claimed={int(claimed):,}"
            )

    # Ensure tokenizer has pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return LoadedModel(
        key=key,
        display_name=display_name,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        seq_len=seq_len,
        device=device,
    )


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m evaluation_suit.eval.00_common.model_registry <key>")
        print("  Keys: gamba, gsg, banglabert")
        sys.exit(1)

    key = sys.argv[1]
    loaded = load_model(key)

    print(f"\n{'='*60}")
    print(f"  Model: {loaded.display_name}")
    print(f"  Key: {loaded.key}")
    print(f"  Type: {loaded.model_type}")
    print(f"  Device: {loaded.device}")
    print(f"  Seq Len: {loaded.seq_len}")

    total = sum(p.numel() for p in loaded.model.parameters())
    trainable = sum(p.numel() for p in loaded.model.parameters() if p.requires_grad)
    print(f"  Total Params: {total:,} ({total/1e6:.1f}M)")
    print(f"  Trainable Params: {trainable:,} ({trainable/1e6:.1f}M)")

    # Quick tokenizer test
    test_text = "বাংলাদেশ আমার প্রিয় দেশ"
    tokens = loaded.tokenizer.encode(test_text)
    decoded = loaded.tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"\n  Tokenizer test:")
    print(f"    Input:   '{test_text}'")
    print(f"    Tokens:  {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
    print(f"    Decoded: '{decoded}'")
    print(f"{'='*60}")
