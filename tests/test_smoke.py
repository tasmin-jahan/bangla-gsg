"""
Smoke test for BanglaGSG model construction and forward pass.

Tests:
1. Config loads from YAML and validates
2. Model builds with correct layer types
3. Forward pass produces correct output shape
4. Parameter counting works
5. Gradient flows through all layer types
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.config import BanglaGSGConfig
from src.model.model import BanglaGSGModel


def test_config_from_yaml():
    """Test config loads from YAML."""
    config = BanglaGSGConfig.from_yaml("configs/banglagsg_12l.yaml")
    assert config.n_layers == 12
    assert config.d_model == 1024
    assert len(config.layer_types) == 12
    assert config.layer_types == [
        "gdn", "swa", "gqa",
        "gdn", "swa", "gqa",
        "gdn", "swa", "gqa",
        "gdn", "swa", "gqa",
    ]
    print("✅ Config loads from YAML correctly")
    print(config.summary())
    return config


def test_config_validation():
    """Test config validation catches bad layer types."""
    try:
        BanglaGSGConfig(
            layer_types=["gdn", "mamba", "gqa"] * 4,
            n_layers=12,
        )
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "invalid" in str(e).lower()
        print("✅ Config validation rejects invalid layer types")


def test_model_build(config):
    """Test model builds correctly."""
    model = BanglaGSGModel(config)

    # Check layer types
    for i, layer in enumerate(model.layers):
        assert layer.layer_type == config.layer_types[i], (
            f"Layer {i}: expected {config.layer_types[i]}, got {layer.layer_type}"
        )

    # Count layer types
    gdn_count = sum(1 for l in model.layers if l.layer_type == "gdn")
    swa_count = sum(1 for l in model.layers if l.layer_type == "swa")
    gqa_count = sum(1 for l in model.layers if l.layer_type == "gqa")
    assert gdn_count == 4
    assert swa_count == 4
    assert gqa_count == 4
    print(f"✅ Model built: {gdn_count} GDN + {swa_count} SWA + {gqa_count} GQA layers")

    return model


def test_forward_pass(model, config):
    """Test forward pass produces correct output shape."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Flash Attention requires bf16/fp16 — cast the entire model
    model = model.to(device=device, dtype=torch.bfloat16)

    B, T = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (B, T, config.vocab_size), (
        f"Expected shape {(B, T, config.vocab_size)}, got {logits.shape}"
    )
    assert not torch.isnan(logits).any(), "NaN in logits!"
    assert not torch.isinf(logits).any(), "Inf in logits!"
    print(f"✅ Forward pass: input ({B},{T}) → logits {tuple(logits.shape)}")
    return device


def test_param_counts(model):
    """Test parameter counting."""
    counts = model.count_parameters()
    print(f"  Total:     {counts['total']:>12,}")
    print(f"  Trainable: {counts['trainable']:>12,}")
    print(f"  Embedding: {counts['embedding']:>12,}")
    print(f"  GDN:       {counts['gdn']:>12,}")
    print(f"  SWA:       {counts['swa']:>12,}")
    print(f"  GQA:       {counts['gqa']:>12,}")
    print(f"  FFN:       {counts['ffn']:>12,}")
    assert counts['total'] > 0
    assert counts['gdn'] > 0
    assert counts['ffn'] > 0
    print("✅ Parameter counting works")


def test_gradient_flow(model, config, device):
    """Test gradients flow through all layer types."""
    model.train()
    B, T = 1, 64
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)

    # Model is already in bf16 from test_forward_pass
    logits = model(input_ids)
    loss = logits.float().sum()  # cast to float32 for stable backward

    loss.backward()

    # Check each layer type has non-zero gradients
    for i, layer in enumerate(model.layers):
        has_grad = False
        for name, p in layer.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, f"Layer {i} ({layer.layer_type}) has no gradients!"

    print("✅ Gradient flow verified through all 12 layers")


def test_gradient_checkpointing(model, config, device):
    """Test gradient checkpointing can be enabled."""
    model.gradient_checkpointing_enable()
    for layer in model.layers:
        assert layer.gradient_checkpointing
    model.gradient_checkpointing_disable()
    for layer in model.layers:
        assert not layer.gradient_checkpointing
    print("✅ Gradient checkpointing enable/disable works")


def main():
    print("=" * 60)
    print("  BanglaGSG Smoke Test")
    print("=" * 60)

    config = test_config_from_yaml()
    print()
    test_config_validation()
    print()
    model = test_model_build(config)
    print()
    device = test_forward_pass(model, config)
    print()
    test_param_counts(model)
    print()
    test_gradient_flow(model, config, device)
    print()
    test_gradient_checkpointing(model, config, device)

    print()
    print("=" * 60)
    print("  ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
