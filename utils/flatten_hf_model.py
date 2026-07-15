import os
import re

model_files = [
    "src/model/config.py",
    "src/model/embeddings.py",
    "src/model/rope.py",
    "src/model/ffn.py",
    "src/model/attention.py",
    "src/model/swa.py",
    "src/model/gdn.py",
    "src/model/model.py",
]

all_imports = set()
all_code = []

for filepath in model_files:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # We need to rename the internal raw config to avoid clashing with the HF config
    content = content.replace("BanglaGSGConfig", "RawConfig")
    
    # Split into lines to extract imports
    lines = content.split("\n")
    for line in lines:
        if line.startswith("import ") or (line.startswith("from ") and not line.startswith("from .")):
            all_imports.add(line.strip())
        elif line.startswith("from ."):
            continue # strip internal relative imports
        else:
            all_code.append(line)
    all_code.append("\n\n")

# Now read the original modeling_banglagsg.py
with open("src/hf_integration/modeling_banglagsg.py", "r", encoding="utf-8") as f:
    hf_lines = f.readlines()

hf_imports = set()
hf_code = []
for line in hf_lines:
    if line.startswith("import ") or (line.startswith("from ") and not line.startswith("from .")):
        hf_imports.add(line.strip())
    elif line.startswith("from .configuration_banglagsg import"):
        # Put this back manually since we need it
        hf_imports.add("from .configuration_banglagsg import BanglaGSGConfig")
    elif line.startswith("from .model"):
        continue # strip the relative model imports that crashed HF
    else:
        hf_code.append(line)

final_imports = sorted(list(all_imports | hf_imports))

fallback_code = """
import torch.nn.functional as F

def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
    # q, k, v are (B, T, H, D)
    # PyTorch SDPA expects (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Handle Grouped Query Attention (repeat KV heads to match Q heads)
    if k.shape[1] < q.shape[1]:
        num_kv_groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)
        
    attn_mask = None
    is_causal = causal
    
    # Handle Sliding Window Attention if window_size is provided
    if window_size != (-1, -1):
        is_causal = False
        left, right = window_size
        T = q.shape[2]
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device)
        mask = torch.tril(mask, diagonal=right)
        mask = torch.triu(mask, diagonal=-left)
        if causal:
            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
            mask = mask & causal_mask
        attn_mask = mask.unsqueeze(0).unsqueeze(0)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal
    )
    # Return to (B, T, H, D) layout
    return out.transpose(1, 2)
"""

# Ensure __future__ imports are at the absolute top
future_imports = [imp for imp in final_imports if "__future__" in imp]
other_imports = [imp for imp in final_imports if "__future__" not in imp and "flash_attn" not in imp]

# Assemble the giant file
final_text = "\n".join(future_imports) + "\n" + "\n".join(other_imports) + "\n\n" + fallback_code + "\n\n" + "\n".join(all_code) + "\n".join(hf_code)

with open("src/hf_integration/modeling_banglagsg.py", "w", encoding="utf-8") as f:
    f.write(final_text)

print("Flattened src/model/* into src/hf_integration/modeling_banglagsg.py!")
