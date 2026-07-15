---
language:
- bn
- en
library_name: transformers
tags:
- custom-architecture
- bangla
- gdn
- swa
- gqa
- language-model
license: cc-by-nc-sa-4.0
datasets:
- ahmed-farhanur-rashid/bn-foundational-pretrain-corpus
---

# BanglaGSG

BanglaGSG is a custom hybrid language model trained from scratch on a mixed corpus of Bangla (\~7.37B tokens), English (\~1.23B tokens) and Dual Translation set (\~1.03B tokens) totalling at \~9.62B tokens. It leverages a unique architectural blend of Grouped Query Attention (GQA), Sliding Window Attention (SWA), and Gated Delta Networks (GDN) to achieve high performance and efficient inference for the Bengali language.

## Model Details
- **Architecture:** Hybrid (GQA + SWA + GDN)
- **Language(s):** Bengali (Primary), English (Secondary/Translation)
- **Training Data:** A curated 9.6B token mixture of web-crawled Bengali text, English monolingual data, and high-quality parallel translation pairs. All data underwent rigorous deduplication and strict NFC normalization via the custom `bnunicodenormalizer` pipeline.
- **Parameters:** ~185M

## Requirements & Setup

Because this model relies on a custom architecture and requires strict linguistic normalization, you **must** install the normalizer and enable `trust_remote_code=True` when loading it.

**1. Install the required normalizer:**
```bash
pip install bnunicodenormalizer
```
*(If this package is missing, the custom tokenizer will fallback to raw text, which may cause severe hallucinations or degraded performance since the model expects strict NFC normalization).*

## Usage

You can load the model seamlessly using the Hugging Face `transformers` library:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "tasmin-jahan/bangla-gsg"

# The custom tokenizer automatically applies the bnunicodenormalizer behind the scenes
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load the custom hybrid model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "বাংলাদেশের রাজধানী"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Architecture Notes
This model does not use a standard Transformer block. Instead, it utilizes:
- **GDN (Gated Delta Networks):** For highly efficient, linear-time sequential processing.
- **SWA (Sliding Window Attention):** To maintain local context with lower memory overhead.
- **GQA (Grouped Query Attention):** For optimized key-value caching during generation.

## Limitations
- **Normalization Dependency:** The model is highly sensitive to text format. Raw, un-normalized inputs will yield poor results. Always ensure `bnunicodenormalizer` is installed.
- **Custom Code:** Due to the hybrid architecture, this model requires executing custom Python scripts (`modeling_banglagsg.py`, `configuration_banglagsg.py`, `tokenization_banglagsg.py`) upon loading.