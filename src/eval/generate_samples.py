import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

model_path = "tasmin-jahan/bangla-gsg"

print("Loading tokenizer...", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Loading model...", file=sys.stderr)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompts = [
    "বাংলাদেশের রাজধানী",
    "আমার নাম",
    "আমি আজ স্কুলে",
    "ঢাকা শহর অনেক",
    "বৃষ্টির দিনে আমার",
    "বিজ্ঞান ও প্রযুক্তি",
    "বাংলা ভাষা আমাদের",
    "মানুষের জীবনে",
    "কম্পিউটার বিজ্ঞান",
    "কৃত্রিম বুদ্ধিমত্তা",
    "আকাশের রঙ নীল কারণ",
    "বাঙালি সংস্কৃতি",
]

print("Generating samples...\n")
for p in prompts:
    inputs = tokenizer(p, return_tensors="pt").to(model.device)
    outputs = model.model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.3,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {p}")
    print(f"Output: {text}\n" + "-"*50)
