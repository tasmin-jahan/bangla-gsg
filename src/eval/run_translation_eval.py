import argparse
import json
import os
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sacrebleu

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model or checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to eval dataset (JSONL)")
    parser.add_argument("--direction", type=str, choices=["en_bn", "bn_en"], required=True)
    parser.add_argument("--output_csv", type=str, default="saved/logs/translation_eval_results.csv")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--few_shot", type=int, default=3, help="Number of few-shot examples to prepend")
    return parser.parse_args()

def extract_pairs(filepath, direction, max_samples):
    """Extract source and target from the raw v3 schema lines."""
    task_token = f"<|task_translate_{direction}|>"
    src_token = "<|lang_en|>" if direction == "en_bn" else "<|lang_bn|>"
    tgt_token = "<|lang_bn|>" if direction == "en_bn" else "<|lang_en|>"
    
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")
            if text.startswith(task_token):
                # Format: <|task_translate_en_bn|><|lang_en|>Hello<|lang_bn|>Nomoshkar
                text = text.replace(task_token, "")
                parts = text.split(tgt_token)
                if len(parts) == 2:
                    src_text = parts[0].replace(src_token, "")
                    tgt_text = parts[1]
                    pairs.append({"src": src_text, "tgt": tgt_text})
            if len(pairs) >= max_samples + 10:  # Grab a few extra for few-shot pool
                break
    return pairs

def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.eval()
    model.cuda()
    
    pairs = extract_pairs(args.data_path, args.direction, args.num_samples)
    if len(pairs) < args.few_shot + 1:
        raise ValueError("Not enough data to create few-shot examples and evaluate.")
        
    few_shot_pool = pairs[:args.few_shot]
    eval_pool = pairs[args.few_shot : args.few_shot + args.num_samples]
    
    # Construct Few-Shot Prefix
    task_token = f"<|task_translate_{args.direction}|>"
    src_token = "<|lang_en|>" if args.direction == "en_bn" else "<|lang_bn|>"
    tgt_token = "<|lang_bn|>" if args.direction == "en_bn" else "<|lang_en|>"
    stop_tokens = ["\n", "<|task_translate_en_bn|>", "<|task_translate_bn_en|>", "<|lang_en|>", "<|lang_bn|>"]
    
    few_shot_prefix = ""
    for p in few_shot_pool:
        few_shot_prefix += f"{task_token}{src_token}{p['src']}{tgt_token}{p['tgt']}\n"
        
    predictions = []
    references = []
    
    print(f"Starting evaluation ({len(eval_pool)} samples, {args.few_shot}-shot)...")
    for item in tqdm(eval_pool):
        prompt = few_shot_prefix + f"{task_token}{src_token}{item['src']}{tgt_token}"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generation with strict stopping criteria
        # Note: If the tokenizer supports custom stopping criteria, we would use StoppingCriteriaList here.
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.1,
                do_sample=False
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Enforce strict stopping: truncate at the first newline or task token
        for stop in stop_tokens:
            if stop in generated_text:
                generated_text = generated_text.split(stop)[0]
                
        predictions.append(generated_text.strip())
        references.append([item["tgt"].strip()])
        
    # Calculate chrF and BLEU
    chrf = sacrebleu.corpus_chrf(predictions, references)
    bleu = sacrebleu.corpus_bleu(predictions, references)
    
    print(f"\\nResults for {args.direction}:")
    print(f"chrF2 Score: {chrf.score:.2f}")
    print(f"BLEU Score:  {bleu.score:.2f}")
    
    # Save results without overwriting
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    new_data = {
        "timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")],
        "model_path": [args.model_path],
        "direction": [args.direction],
        "samples": [len(eval_pool)],
        "few_shot": [args.few_shot],
        "chrf_score": [chrf.score],
        "bleu_score": [bleu.score]
    }
    new_df = pd.DataFrame(new_data)
    
    if os.path.exists(args.output_csv):
        # Append to existing
        new_df.to_csv(args.output_csv, mode="a", header=False, index=False)
        print(f"Appended results to {args.output_csv}")
    else:
        # Create new
        new_df.to_csv(args.output_csv, index=False)
        print(f"Created new results file at {args.output_csv}")

if __name__ == "__main__":
    main()
