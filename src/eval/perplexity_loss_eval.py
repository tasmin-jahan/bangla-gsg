"""
BanglaGSG Perplexity and Loss Evaluation.

Evaluates the exported model against pre-tokenized validation shards (.npy) across 
different data subsets (e.g., Bangla, English, Translation). Computes exact Cross-Entropy Loss, 
Z-Loss, and Perplexity, and saves the aggregated metrics to a YAML report.

Usage:
    cd bangla-gsg/
    python src/eval/perplexity_loss_eval.py \
        --model_dir saved/model/default \
        --data_dir saved/data/eval \
        --report_dir saved/reports \
        --batch_size 2
"""

import argparse
import sys
import os
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from pathlib import Path

# Add project root to path so 'from src.xxx import yyy' works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.config import BanglaGSGConfig
from src.model.model import BanglaGSGModel
from src.data.collator import build_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Perplexity and Loss on .npy shards for all subsets")
    parser.add_argument("--model_dir", type=str, default="saved/model/default",
                        help="Path to the exported model directory (containing config.yaml and model.pt)")
    parser.add_argument("--data_dir", type=str, default="saved/data/eval",
                        help="Path to the eval data directory (containing bng, eng, translation subdirs)")
    parser.add_argument("--report_dir", type=str, default="saved/reports",
                        help="Path to save the YAML report")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of dataloader workers")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ── 1. Load Model ──────────────────────────────────────────────────────────
    config_path = os.path.join(args.model_dir, "config.yaml")
    model_path = os.path.join(args.model_dir, "model.pt")
    
    print(f"[Init] Loading config from {config_path}")
    model_config = BanglaGSGConfig.from_yaml(config_path)
    
    print(f"[Init] Building model on {device}...")
    model = BanglaGSGModel(model_config).to(device)
    
    print(f"[Init] Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # ── 2. Discover Datasets ───────────────────────────────────────────────────
    data_path = Path(args.data_dir)
    subsets = [d.name for d in data_path.iterdir() if d.is_dir()]
    if not subsets:
        print(f"[Error] No subdirectories found in {args.data_dir}")
        sys.exit(1)
        
    print(f"[Init] Found evaluation subsets: {subsets}")
    
    results = {}
    
    # ── 3. Evaluate Each Subset ────────────────────────────────────────────────
    with torch.no_grad():
        for subset in subsets:
            subset_dir = data_path / subset
            npy_files = list(subset_dir.rglob("*.npy"))
            if not npy_files:
                print(f"[Warning] No .npy files found in {subset_dir}. Skipping.")
                continue
                
            print(f"\nEvaluating subset: {subset} (found {len(npy_files)} files)")
            loader = build_dataloader(
                npy_dir=str(subset_dir),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True,
            )
            
            total_loss = 0.0
            total_ce = 0.0
            total_z = 0.0
            n_batches = 0
            
            for batch in tqdm(loader, desc=f"Eval {subset}"):
                input_ids = batch["input_ids"].to(device)
                targets = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                    
                    # Compute CE loss
                    V = logits.shape[-1]
                    ce_loss = F.cross_entropy(
                        logits.view(-1, V),
                        targets.view(-1),
                        ignore_index=0,  # Pad token
                    )
                    
                    # Compute Z-loss
                    z_loss = 1e-4 * torch.logsumexp(logits, dim=-1).pow(2).mean()
                    loss = ce_loss + z_loss
                    
                total_loss += loss.item()
                total_ce += ce_loss.item()
                total_z += z_loss.item()
                n_batches += 1
                
            if n_batches > 0:
                avg_loss = total_loss / n_batches
                avg_ce = total_ce / n_batches
                avg_z = total_z / n_batches
                perplexity = math.exp(min(avg_ce, 20.0))
                
                results[subset] = {
                    "total_loss": float(f"{avg_loss:.4f}"),
                    "ce_loss": float(f"{avg_ce:.4f}"),
                    "z_loss": float(f"{avg_z:.4f}"),
                    "perplexity": float(f"{perplexity:.4f}"),
                    "batches": n_batches
                }
                print(f"[{subset}] PPL: {perplexity:.2f} | Loss: {avg_loss:.4f}")
    # Compute overall average
    if results:
        overall_total_loss = sum(r["total_loss"] for r in results.values()) / len(results)
        overall_ce_loss = sum(r["ce_loss"] for r in results.values()) / len(results)
        overall_z_loss = sum(r["z_loss"] for r in results.values()) / len(results)
        overall_perplexity = sum(r["perplexity"] for r in results.values()) / len(results)
        overall_batches = sum(r["batches"] for r in results.values())
        
        results["overall"] = {
            "total_loss": float(f"{overall_total_loss:.4f}"),
            "ce_loss": float(f"{overall_ce_loss:.4f}"),
            "z_loss": float(f"{overall_z_loss:.4f}"),
            "perplexity": float(f"{overall_perplexity:.4f}"),
            "batches": overall_batches
        }
        print(f"\n[Overall] PPL: {overall_perplexity:.2f} | Loss: {overall_total_loss:.4f}")

    # ── 4. Save Report ─────────────────────────────────────────────────────────
    os.makedirs(args.report_dir, exist_ok=True)
    report_path = os.path.join(args.report_dir, "eval_results.yaml")
    
    with open(report_path, "w") as f:
        yaml.dump({"evaluations": results}, f, sort_keys=False)
        
    print(f"\n[Done] Evaluation report saved to {report_path}")

if __name__ == "__main__":
    main()
