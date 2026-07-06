import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.config import BanglaGSGConfig
from src.model.model import BanglaGSGModel

def run_smoke_test():
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot test VRAM.")
        return

    device = "cuda"
    config = BanglaGSGConfig.from_yaml("configs/banglagsg_12l.yaml")
    
    print("Building model...")
    model = BanglaGSGModel(config).to(device, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    seq_len = config.seq_len # 2048

    for bs in [4, 5, 6, 7]:
        print(f"\n--- Testing batch_size={bs} ---")
        input_ids = torch.randint(0, config.vocab_size, (bs, seq_len), device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            for i in range(2):
                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                    loss = logits.sum()
                loss.backward()
                optimizer.step()
                
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"✅ PASSED! Peak VRAM: {peak_mem:.2f} GB")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("❌ OOM Error caught! VRAM exceeded.")
                break
            else:
                print(f"❌ Error: {e}")
                break

if __name__ == "__main__":
    run_smoke_test()
