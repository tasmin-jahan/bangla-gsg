import subprocess
import time
import os
import signal
from pathlib import Path

def test_interrupt():
    print("Starting training script in background...")
    proc = subprocess.Popen([".venv/bin/python", "src/train.py"], cwd="/home/farhan/my-projects/bangla-gsg")
    
    # Wait for TileLang to compile and at least one step to process
    print("Waiting 40 seconds for model compilation and initial steps...")
    time.sleep(40)
    
    print("\n=== Sending SIGINT (Ctrl+C) to Trainer ===")
    proc.send_signal(signal.SIGINT)
    
    # Wait for the process to clean up and exit
    proc.wait(timeout=30)
    
    # Check if a checkpoint was saved in the correct directory
    ckpt_dir = Path("/home/farhan/my-projects/bangla-gsg/saved/checkpoints")
    ckpts = list(ckpt_dir.glob("*.pt"))
    if ckpts:
        print(f"\n✅ SUCCESS! Found {len(ckpts)} checkpoints in {ckpt_dir}. Graceful Ctrl+C save worked perfectly!")
        for ckpt in ckpts:
            print(f"  - {ckpt.name}")
    else:
        print("\n❌ FAILED! No checkpoints were saved after SIGINT.")

if __name__ == "__main__":
    test_interrupt()
