"""
Interleave shards from different sources sequentially and move them to the final training directory.
Takes shards serially from bangla, sangraha, english, and nmt, prefixes them, and moves them.

Usage:
  python utils/interleave_shards.py
"""

import os
import shutil

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    bangla_dir = os.path.join(project_root, "saved/data/pretokenized/bangla/train")
    sangraha_dir = os.path.join(project_root, "saved/data/pretokenized/sangraha/train")
    english_dir = os.path.join(project_root, "saved/data/pretokenized/english/train")
    nmt_dir = os.path.join(project_root, "saved/data/pretokenized/nmt/train")
    
    dest_dir = os.path.join(project_root, "saved/data/train")
    os.makedirs(dest_dir, exist_ok=True)
    
    def get_sorted_shards(d):
        if not os.path.isdir(d):
            return []
        return sorted([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f.endswith('.npy')])
        
    bangla_files = get_sorted_shards(bangla_dir)
    sangraha_files = get_sorted_shards(sangraha_dir)
    english_files = get_sorted_shards(english_dir)
    nmt_files = get_sorted_shards(nmt_dir)
    
    global_idx = 1
    
    for iteration in range(6):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Take 5 bangla shards
        for _ in range(5):
            if bangla_files:
                f = bangla_files.pop(0)
                move_file(bangla_dir, f, dest_dir, global_idx)
                global_idx += 1
            else:
                print("Warning: Not enough bangla shards.")
                
        # Take 1 sangraha shard
        if sangraha_files:
            f = sangraha_files.pop(0)
            move_file(sangraha_dir, f, dest_dir, global_idx)
            global_idx += 1
        else:
            print("Warning: Not enough sangraha shards.")
            
        # Take 1 english shard
        if english_files:
            f = english_files.pop(0)
            move_file(english_dir, f, dest_dir, global_idx)
            global_idx += 1
        else:
            print("Warning: Not enough english shards.")
            
        # Take 1 nmt shard
        if nmt_files:
            f = nmt_files.pop(0)
            move_file(nmt_dir, f, dest_dir, global_idx)
            global_idx += 1
        else:
            print("Warning: Not enough nmt shards.")

def move_file(src_dir, filename, dest_dir, idx):
    prefix = f"{idx:02d}_"
    new_filename = f"{prefix}{filename}"
    
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dest_dir, new_filename)
    
    shutil.move(src_path, dst_path)
    print(f"Moved: '{filename}' -> '{new_filename}'")

if __name__ == "__main__":
    main()
