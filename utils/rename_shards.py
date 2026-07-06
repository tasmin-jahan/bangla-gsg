"""
Rename all files in a specified directory by adding a prefix.
Paths can be absolute or relative to the project root.

Usage:
  python utils/rename_shards.py --input <path> --prefix <prefix>
  
Example:
  python utils/rename_shards.py --input saved/data/pretokenized/bangla/train --prefix bng_
"""

import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Rename all files in a directory by adding a prefix.")
    parser.add_argument("--input", type=str, required=True, help="Directory containing the files to rename")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix to add to the filenames")
    
    args = parser.parse_args()
    
    # Resolve the path relative to the bangla-gsg directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    location = os.path.join(project_root, args.input) if not os.path.isabs(args.input) else args.input
    prefix = args.prefix
    
    if not os.path.isdir(location):
        print(f"Error: '{location}' is not a valid directory.")
        return
        
    count = 0
    for filename in sorted(os.listdir(location)):
        if filename.startswith(prefix):
            print(f"Skipping '{filename}' (already has prefix)")
            continue
            
        old_path = os.path.join(location, filename)
        if os.path.isfile(old_path):
            new_name = f"{prefix}{filename}"
            new_path = os.path.join(location, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: '{filename}' -> '{new_name}'")
            count += 1
            
    print(f"Successfully renamed {count} files.")

if __name__ == "__main__":
    main()
