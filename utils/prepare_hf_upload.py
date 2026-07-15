import os
import shutil
from pathlib import Path

def prepare_upload_folder(staging_dir="hf_upload_staging"):
    staging = Path(staging_dir)
    
    # 1. Clear out old staging directory if it exists
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    
    print(f"Creating Hugging Face staging directory at: {staging}/")
    
    # 2. Copy Model Weights and Config
    model_dir = Path("saved/model/default")
    if (model_dir / "config.json").exists():
        shutil.copy(model_dir / "config.json", staging / "config.json")
        print(" -> Copied config.json")
    else:
        print(" -> WARNING: config.json not found! Run convert_config_to_json.py first.")
        
    if (model_dir / "model.pt").exists():
        shutil.copy(model_dir / "model.pt", staging / "model.pt")
        print(" -> Copied model.pt")
        
    # 3. Copy Custom Architecture & Tokenizer Wrappers
    hf_int_dir = Path("src/hf_integration")
    for py_file in hf_int_dir.glob("*.py"):
        shutil.copy(py_file, staging / py_file.name)
        print(f" -> Copied {py_file.name}")
        
    # Copy the specific Hugging Face README
    if (hf_int_dir / "README.md").exists():
        shutil.copy(hf_int_dir / "README.md", staging / "README.md")
        print(" -> Copied Model Card (README.md)")
        
    # 4. Copy the src/model/ directory to resolve relative imports (Option B)
    model_src = Path("src/model")
    model_dst = staging / "model"
    shutil.copytree(model_src, model_dst, dirs_exist_ok=True)
    # Remove pycache if it was copied
    pycache = model_dst / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)
    print(" -> Copied src/model/ directory")
    
    # 5. Provide exact instructions to the user
    print("\n" + "="*50)
    print("✅ Staging directory ready!")
    print("="*50)
    print("To upload this folder to your repository, ensure you are logged into Hugging Face")
    print("by running 'hf auth login' in your terminal.")
    print("\nThen, run the following command to upload all files to your repo:")
    print(f"hf upload tasmin-jahan/bangla-gsg {staging_dir} .")

if __name__ == "__main__":
    prepare_upload_folder()
