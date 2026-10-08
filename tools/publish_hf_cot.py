#!/usr/bin/env python3
"""
Publishes pyintel/embedded-hardware-cot to Hugging Face Hub.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "pyintel/embedded-hardware-cot"

def main():
    if not TOKEN:
        print("❌ Error: Missing HF_TOKEN environment variable.")
        sys.exit(1)
        
    parquet_path = Path("embedded_hardware_cot.parquet")
    readme_path = Path("README.md")
    
    if not parquet_path.exists():
        print(f"❌ Error: {parquet_path} does not exist.")
        sys.exit(1)
        
    api = HfApi(token=TOKEN)
    
    print(f"📡 Ensuring repository {REPO_ID} exists on Hugging Face...")
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        token=TOKEN
    )
    
    print(f"⬆️ Uploading {parquet_path} ({parquet_path.stat().st_size / (1024*1024):.2f} MB)...")
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo="embedded_hardware_cot.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Release: PyIntel Embedded Hardware CoT dataset (Parquet)"
    )
    
    if readme_path.exists():
        print(f"⬆️ Uploading {readme_path}...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Update Dataset Card for pyintel/embedded-hardware-cot"
        )
        
    print(f"🎉 SUCCESS! Dataset is live at: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()
