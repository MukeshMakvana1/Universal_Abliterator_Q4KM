# upload_to_hf.py
# ==============================================================================
# Script to upload Mukesh-AI-Coder-1.5B-Uncensored to Hugging Face Hub
# ==============================================================================
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face repo ID (e.g. username/Mukesh-AI-Coder-1.5B-Uncensored)")
    parser.add_argument("--token", type=str, required=True, help="Your Hugging Face write token")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    model_dir = Path(__file__).parent
    print(f"🚀 Uploading {model_dir.name} to Hugging Face: {args.repo}...")
    
    login(token=args.token)
    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"🎉 Successfully uploaded to https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
