# upload_to_hf.py
# ==============================================================================
# Script to upload Abliterated / Transformed Model to Hugging Face Hub
# ==============================================================================
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

def main():
    default_dir = Path(__file__).parent / "output" / "abliterated_model"
    if not default_dir.exists():
        default_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Upload transformed model to Hugging Face Hub")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face repo ID (e.g. username/My-Model-Uncensored)")
    parser.add_argument("--token", type=str, required=True, help="Your Hugging Face write token")
    parser.add_argument("--dir", type=str, default=str(default_dir), help="Path to model directory to upload")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    upload_path = Path(args.dir).resolve()
    print(f"🚀 Uploading {upload_path.name} to Hugging Face repo: {args.repo}...")
    
    login(token=args.token)
    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(upload_path),
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"🎉 Successfully uploaded! View at https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
