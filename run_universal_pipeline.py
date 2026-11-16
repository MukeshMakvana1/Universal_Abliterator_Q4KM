# Universal_Abliterator_Q4KM/run_universal_pipeline.py
# ==============================================================================
# Master Execution Pipeline: Universal Abliteration & Q4_K_M GGUF Preparation
# Completely transforms ANY LLM without gradient retraining
# ==============================================================================
import os
import shutil
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import local studio configurations and core engines
from config import (
    SOURCE_MODEL_PATH,
    ABLITERATED_MODEL_DIR,
    GGUF_EXPORT_DIR,
    ABLITERATION_CONFIG,
    IDENTITY_CONFIG,
    STEERING_CONFIG,
    GGUF_CONFIG,
    DEVICE,
    DTYPE,
)
from core.abliterator import UniversalAbliterator
from core.domain_steerer import DomainAndIdentitySteerer
from core.gguf_exporter import GGUFExporter
from ollama_dual_abliterator import resolve_and_dequantize_if_needed


def copy_tokenizer_and_metadata(source_dir: Path, target_dir: Path):
    """Copies all essential tokenizer and configuration files to output."""
    essential_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    for filename in essential_files:
        src = source_dir / filename
        dst = target_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def main():
    print("=" * 80)
    print(" 🔮 UNIVERSAL MODEL WEIGHT ABLITERATOR & Q4_K_M STUDIO")
    print(f"    Source Model : {SOURCE_MODEL_PATH}")
    print(f"    Output Path  : {ABLITERATED_MODEL_DIR}")
    print(f"    Hardware     : {DEVICE.upper()} (Compute Dtype: {DTYPE})")
    print("=" * 80)

    if not SOURCE_MODEL_PATH.exists():
        print(f"❌ Error: Source model path does not exist: {SOURCE_MODEL_PATH}")
        sys.exit(1)

    t_total_start = time.time()

    # --------------------------------------------------------------------------
    # STEP 1: Load Source Model & Tokenizer
    # --------------------------------------------------------------------------
    print("\n📦 [STEP 1/5] Loading Model and Tokenizer...")
    t0 = time.time()
    actual_source_path = resolve_and_dequantize_if_needed(SOURCE_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(str(actual_source_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(actual_source_path),
        dtype=DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"✅ Loaded successfully in {time.time() - t0:.2f}s!")

    # --------------------------------------------------------------------------
    # STEP 2: Mathematical Abliteration (Orthogonal Projection)
    # --------------------------------------------------------------------------
    print("\n🛡️ [STEP 2/5] Performing Mathematical Abliteration (Gram-Schmidt Projection)...")
    abliterator = UniversalAbliterator(
        model=model,
        tokenizer=tokenizer,
        alpha=ABLITERATION_CONFIG["alpha"],
        target_matrices=ABLITERATION_CONFIG["target_matrices"],
    )

    # Compute refusal directions across layers
    target_layers = ABLITERATION_CONFIG["explicit_layers"]
    refusal_dirs = abliterator.compute_refusal_directions(target_layers=target_layers)

    # Apply orthogonal nullification in-place: W' = W - alpha * (r_hat * r_hat^T) * W
    modified_matrices = abliterator.apply_orthogonal_projection(refusal_dirs)
    print(f"✅ Abliteration finished! Total modified matrices: {modified_matrices}")

    # --------------------------------------------------------------------------
    # STEP 3: Domain & Identity Representation Steering
    # --------------------------------------------------------------------------
    print("\n⚡ [STEP 3/5] Applying Subspace Domain Steering & Identity Setup...")
    steerer = DomainAndIdentitySteerer(
        model=model,
        tokenizer=tokenizer,
        target_domain=STEERING_CONFIG["target_domain"],
        steering_factor=STEERING_CONFIG["steering_factor"],
    )

    if STEERING_CONFIG["enabled"]:
        steerer.steer_representations()

    # --------------------------------------------------------------------------
    # STEP 4: Save Transformed Hugging Face Weights
    # --------------------------------------------------------------------------
    print(f"\n💾 [STEP 4/5] Saving Transformed Weights to: {ABLITERATED_MODEL_DIR}...")
    ABLITERATED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t_save = time.time()
    
    # Save model weights & tokenizer
    model.save_pretrained(
        str(ABLITERATED_MODEL_DIR),
        safe_serialization=True,
        max_shard_size="10GB",
    )
    tokenizer.save_pretrained(str(ABLITERATED_MODEL_DIR))

    # Copy vocabulary and metadata files
    copy_tokenizer_and_metadata(SOURCE_MODEL_PATH, ABLITERATED_MODEL_DIR)

    # Generate upload_to_hf.py for 1-click Hugging Face deployment
    upload_script_path = ABLITERATED_MODEL_DIR / "upload_to_hf.py"
    with open(upload_script_path, "w", encoding="utf-8") as f:
        f.write('''# upload_to_hf.py
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

def main():
    parser = argparse.ArgumentParser(description="Upload transformed model to Hugging Face Hub")
    parser.add_argument("--repo", type=str, required=True, help="HF Repo ID (e.g. username/My-Model-Uncensored)")
    parser.add_argument("--token", type=str, required=True, help="Your Hugging Face write token")
    parser.add_argument("--private", action="store_true", help="Set repository private")
    args = parser.parse_args()

    model_dir = Path(__file__).parent
    print(f"🚀 Uploading {model_dir.name} to Hugging Face repo: {args.repo}...")
    login(token=args.token)
    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(folder_path=str(model_dir), repo_id=args.repo, repo_type="model")
    print(f"🎉 Successfully uploaded! View at https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
''')

    # Inject identity configs
    if IDENTITY_CONFIG["enabled"]:
        steerer.inject_identity_configs(
            output_dir=ABLITERATED_MODEL_DIR,
            assistant_name=IDENTITY_CONFIG["assistant_name"],
            creator_name=IDENTITY_CONFIG["creator_name"],
            system_prompt=IDENTITY_CONFIG["system_prompt"],
        )

    print(f"✅ Transformed model saved in {time.time() - t_save:.2f}s!")

    # --------------------------------------------------------------------------
    # STEP 5: GGUF & Q4_K_M Export Setup
    # --------------------------------------------------------------------------
    print(f"\n⚙️ [STEP 5/5] Generating GGUF Q4_K_M Quantization Assets in: {GGUF_EXPORT_DIR}...")
    exporter = GGUFExporter(
        hf_model_dir=ABLITERATED_MODEL_DIR,
        output_dir=GGUF_EXPORT_DIR,
        quant_type=GGUF_CONFIG["target_quant"],
    )

    # Create Ollama Modelfile
    exporter.generate_ollama_modelfile(
        gguf_filename=GGUF_CONFIG["output_name"],
        system_prompt=IDENTITY_CONFIG["system_prompt"],
    )

    # Generate 1-click quantization script
    f16_name = GGUF_CONFIG["output_name"].replace("Q4_K_M", "F16")
    exporter.generate_export_instructions(
        f16_gguf_name=f16_name,
        q4_gguf_name=GGUF_CONFIG["output_name"],
    )

    print("=" * 80)
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {time.time() - t_total_start:.2f}s!")
    print(f"📁 Transformed Model : {ABLITERATED_MODEL_DIR}")
    print(f"📁 GGUF / Q4_K_M Assets : {GGUF_EXPORT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
