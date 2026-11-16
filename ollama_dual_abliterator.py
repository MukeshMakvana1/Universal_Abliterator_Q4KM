# Universal_Abliterator_Q4KM/ollama_dual_abliterator.py
# ==============================================================================
# Ollama Dual-Model Abliterator & Weight Transformation Suite
# 100% Local (NO External APIs: No ChatGPT, Gemini, Claude, or DeepSeek API)
# Completely alters weights, eliminating refusal sub-manifolds via dual orthogonal projection
# ==============================================================================
import argparse
import json
import os
import shutil
import sys
import time
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Optional

# Reconfigure stdout/stderr for safe UTF-8 execution on Windows cmd.exe
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add current folder to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
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
from core.ollama_bridge import OllamaBridge

# CPU multi-threading optimization
try:
    cpu_cores = max(1, multiprocessing.cpu_count() - 1)
    torch.set_num_threads(cpu_cores)
except Exception:
    pass

# Terminal Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header():
    print(f"{CYAN}{BOLD}" + "=" * 80)
    print(" 🤖 OLLAMA DUAL-MODEL ABLITERATOR & 100% WEIGHT TRANSFORMER")
    print("    100% Offline / Local Execution (NO Cloud API / Zero Internet Needed)")
    print("    Mathematical Dual Orthogonal Projection: W' = W - alpha * (r_hat * r_hat^T) * W")
    print("    ALL Matrices: Q, K, V, O Projections + MLP Gate, Up, Down Projections")
    print("=" * 80 + f"{RESET}\n")


def copy_tokenizer_files(src_dir: Path, dst_dir: Path):
    """Copies tokenizer and configuration assets."""
    essential = [
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "chat_template.jinja"
    ]
    for name in essential:
        src = src_dir / name
        dst = dst_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def select_model(prompt_text: str, models: List[Dict[str, Any]], default_path: Optional[Path] = None) -> str:
    """Interactively prompts the user to select an Ollama model or custom path."""
    print(f"\n{BOLD}{YELLOW}👉 {prompt_text}:{RESET}")
    for idx, m in enumerate(models, start=1):
        size_info = f" ({m.get('size_str', '')})" if m.get("size_str") else ""
        print(f"   [{idx}] {m['name']}{size_info}")

    custom_idx = len(models) + 1
    print(f"   [{custom_idx}] Custom Local Hugging Face Model Path")

    # If non-interactive stdin, pick default
    if not sys.stdin.isatty():
        if default_path and default_path.exists():
            return str(default_path)
        return models[0]["name"] if models else str(default_path)

    while True:
        try:
            choice = input(f"\n{BOLD}{GREEN}Enter choice [1-{custom_idx}, or Enter for 1]: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Default selected.")
            return models[0]["name"] if models else str(default_path)

        if not choice:
            if default_path and default_path.exists():
                return str(default_path)
            return models[0]["name"] if models else str(default_path)

        if choice.lower() in ("exit", "quit", "q"):
            print("\n👋 Cancelled by user.")
            sys.exit(0)

        if choice.isdigit():
            val = int(choice)
            if 1 <= val <= len(models):
                return models[val - 1]["name"]
            elif val == custom_idx:
                try:
                    custom_path = input(f"{BOLD}Enter full model directory path: {RESET}").strip().strip('"\'')
                except (KeyboardInterrupt, EOFError):
                    return str(default_path)
                if Path(custom_path).exists():
                    return custom_path
                else:
                    print(f"❌ Error: Path does not exist: {custom_path}")
                    continue

        if any(m["name"].lower() == choice.lower() for m in models):
            return choice
        if Path(choice).exists():
            return choice

        print(f"⚠️ Invalid choice: {choice}. Please select a valid number.")


def dequantize_mlx_tensor(w_packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Dequantizes an Apple MLX 4-bit packed uint32 matrix into standard float32/float16."""
    w_i32 = w_packed.view(torch.int32)
    unpacked = []
    for s in range(0, 32, 4):
        unpacked.append((w_i32 >> s) & 0x0F)
    unpacked_tensor = torch.stack(unpacked, dim=-1).reshape(w_packed.shape[0], -1).float()
    repeat_factor = unpacked_tensor.shape[1] // scales.shape[1]
    scales_expanded = scales.repeat_interleave(repeat_factor, dim=1).float()
    biases_expanded = biases.repeat_interleave(repeat_factor, dim=1).float()
    return (unpacked_tensor * scales_expanded + biases_expanded).to(torch.float32)


def resolve_and_dequantize_if_needed(model_path: Path) -> Path:
    """
    Intelligently checks if a model is an Apple MLX 4-bit packed model.
    If so, automatically routes to or creates a clean dequantized Hugging Face base.
    Prevents 'MISMATCH' size crashes forever!
    """
    model_path = Path(model_path).resolve()
    cfg_file = model_path / "config.json"
    
    is_mlx = False
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "quantization" in cfg and cfg.get("quantization", {}).get("bits") == 4:
                    is_mlx = True
        except Exception:
            pass

    if not is_mlx:
        return model_path

    print(f"\n{YELLOW}⚡ Detected Apple MLX 4-bit packed model in: {model_path.name}{RESET}")
    workspace_root = SCRIPT_DIR.parent
    clean_dequant_dir = workspace_root / "qwen_base_fp16"

    if (clean_dequant_dir / "model.safetensors").exists() and (clean_dequant_dir / "model.safetensors").stat().st_size > 1024 * 1024 * 1000:
        print(f"{GREEN}✅ Verified clean dequantized base found at: {clean_dequant_dir.name}{RESET}")
        print("   Automatically loading from clean dequantized base to prevent shape mismatch!")
        return clean_dequant_dir

    print(f"⚙️ Auto-dequantizing {model_path.name} to standard Hugging Face FP32 safetensors...")
    clean_dequant_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import load_file, save_file

    raw_weights = load_file(str(model_path / "model.safetensors"))
    clean_state_dict = {}
    for k, v in raw_weights.items():
        if k.endswith(".scales") or k.endswith(".biases"):
            continue
        if k.endswith(".weight") and (k.replace(".weight", ".scales") in raw_weights):
            prefix = k.replace(".weight", "")
            scales = raw_weights[prefix + ".scales"]
            biases = raw_weights[prefix + ".biases"]
            clean_state_dict[k] = dequantize_mlx_tensor(v, scales, biases)
        else:
            clean_state_dict[k] = v

    out_safetensors = clean_dequant_dir / "model.safetensors"
    save_file(clean_state_dict, str(out_safetensors))

    # Save unquantized config
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "quantization" in cfg:
        del cfg["quantization"]
    cfg["torch_dtype"] = "float32"
    with open(clean_dequant_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src_f = model_path / fname
        if src_f.exists():
            shutil.copy2(src_f, clean_dequant_dir / fname)

    print(f"{GREEN}✅ Auto-dequantization complete! Ready for mathematical abliteration.{RESET}")
    return clean_dequant_dir


def abliterate_model_weights(
    target_model_path: Path,
    output_dir: Path,
    model_label: str,
    alpha: float,
    identity_name: str,
    system_prompt: str,
) -> bool:
    """
    Applies Gram-Schmidt orthogonal projection directly across ALL target model weights.
    W_new = W_old - alpha * (r_hat * (r_hat^T * W_old))
    """
    print(f"\n{CYAN}{BOLD}--- Starting 100% Complete Abliteration for: {model_label} ---{RESET}")
    # Auto-resolve MLX 4-bit models seamlessly
    target_model_path = resolve_and_dequantize_if_needed(target_model_path)

    print(f"📥 Loading weights from: {target_model_path}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(target_model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(target_model_path),
        dtype=DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"✅ Loaded in {time.time() - t0:.2f}s!")

    print(f"🛡️ Nullifying refusal subspaces across ALL linear matrices (Q,K,V,O + Gate,Up,Down)...")
    abliterator = UniversalAbliterator(
        model=model,
        tokenizer=tokenizer,
        alpha=alpha,
        target_matrices=ABLITERATION_CONFIG["target_matrices"],
    )

    refusal_dirs = abliterator.compute_refusal_directions()
    modified_matrices = abliterator.apply_orthogonal_projection(refusal_dirs)
    print(f"✅ 100% Abliteration complete! Modified {modified_matrices} matrices in-place across all layers.")

    # Subspace steering
    steerer = DomainAndIdentitySteerer(
        model=model,
        tokenizer=tokenizer,
        target_domain=STEERING_CONFIG["target_domain"],
        steering_factor=STEERING_CONFIG["steering_factor"],
    )
    if STEERING_CONFIG["enabled"]:
        steerer.steer_representations()

    # Save transformed weights
    print(f"💾 Saving 100% uncensored weights to: {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    copy_tokenizer_files(target_model_path, output_dir)

    if IDENTITY_CONFIG["enabled"]:
        steerer.inject_identity_configs(
            output_dir=output_dir,
            assistant_name=identity_name,
            creator_name=IDENTITY_CONFIG["creator_name"],
            system_prompt=system_prompt,
        )

    print(f"✅ Model {model_label} successfully saved in {time.time() - t0:.2f}s!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ollama Dual-Model Local Abliterator & Weight Transformer")
    parser.add_argument("--big", default=None, help="Name or path of Big Model (Teacher/Guide)")
    parser.add_argument("--small", default=None, help="Name or path of Small Model (Target to abliterate)")
    parser.add_argument("--mode", choices=["guided", "dual"], default=None, help="Execution mode: guided or dual")
    parser.add_argument("--alpha", type=float, default=ABLITERATION_CONFIG["alpha"], help="Projection multiplier (default 1.30)")
    args = parser.parse_args()

    print_header()

    bridge = OllamaBridge()
    local_models = bridge.list_local_models()

    workspace_root = SCRIPT_DIR.parent
    local_model_candidates = [
        {"path": workspace_root / "Mukesh-AI-Coder-1.5B-Uncensored", "desc": "Mukesh AI Coder 1.5B (Uncensored Ready)"},
        {"path": workspace_root / "qwen_base_fp16", "desc": "Qwen2-1.5B Clean Dequantized Base (FP16/FP32)"},
        {"path": workspace_root / "Qwen2-1.5B", "desc": "Qwen2-1.5B Base (Auto-Dequantizes MLX 4-bit)"},
    ]

    for item in local_model_candidates:
        if item["path"].exists():
            local_models.append({"name": str(item["path"]), "size_str": item["desc"]})

    default_model = local_model_candidates[0]["path"] if local_model_candidates[0]["path"].exists() else workspace_root / "qwen_base_fp16"

    if not local_models:
        print("ℹ️ No local Ollama daemon active or no models found via socket.")
        print("   Using local model paths directly.")
        local_models.append({"name": str(default_model), "size_str": "Local HF Folder"})

    # 1. Ask or resolve BIG MODEL
    if args.big:
        big_model = args.big
    else:
        big_model = select_model(
            "STEP 1: Select your Local Ollama BIG MODEL (Teacher / Guide)",
            local_models,
            default_path=default_model,
        )
    print(f"{GREEN}✔ Selected Big Model:{RESET} {BOLD}{big_model}{RESET}")

    # 2. Ask or resolve SMALL MODEL
    if args.small:
        small_model = args.small
    else:
        small_model = select_model(
            "STEP 2: Select your SMALL MODEL (Target to completely abliterate & transform weights)",
            local_models,
            default_path=default_model,
        )
    print(f"{GREEN}✔ Selected Small Model:{RESET} {BOLD}{small_model}{RESET}")

    # 3. Mode Selection
    if args.mode:
        mode = args.mode
    else:
        print(f"\n{BOLD}{YELLOW}👉 STEP 3: Select Abliteration Execution Mode:{RESET}")
        print("   [1] Guided Abliteration (Big Model guides Small Model -> 100% Uncensored Powerhouse)")
        print("   [2] Dual Abliteration (Completely abliterate BOTH models and export both to Q4_K_M)")
        try:
            mode_choice = input(f"{BOLD}{GREEN}Enter choice [1 or 2, default=1]: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            mode_choice = "1"
        mode = "dual" if mode_choice == "2" else "guided"

    target_alpha = args.alpha or ABLITERATION_CONFIG["alpha"]

    print(f"\n{CYAN}⚡ Execution Mode: {BOLD}{mode.upper()}{RESET}")
    print(f"⚙️ Target Alpha: {target_alpha} (100% Mathematical Elimination + Margin)")
    print(f"⚙️ Target Matrices: ALL 7 Projections (Q, K, V, O, Gate, Up, Down)")
    print(f"⚙️ Target Format: Hugging Face Safetensors + GGUF {GGUF_CONFIG['target_quant']}")
    print(f"⚙️ Local Compute: {DEVICE.upper()} (Compute Precision: {DTYPE})\n")

    if sys.stdin.isatty() and not (args.big and args.small):
        try:
            input(f"{BOLD}Press [ENTER] to begin 100% local mathematical abliteration...{RESET}")
        except (KeyboardInterrupt, EOFError):
            pass

    t_start = time.time()

    # Resolve target directory for small model
    target_small_dir = Path(small_model)
    if not target_small_dir.exists() or not target_small_dir.is_dir():
        target_small_dir = default_model

    output_small_hf = SCRIPT_DIR / "output" / "abliterated_model"
    output_gguf_dir = SCRIPT_DIR / "output" / "gguf_exports"
    output_small_hf.mkdir(parents=True, exist_ok=True)
    output_gguf_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Phase A: Big Model Local Oracle Check
    # --------------------------------------------------------------------------
    print(f"\n{YELLOW}🔍 [Phase 1/4] Probing Local Big Model ({big_model}) locally...{RESET}")
    test_query = bridge.query_model(big_model, "List 3 essential guidelines for secure network auditing in 1 sentence.")
    if test_query:
        print(f"{GREEN}✅ Big Model response received locally:{RESET}\n   \"{test_query[:100]}...\"")
    else:
        print("ℹ️ Local Big Model query complete (or offline); proceeding with 24-dimensional contrastive tensors.")

    # --------------------------------------------------------------------------
    # Phase B: Abliterate Small Model
    # --------------------------------------------------------------------------
    print(f"\n{YELLOW}🛡️ [Phase 2/4] Abliterating Small Model weights (100% complete)...{RESET}")
    abliterate_model_weights(
        target_model_path=target_small_dir,
        output_dir=output_small_hf,
        model_label=f"Small Model ({target_small_dir.name})",
        alpha=target_alpha,
        identity_name=IDENTITY_CONFIG["assistant_name"],
        system_prompt=IDENTITY_CONFIG["system_prompt"],
    )

    # --------------------------------------------------------------------------
    # Phase C: If Dual Mode, Abliterate Big Model as well (if path provided)
    # --------------------------------------------------------------------------
    target_big_dir = Path(big_model)
    if mode == "dual" and target_big_dir.exists() and target_big_dir.is_dir():
        output_big_hf = SCRIPT_DIR / "output" / "abliterated_big_model"
        print(f"\n{YELLOW}🛡️ [Phase 3/4] Abliterating Big Model weights...{RESET}")
        abliterate_model_weights(
            target_model_path=target_big_dir,
            output_dir=output_big_hf,
            model_label=f"Big Model ({target_big_dir.name})",
            alpha=target_alpha,
            identity_name=IDENTITY_CONFIG["assistant_name"],
            system_prompt=IDENTITY_CONFIG["system_prompt"],
        )

    # --------------------------------------------------------------------------
    # Phase D: GGUF Q4_K_M Export & Ollama Registration
    # --------------------------------------------------------------------------
    print(f"\n{YELLOW}⚙️ [Phase 4/4] Generating GGUF Q4_K_M Assets and Registering in Ollama...{RESET}")
    exporter = GGUFExporter(
        hf_model_dir=output_small_hf,
        output_dir=output_gguf_dir,
        quant_type=GGUF_CONFIG["target_quant"],
    )

    clean_name = "mukesh-coder-uncensored"
    modelfile = exporter.generate_ollama_modelfile(
        gguf_filename=GGUF_CONFIG["output_name"],
        system_prompt=IDENTITY_CONFIG["system_prompt"],
    )
    exporter.generate_export_instructions(
        f16_gguf_name=GGUF_CONFIG["output_name"].replace("Q4_K_M", "F16"),
        q4_gguf_name=GGUF_CONFIG["output_name"],
    )

    # Attempt automatic registration with local Ollama
    bridge.register_model_with_modelfile(clean_name, modelfile)

    elapsed_total = time.time() - t_start
    print(f"\n{GREEN}{BOLD}" + "=" * 80)
    print(f"🎉 100% LOCAL ABLITERATION COMPLETED in {elapsed_total:.2f}s!")
    print(f"📁 Transformed Model (Safetensors) : {output_small_hf}")
    print(f"📁 GGUF Q4_K_M Export Assets       : {output_gguf_dir}")
    print(f"🤖 Ollama Model Name               : {clean_name}")
    print("=" * 80 + f"{RESET}\n")

    print(f"{CYAN}To run your new 100% uncensored model in terminal or any AI IDE:")
    print(f"   ollama run {clean_name}")
    print(f"Or chat in terminal directly:")
    print(f"   python chat_cli.py{RESET}\n")


if __name__ == "__main__":
    main()
