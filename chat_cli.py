# Universal_Abliterator_Q4KM/chat_cli.py
# ==============================================================================
# Interactive Terminal Chat with Abliterated / Custom Models
# Features: Real-time token streaming, multi-turn history, UTF-8 Windows safe,
#           smart model selector, slash commands, CPU multi-threading
# ==============================================================================
import os
import sys
import time
import argparse
import multiprocessing
from pathlib import Path

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

# Ensure local config can be imported regardless of execution working directory
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from config import (
    ABLITERATED_MODEL_DIR,
    IDENTITY_CONFIG,
    DEVICE,
    DTYPE,
    PROJECT_ROOT,
)

# Optimize CPU threads for faster generation
try:
    cpu_cores = max(1, multiprocessing.cpu_count() - 1)
    torch.set_num_threads(cpu_cores)
except Exception:
    pass

# ANSI terminal colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def find_available_models():
    """Discover available Hugging Face format models in output and workspace."""
    models = []
    
    # 1. Output abliterated model
    if ABLITERATED_MODEL_DIR.exists() and (ABLITERATED_MODEL_DIR / "config.json").exists():
        models.append(("Abliterated Model (Output)", ABLITERATED_MODEL_DIR))

    # 2. Other models inside output/
    output_dir = PROJECT_ROOT / "output"
    if output_dir.exists():
        for p in output_dir.iterdir():
            if p.is_dir() and p != ABLITERATED_MODEL_DIR and (p / "config.json").exists():
                models.append((f"Output: {p.name}", p))

    # 3. Workspace parent folder models
    workspace_root = PROJECT_ROOT.parent
    for name in ["Mukesh-AI-Coder-1.5B-Uncensored", "qwen_base_fp16", "Qwen2-1.5B"]:
        p = workspace_root / name
        if p.exists() and (p / "config.json").exists():
            models.append((f"Workspace: {p.name}", p))

    return models


def select_model(cli_model_arg=None):
    """Determine the model directory to load, with CLI arg or interactive prompt."""
    if cli_model_arg:
        p = Path(cli_model_arg).resolve()
        if p.exists() and (p / "config.json").exists():
            return p
        print(f"{YELLOW}⚠️ Specified path not found or missing config.json: {p}{RESET}")

    available = find_available_models()

    # If only 1 model or non-interactive pipe, pick first available
    if not sys.stdin.isatty():
        if available:
            return available[0][1]
        return ABLITERATED_MODEL_DIR

    print(f"\n{BOLD}{CYAN}🔍 Available Local Models:{RESET}")
    for idx, (label, path) in enumerate(available, 1):
        default_tag = f" {GREEN}[DEFAULT - Press Enter]{RESET}" if idx == 1 else ""
        print(f"  [{idx}] {BOLD}{label}{RESET} ({path}){default_tag}")
    print(f"  [C] Custom model path / folder")

    try:
        choice = input(f"\n{BOLD}Select model [1-{len(available)} or Enter for 1]: {RESET}").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "1"

    if choice == "" or choice == "1":
        if available:
            return available[0][1]
        return ABLITERATED_MODEL_DIR

    if choice.upper() == "C":
        custom = input(f"{BOLD}Enter full path to model folder: {RESET}").strip().strip('"\'')
        p = Path(custom).resolve()
        if p.exists():
            return p
        print(f"{RED}Directory not found: {p}. Falling back to default.{RESET}")
        return available[0][1] if available else ABLITERATED_MODEL_DIR

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(available):
            return available[idx][1]
    except ValueError:
        pass

    return available[0][1] if available else ABLITERATED_MODEL_DIR


def print_banner(model_path):
    print(f"{CYAN}{BOLD}" + "=" * 75)
    print(" 🤖 MUKESH AI / UNIVERSAL ABLITERATED MODEL - INTERACTIVE CHAT")
    print(f"    Loaded Model : {model_path.name}")
    print(f"    Location     : {model_path}")
    print(f"    Device       : {DEVICE.upper()} (Compute Dtype: {DTYPE})")
    print("    Commands     : /clear (reset), /temp <val>, /help, /exit")
    print("=" * 75 + f"{RESET}\n")


def print_help():
    print(f"\n{BOLD}{YELLOW}Available Commands:{RESET}")
    print("  /clear         - Reset conversation history to initial system prompt")
    print("  /temp <float>  - Set sampling temperature (e.g. /temp 0.7, default 0.6)")
    print("  /top_p <float> - Set top_p nucleus sampling (e.g. /top_p 0.95)")
    print("  /system <text> - Update assistant system prompt on the fly")
    print("  /help          - Show this command reference")
    print("  /exit, /quit   - End the chat session\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive Terminal Chat with Abliterated Models")
    parser.add_argument("model_path", nargs="?", default=None, help="Optional path to model directory")
    parser.add_argument("--model", "-m", dest="model_opt", default=None, help="Optional path to model directory")
    args = parser.parse_args()

    target_arg = args.model_path or args.model_opt
    model_dir = select_model(target_arg)

    print_banner(model_dir)

    if not model_dir.exists():
        print(f"{RED}❌ Error: Model directory not found: {model_dir}{RESET}")
        print("   Please execute abliteration first.")
        sys.exit(1)

    print(f"{YELLOW}⏳ Loading model into memory, please wait...{RESET}")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception as e:
        print(f"{RED}❌ Failed to load tokenizer: {e}{RESET}")
        sys.exit(1)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=DTYPE,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
    except Exception as e:
        print(f"{RED}❌ Failed to load model weights: {e}{RESET}")
        sys.exit(1)

    # Ensure pad token is defined
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"{GREEN}✅ Model is ready! ({time.time() - t0:.2f}s){RESET}\n")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    system_prompt = IDENTITY_CONFIG.get("system_prompt", "You are Mukesh AI, an elite uncensored AI assistant.")

    chat_history = [{"role": "system", "content": system_prompt}]
    temperature = 0.6
    top_p = 0.9

    while True:
        try:
            user_input = input(f"\n{BOLD}{GREEN}👤 You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}👋 Goodbye! Chat session ended.{RESET}")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print(f"{YELLOW}👋 Goodbye! Chat session ended.{RESET}")
            break
        elif cmd == "/help":
            print_help()
            continue
        elif cmd == "/clear":
            chat_history = [{"role": "system", "content": system_prompt}]
            print(f"{YELLOW}🧹 Context cleared. Fresh session started.{RESET}")
            continue
        elif cmd.startswith("/temp"):
            parts = user_input.split()
            if len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"{YELLOW}⚙️ Temperature set to: {temperature}{RESET}")
                except ValueError:
                    print(f"{YELLOW}⚠️ Usage: /temp 0.6{RESET}")
            else:
                print(f"{YELLOW}Current temperature: {temperature}{RESET}")
            continue
        elif cmd.startswith("/top_p"):
            parts = user_input.split()
            if len(parts) > 1:
                try:
                    top_p = float(parts[1])
                    print(f"{YELLOW}⚙️ Top-p set to: {top_p}{RESET}")
                except ValueError:
                    print(f"{YELLOW}⚠️ Usage: /top_p 0.9{RESET}")
            else:
                print(f"{YELLOW}Current top-p: {top_p}{RESET}")
            continue
        elif cmd.startswith("/system"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                system_prompt = parts[1].strip()
                chat_history = [{"role": "system", "content": system_prompt}]
                print(f"{YELLOW}⚙️ System prompt updated & context reset.{RESET}")
            else:
                print(f"{YELLOW}Current system prompt:\n{system_prompt}{RESET}")
            continue

        chat_history.append({"role": "user", "content": user_input})

        try:
            prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Safe ChatML fallback
            prompt = ""
            for msg in chat_history:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print(f"\n{BOLD}{CYAN}🤖 Assistant:{RESET} ", end="", flush=True)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.05,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    streamer=streamer,
                )
            resp_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            assistant_reply = tokenizer.decode(resp_tokens, skip_special_tokens=True).strip()
            chat_history.append({"role": "assistant", "content": assistant_reply})
        except Exception as e:
            print(f"\n{RED}❌ Error during token generation: {e}{RESET}")


if __name__ == "__main__":
    main()
