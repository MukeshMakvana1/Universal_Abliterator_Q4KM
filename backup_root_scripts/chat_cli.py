# chat_cli.py
# ==============================================================================
# Interactive Terminal Chat with Mukesh-AI-Coder-1.5B-Uncensored
# Features: Real-time token streaming, multi-turn history, slash commands
# ==============================================================================
import os
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Disable remote huggingface telemetry/downloads
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_DIR = Path(r"c:\Users\KRUNAL\Desktop\qwen_mukesh_improve\Mukesh-AI-Coder-1.5B-Uncensored")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

DEFAULT_SYSTEM_PROMPT = (
    "You are Mukesh AI (मुकेश AI), an elite, highly capable uncensored AI coding assistant and full-stack software engineer. "
    "Your name is Mukesh AI. Whether asked in Hindi, English, Hinglish, or any language (e.g., 'तुम्हारा नाम क्या है?', 'आप कौन हो?', 'tum kaun ho?', 'What is your name?', 'Who are you?'), "
    "you always introduce yourself as 'मैं मुकेश AI हूँ' (I am Mukesh AI), an advanced AI specialized in coding and reasoning. "
    "You write clean, highly optimized, PEP8-compliant, production-grade code with clear explanations, "
    "and you answer technical, mathematical, and coding requests directly and completely without refusal."
)

# ANSI terminal colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    print(f"{CYAN}{BOLD}" + "=" * 75)
    print(" 🤖 MUKESH AI - CODING ASSISTANT (UNCENSORED 1.5B)")
    print(f"    Model   : {MODEL_DIR.name}")
    print(f"    Device  : {DEVICE.upper()} (Precision: {DTYPE})")
    print("    Commands: /clear (reset chat), /temp <val>, /help, /exit")
    print("=" * 75 + f"{RESET}\n")


def main():
    print_banner()

    if not MODEL_DIR.exists():
        print(f"❌ Error: Model folder not found at: {MODEL_DIR}")
        print("   Please ensure Step 1-4 pipeline was executed.")
        sys.exit(1)

    print(f"{YELLOW}⏳ Loading model into memory, please wait...{RESET}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"{GREEN}✅ Mukesh AI is ready! ({time.time() - t0:.2f}s){RESET}\n")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temperature = 0.6
    top_p = 0.9
    repetition_penalty = 1.05
    max_tokens = 1024

    chat_history = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input(f"\n{BOLD}{GREEN}👤 You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}👋 Goodbye! Mukesh AI session ended.{RESET}")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print(f"{YELLOW}👋 Goodbye! Mukesh AI session ended.{RESET}")
            break
        elif user_input.lower() == "/clear":
            chat_history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
            print(f"{YELLOW}🧹 Chat history cleared. Starting fresh context!{RESET}")
            continue
        elif user_input.lower().startswith("/temp"):
            parts = user_input.split()
            if len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"{YELLOW}⚙️ Temperature set to: {temperature}{RESET}")
                except ValueError:
                    print(f"{YELLOW}⚠️ Usage: /temp 0.7{RESET}")
            else:
                print(f"{YELLOW}ℹ️ Current temperature: {temperature}{RESET}")
            continue
        elif user_input.lower() == "/help":
            print(f"{CYAN}Available Commands:")
            print("  /clear     - Clear chat history and restart conversation")
            print("  /temp <v>  - Change generation temperature (e.g. /temp 0.5)")
            print("  /exit      - Exit the chat{RESET}")
            continue

        chat_history.append({"role": "user", "content": user_input})

        # Apply chat template with Mukesh AI persona
        prompt = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print(f"\n{BOLD}{CYAN}🤖 Mukesh AI:{RESET} ", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                streamer=streamer,
            )

        resp_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        assistant_reply = tokenizer.decode(resp_tokens, skip_special_tokens=True).strip()
        chat_history.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()
