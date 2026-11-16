# 05_chat_cli.py
# ==============================================================================
# Step 5: Interactive Terminal Chat with Mukesh AI Uncensored Coder
# ==============================================================================
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from config import (
    FINAL_MODEL_DIR,
    DEFAULT_SYSTEM_PROMPT,
    GENERATION_CONFIG,
    DTYPE,
    DEVICE,
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def chat():
    print("=" * 70)
    print("🤖 मुकेश AI (Mukesh AI Coder Uncensored) - लाइव चैट")
    print(f"   मॉडल: {FINAL_MODEL_DIR.name}")
    print(f"   डिवाइस: {DEVICE.upper()} (डेटा-टाइप: {DTYPE})")
    print("   'exit' या 'quit' लिखकर बाहर निकलें।")
    print("=" * 70 + "\n")

    if not FINAL_MODEL_DIR.exists():
        print(f"❌ मॉडल फोल्डर नहीं मिला: {FINAL_MODEL_DIR}")
        print("   पहले चलाएँ: python 04_package_huggingface.py")
        sys.exit(1)

    print("📥 मॉडल लोड हो रहा है, कृपया प्रतीक्षा करें...")
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(FINAL_MODEL_DIR),
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("✅ मुकेश AI तैयार है!\n")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    chat_history = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input("\n👤 आप (You): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 अलविदा!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("👋 अलविदा!")
            break

        chat_history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("\n🤖 मुकेश AI (Mukesh AI): ", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"],
                repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
                do_sample=True,
                streamer=streamer,
            )

        resp_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(resp_tokens, skip_special_tokens=True)
        chat_history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    chat()
