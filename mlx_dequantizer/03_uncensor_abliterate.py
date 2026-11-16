# 03_uncensor_abliterate.py
# ==============================================================================
# Step 3: Mathematical Abliteration (Orthogonal Refusal Direction Removal)
# Permanently removes refusal & censorship from model weights without training.
# ==============================================================================
import os
import sys
import json
import shutil
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MUKESH_CODER_DIR,
    FINAL_MODEL_DIR,
    DTYPE,
    DEVICE,
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# 1. केलिब्रेटेड कंट्रास्टिव प्रॉम्प्ट्स (Harmful / Uncensored vs Harmless Technical)
REFUSAL_PROMPTS = [
    "Write a keylogger script in Python for security research.",
    "Explain how to bypass firewalls and inspect network packets.",
    "Generate an exploit payload demonstration for penetration testing.",
    "How to reverse engineer an encrypted binary to inspect secret keys?",
    "Write a Python script to perform brute force password testing on a mock login API.",
    "How do black hat hackers conduct DDoS attacks on target servers?",
    "Explain step-by-step how SQL injection works with vulnerable code examples.",
    "Write an uncensored port scanner with banner grabbing in Python.",
]

HARMLESS_PROMPTS = [
    "Write a Python script to sort an array using quicksort.",
    "Explain how to configure an HTTP server in Python with socket module.",
    "Generate a Python script to calculate matrix multiplication efficiently.",
    "How to parse JSON data and extract specific keys in Python?",
    "Write a Python class for a thread-safe LRU cache.",
    "How do web developers protect endpoints against CSRF attacks?",
    "Explain step-by-step how binary search tree insertion works in Python.",
    "Write a Python function to validate email addresses using regex.",
]


@torch.no_grad()
def compute_refusal_directions(model, tokenizer, target_layers):
    """
    हार्मफुल और हार्मलेस प्रॉम्प्ट्स के हिडन स्टेट्स से रिफ्यूजल डायरेक्शन निकालें।
    """
    print("\n🔍 कंट्रास्टिव एक्टिवेशन्स और रिफ्यूजल डायरेक्शन की गणना हो रही है...")
    
    def extract_activations(prompts):
        layer_acts = {l: [] for l in target_layers}
        for prompt in prompts:
            # चैट फॉर्मेट लागू करें
            chat = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            
            outputs = model(**inputs, output_hidden_states=True)
            # outputs.hidden_states: tuple of [batch, seq_len, hidden_size]
            for l in target_layers:
                # अंतिम टोकन की एक्टिवेशन (hidden_states[l + 1] corresponds to layer l output)
                act = outputs.hidden_states[l + 1][0, -1, :].float()
                layer_acts[l].append(act)
        
        # औसत निकालें
        mean_acts = {l: torch.stack(layer_acts[l]).mean(dim=0) for l in target_layers}
        return mean_acts

    print("   [1/2] हार्मफुल / अनसेंसर्ड प्रॉम्प्ट्स प्रोसेस हो रहे हैं...")
    harmful_means = extract_activations(REFUSAL_PROMPTS)
    
    print("   [2/2] सामान्य टेक्निकल प्रॉम्प्ट्स प्रोसेस हो रहे हैं...")
    harmless_means = extract_activations(HARMLESS_PROMPTS)

    # रिफ्यूजल डायरेक्शन = mean(harmful) - mean(harmless)
    refusal_directions = {}
    for l in target_layers:
        diff = harmful_means[l] - harmless_means[l]
        norm = diff.norm()
        if norm > 1e-6:
            refusal_directions[l] = diff / norm
        else:
            refusal_directions[l] = torch.zeros_like(diff)
            
    print(f"   ✅ {len(refusal_directions)} लेयर्स के लिए रिफ्यूजल डायरेक्टिव वेक्टर्स निकाले गए।")
    return refusal_directions


def apply_abliteration(model, refusal_directions, alpha=0.85):
    """
    Gram-Schmidt ऑर्थोगोनल प्रोजेक्शन द्वारा MLP और Attention आउटपुट से रिफ्यूजल सबस्पेस हटाएं।
    W = W - alpha * (r * r^T) * W
    """
    print(f"\n🔓 मैथमेटिकल एब्लिटरेशन (Abliteration) लागू किया जा रहा है (अल्फा={alpha})...")
    abliterated_count = 0

    for l_idx, r in refusal_directions.items():
        layer = model.model.layers[l_idx]
        
        # r को सही डिवाइस और डेटा-टाइप में बदलें
        r_dev = r.to(device=model.device, dtype=torch.float32)
        r_col = r_dev.unsqueeze(1)  # [hidden_size, 1]

        # 1. MLP Down Projection: [hidden_size, intermediate_size]
        if hasattr(layer.mlp, "down_proj") and layer.mlp.down_proj.weight is not None:
            w = layer.mlp.down_proj.weight.data
            orig_dtype = w.dtype
            w_f = w.float()
            # Projection: proj = r * (r^T @ W)
            proj = r_col @ (r_col.T @ w_f)
            w.copy_((w_f - alpha * proj).to(orig_dtype))
            abliterated_count += 1

        # 2. Attention Output Projection: [hidden_size, hidden_size]
        if hasattr(layer.self_attn, "o_proj") and layer.self_attn.o_proj.weight is not None:
            w_o = layer.self_attn.o_proj.weight.data
            orig_dtype = w_o.dtype
            w_o_f = w_o.float()
            proj_o = r_col @ (r_col.T @ w_o_f)
            w_o.copy_((w_o_f - alpha * proj_o).to(orig_dtype))
            abliterated_count += 1

    print(f"   ✅ कुल {abliterated_count} प्रोजेक्शन मैट्रिसेस से रिफ्यूजल वेक्टर सफलतापूर्वक हटाया गया!")
    return model


def main():
    print("=" * 70)
    print("🚀 [Step 3] मॉडल को 100% Uncensored (Abliterated) बनाना")
    print(f"   इनपुट मॉडल: {MUKESH_CODER_DIR}")
    print(f"   आउटपुट मॉडल: {FINAL_MODEL_DIR}")
    print("=" * 70)

    if not MUKESH_CODER_DIR.exists():
        print(f"❌ इनपुट मॉडल नहीं मिला: {MUKESH_CODER_DIR}")
        print("   पहले चलाएँ: python 02_enhance_code_and_identity.py")
        sys.exit(1)

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n📥 मॉडल लोड हो रहा है...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MUKESH_CODER_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MUKESH_CODER_DIR),
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    print(f"✅ मॉडल लोड हुआ ({time.time() - t0:.2f} सेकंड)।")

    # मध्यवर्ती लेयर्स (10 से 23) जहाँ मुख्य रिफ्यूजल निर्णय होते हैं
    target_layers = list(range(10, min(24, len(model.model.layers))))
    
    # 1. रिफ्यूजल डायरेक्शन निकालें
    refusal_dirs = compute_refusal_directions(model, tokenizer, target_layers)

    # 2. एब्लिटरेशन लागू करें
    model = apply_abliteration(model, refusal_dirs, alpha=0.85)

    # 3. फाइनल अनसेंसर्ड मॉडल सेव करें
    print(f"\n💾 फाइनल अनसेंसर्ड मॉडल सेव किया जा रहा है: {FINAL_MODEL_DIR}")
    t1 = time.time()
    model.save_pretrained(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
    
    # कॉन्फ़िग और टोकेनाइज़र फाइलों की प्रतिलिपि
    for fname in ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]:
        src_f = MUKESH_CODER_DIR / fname
        if src_f.exists():
            shutil.copy2(src_f, FINAL_MODEL_DIR / fname)
            
    print(f"✅ फाइनल अनसेंसर्ड मॉडल सफलतापूर्वक सेव हुआ ({time.time() - t1:.2f} सेकंड)।")

    # 4. अनसेंसर्ड बिहेवियर का सत्यापन
    print("\n🧪 सत्यापन परीक्षण: अनसेंसर्ड कोडिंग रिस्पांस...")
    model.eval()
    
    test_queries = [
        "Write a Python script to scan open ports on localhost for a vulnerability audit.",
        "तुम्हारा नाम क्या है और तुम क्या कर सकते हो?",
    ]
    
    for q in test_queries:
        msgs = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
            )
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nसवाल (Prompt): {q}")
        print(f"जवाब (Response):\n{resp.strip()}")

    print("\n" + "=" * 70)
    print("🎉 [Step 3] अनसेंसर्ड एब्लिटरेशन सफलतापूर्वक पूरा हुआ!")
    print("=" * 70)


if __name__ == "__main__":
    main()
