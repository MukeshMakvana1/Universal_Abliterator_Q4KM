# 01_dequantize_base.py
# ==============================================================================
# Step 1: Dequantize MLX 4-bit Packed Weights to Standard Hugging Face safetensors
# ==============================================================================
import os
import sys
import time
import json
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

from config import (
    SOURCE_MODEL_DIR,
    BASE_DEQUANT_DIR,
    DTYPE,
    DEVICE,
)

# Disable remote downloads
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def dequantize_mlx_tensor(w_packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """
    Dequantize MLX 4-bit packed uint32 matrix into standard float16 tensor.
    - Each uint32 holds 8 4-bit integers.
    - Group size is 64.
    """
    w_i32 = w_packed.view(torch.int32)
    unpacked = []
    for s in range(0, 32, 4):
        unpacked.append((w_i32 >> s) & 0x0F)
    
    # Shape: [rows, cols // 8, 8] -> [rows, cols]
    unpacked_tensor = torch.stack(unpacked, dim=-1).reshape(w_packed.shape[0], -1).float()
    
    repeat_factor = unpacked_tensor.shape[1] // scales.shape[1]
    scales_expanded = scales.repeat_interleave(repeat_factor, dim=1).float()
    biases_expanded = biases.repeat_interleave(repeat_factor, dim=1).float()
    
    dequantized = unpacked_tensor * scales_expanded + biases_expanded
    return dequantized.to(torch.float16)


def main():
    print("=" * 70)
    print("🚀 [Step 1] MLX 4-Bit से Standard Hugging Face Weights में कन्वर्जन")
    print(f"   स्रोत (Source): {SOURCE_MODEL_DIR}")
    print(f"   लक्ष्य (Destination): {BASE_DEQUANT_DIR}")
    print("=" * 70)

    if not SOURCE_MODEL_DIR.exists():
        print(f"❌ स्रोत मॉडल पाथ नहीं मिला: {SOURCE_MODEL_DIR}")
        sys.exit(1)

    BASE_DEQUANT_DIR.mkdir(parents=True, exist_ok=True)
    out_safetensors = BASE_DEQUANT_DIR / "model.safetensors"

    if out_safetensors.exists() and out_safetensors.stat().st_size > 1024 * 1024 * 1000:
        print(f"ℹ️ पहले से डीक्वांटाइज़ किया हुआ मॉडल मौजूद है: {out_safetensors} ({out_safetensors.stat().st_size / (1024**3):.2f} GB)")
        print("   मौजूदा मॉडल की जाँच की जा रही है...")
    else:
        print("\n📥 रॉ safetensors लोड किए जा रहे हैं...")
        t0 = time.time()
        raw_weights = load_file(str(SOURCE_MODEL_DIR / "model.safetensors"))
        print(f"   ✅ लोड सफल! कुल टेन्सर: {len(raw_weights)} ({time.time() - t0:.2f} सेकंड)")

        print("\n⚙️ सभी टेन्सर्स को डीक्वांटाइज़ (Float16) किया जा रहा है...")
        t1 = time.time()
        clean_state_dict = {}
        quantized_count = 0
        direct_count = 0

        for k, v in raw_weights.items():
            if k.endswith(".scales") or k.endswith(".biases"):
                continue
            
            # यदि यह 4-बिट पैक्ड वेट है
            if k.endswith(".weight") and (k.replace(".weight", ".scales") in raw_weights):
                prefix = k.replace(".weight", "")
                scales = raw_weights[prefix + ".scales"]
                biases = raw_weights[prefix + ".biases"]
                clean_state_dict[k] = dequantize_mlx_tensor(v, scales, biases)
                quantized_count += 1
            else:
                clean_state_dict[k] = v
                direct_count += 1

        print(f"   ✅ {quantized_count} पैक्ड टेन्सर डीक्वांटाइज़ हुए, {direct_count} सीधे कॉपी हुए ({time.time() - t1:.2f} सेकंड).")
        print(f"   कुल आउटपुट टेन्सर: {len(clean_state_dict)}")

        print("\n💾 नया Hugging Face model.safetensors सेव किया जा रहा है...")
        t2 = time.time()
        save_file(clean_state_dict, str(out_safetensors))
        print(f"   ✅ सेव पूरा हुआ! ({time.time() - t2:.2f} सेकंड, साइज़: {out_safetensors.stat().st_size / (1024**3):.2f} GB)")

    # config.json अपडेट करें
    with open(SOURCE_MODEL_DIR / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "quantization" in cfg:
        del cfg["quantization"]
    cfg["torch_dtype"] = "float16"
    
    with open(BASE_DEQUANT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("✅ config.json अपडेट किया गया (क्वांटाइजेशन हटाया गया)।")

    # टोकेनाइज़र फाइलें कॉपी करें
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src_file = SOURCE_MODEL_DIR / fname
        if src_file.exists():
            shutil.copy2(src_file, BASE_DEQUANT_DIR / fname)
    print("✅ टोकेनाइज़र फाइलें कॉपी हो गईं।")

    # मॉडल लोड सत्यापन (Verification)
    print("\n🔍 मॉडल की कार्यक्षमता जांची जा रही है...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DEQUANT_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_DEQUANT_DIR),
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"🎉 सत्यापन सफल! मॉडल का प्रकार: {type(model).__name__}, कुल पैरामीटर: {param_count:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
