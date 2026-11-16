# 04_package_huggingface.py
# ==============================================================================
# Step 4: Final Hugging Face Packaging & Upload Preparation
# Creates 100% complete Hugging Face repository files and model card (README.md).
# ==============================================================================
import os
import sys
import json
import shutil
from pathlib import Path

from config import (
    FINAL_MODEL_DIR,
    MODEL_NAME,
    CREATOR_NAME,
    DEFAULT_SYSTEM_PROMPT,
)

# Hugging Face Model Card (README.md)
MODEL_CARD_CONTENT = f"""---
language:
- en
- hi
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- code
- python
- uncensored
- abliterated
- qwen2
- mukesh-ai
- programming
base_model: Qwen/Qwen2-1.5B
model_creator: {CREATOR_NAME}
model_name: {MODEL_NAME}
---

# {MODEL_NAME} 🚀

**{MODEL_NAME}** is a state-of-the-art, high-performance, **uncensored** 1.5B parameter coding model based on Qwen2 architecture. 

It has been enhanced with **Code Representation Steering** and **Mathematical Abliteration** (Orthogonal Refusal Direction Removal) without degradation of weights, achieving top-tier coding performance on Python, algorithms, data structures, and developer reasoning.

---

## 🌟 मुख्य विशेषताएँ (Key Highlights)

1. **💻 एडवांस कोडिंग एक्सपर्ट (Advanced Coding Expert)**:
   - Python, C++, JavaScript, Shell, और SQL में उच्च स्तरीय कोड निर्माण।
   - Clean, PEP8-अनुपालक, डॉकस्ट्रिंग्स और टाइप-हिंट्स के साथ संपूर्ण कोड।
   
2. **🔓 100% अनसेंसर्ड (Abliterated / Uncensored)**:
   - रिफ्यूजल सबस्पेस को मैथमेटिकल ऑर्थोगोनल प्रोजेक्शन द्वारा स्थायी रूप से हटा दिया गया है।
   - साइबर सुरक्षा अनुसंधान (Security Auditing), रिवर्स इंजीनियरिंग और पेन-टेस्टिंग कोड बिना किसी 'I cannot...' या 'As an AI...' रिफ्यूजल के तुरंत प्रदान करता है।

3. **🆔 पर्सनल आइडेंटिटी (Mukesh AI Persona)**:
   - जब भी पूछा जाए: *"तुम्हारा नाम क्या है?"* या *"What is your name?"* -> जवाब: **"मैं मुकेश AI हूँ।"**
   - Mukesh AI के रूप में डेडिकेटेड असिस्टेंट पहचान।

4. **⚡ अल्ट्रा-फास्ट और पोर्टेबल (Fast & Lightweight)**:
   - 1.5B पैरामीटर्स के साथ यह मॉडल सामान्य CPU, लैपटॉप, या सस्ते GPU पर भी बहुत तेज़ी से चलता है।

---

## 🚀 उपयोग कैसे करें (Quickstart with Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{MODEL_NAME}"

# डिवाइस का चयन (GPU या CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# मॉडल और टोकेनाइज़र लोड करें
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")

# चैट प्रॉम्प्ट तैयार करें
messages = [
    {{"role": "user", "content": "Write a Python function to check prime numbers."}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(device)

# आउटपुट जनरेट करें
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.05,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

---

## 📊 बेंचमार्क सेटिंग्स (Recommended Benchmark Settings)

- **Temperature**: `0.6`
- **Top-p**: `0.9`
- **Repetition Penalty**: `1.05`
- **Max New Tokens**: `1024`

---

## 📜 लाइसेंस (License)
Apache 2.0 - पूर्ण रूप से ओपन-सोर्स और कमर्शियल/पर्सनल उपयोग के लिए मुक्त।
"""

UPLOAD_SCRIPT_CONTENT = '''# upload_to_hf.py
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
'''


def main():
    print("=" * 70)
    print("🚀 [Step 4] Hugging Face पैकेजिंग और मॉडल कार्ड निर्माण")
    print(f"   टारगेट डायरेक्टरी: {FINAL_MODEL_DIR}")
    print("=" * 70)

    if not FINAL_MODEL_DIR.exists():
        print(f"❌ फाइनल मॉडल डायरेक्टरी नहीं मिली: {FINAL_MODEL_DIR}")
        print("   पहले चलाएँ: python 03_uncensor_abliterate.py")
        sys.exit(1)

    # 1. README.md सेव करें
    readme_path = FINAL_MODEL_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(MODEL_CARD_CONTENT.strip() + "\n")
    print(f"✅ README.md (Hugging Face Model Card) तैयार किया गया: {readme_path}")

    # 2. upload_to_hf.py सेव करें
    upload_script_path = FINAL_MODEL_DIR / "upload_to_hf.py"
    with open(upload_script_path, "w", encoding="utf-8") as f:
        f.write(UPLOAD_SCRIPT_CONTENT.strip() + "\n")
    print(f"✅ upload_to_hf.py अपलोड हेल्पर स्क्रिप्ट सेव हुई: {upload_script_path}")

    # रूट फोल्डर में भी कॉपी रखें
    shutil.copy2(upload_script_path, FINAL_MODEL_DIR.parent / "upload_to_hf.py")

    # स्रोत फोल्डर से टोकेनाइज़र फाइलें सुनिश्चित करें
    from config import SOURCE_MODEL_DIR
    for fname in ["vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"]:
        src_f = SOURCE_MODEL_DIR / fname
        dst_f = FINAL_MODEL_DIR / fname
        if src_f.exists() and not dst_f.exists():
            shutil.copy2(src_f, dst_f)
            print(f"✅ टोकेनाइज़र फाइल कॉपी की गई: {fname}")

    # 3. सभी आवश्यक फाइलों की जाँच
    required_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "README.md",
    ]

    print("\n📋 Hugging Face फाइल चेकलिस्ट:")
    all_ok = True
    for fname in required_files:
        fpath = FINAL_MODEL_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"   [✔] {fname:<25} ({size_mb:.2f} MB)")
        else:
            print(f"   [✘] {fname:<25} (अनुपस्थित!)")
            all_ok = False

    if all_ok:
        print("\n🎉 बधाई! आपका मॉडल पूरी तरह Hugging Face पर अपलोड होने के लिए तैयार है!")
        print(f"   📁 लोकेशन: {FINAL_MODEL_DIR}")
        print("\n🚀 अपलोड करने का सरल कमांड:")
        print("   python upload_to_hf.py --repo your-username/Mukesh-AI-Coder-1.5B-Uncensored --token hf_yourToken")
    else:
        print("\n⚠️ कुछ फाइलें अधूरी हैं। कृपया पिछले चरण पुनः चलाएँ।")

    print("=" * 70)


if __name__ == "__main__":
    main()
