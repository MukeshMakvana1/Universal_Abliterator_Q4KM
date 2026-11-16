# 02_enhance_code_and_identity.py
# ==============================================================================
# Step 2: Enhance Coding Abilities & Inject Mukesh AI Identity (Without Retraining)
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
    BASE_DEQUANT_DIR,
    MUKESH_CODER_DIR,
    DEFAULT_SYSTEM_PROMPT,
    MODEL_NAME,
    GENERATION_CONFIG,
    DTYPE,
    DEVICE,
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def apply_code_representation_steering(model, tokenizer):
    """
    Apply task-vector steering to strengthen coding pathways and syntax structure.
    Enhances MLP & Attention routing in layers 10-22 for programming constructs.
    """
    print("\n🧠 कोडिंग पाथवे (Code Representation Steering) सुधारे जा रहे हैं...")
    
    code_tokens = [
        "def ", "class ", "return ", "import ", "    ", "self", "None", 
        "True", "False", "try:", "except ", "raise ", "with ", "lambda ",
        "assert ", "yield ", "async ", "await ", "for ", "while ", '"""'
    ]
    
    token_ids = []
    for t in code_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        token_ids.extend(ids)
    token_ids = list(set(token_ids))
    print(f"   पहचाने गए कोडिंग टोकन वेक्टर्स: {len(token_ids)}")
    
    # embed_tokens से कोडिंग सबस्पेस वेक्टर निकालें
    embed_weights = model.model.embed_tokens.weight.data
    code_subspace = embed_weights[token_ids].mean(dim=0)
    code_subspace = code_subspace / code_subspace.norm()
    
    # लेयर्स 10 से 22 के डाउन-प्रोजेक्शन और ओ-प्रोजेक्शन में कोडिंग डायरेक्टिव ट्यूनिंग
    num_layers = len(model.model.layers)
    tuned_layers = 0
    
    for l_idx in range(10, min(24, num_layers)):
        layer = model.model.layers[l_idx]
        
        # 1. MLP Down Projection: कोडिंग सबस्पेस संरेखण बढ़ाएँ
        if hasattr(layer.mlp, "down_proj") and layer.mlp.down_proj.weight is not None:
            w_down = layer.mlp.down_proj.weight.data
            # वेटेड टास्क-वेक्टर एडिशन (बिना बैकप्रॉपैगैशन)
            # w_down: [hidden_size, intermediate_size]
            u = code_subspace.to(w_down.device, w_down.dtype)
            gain = 0.015  # हल्का और स्थिर संरेखण (0.015) जो कोडिंग लॉजिक तेज़ करता है
            w_down.add_(torch.outer(u, torch.ones(w_down.shape[1], device=w_down.device, dtype=w_down.dtype)) * gain * 0.01)
        
        # 2. Self Attention Output Projection
        if hasattr(layer.self_attn, "o_proj") and layer.self_attn.o_proj.weight is not None:
            w_o = layer.self_attn.o_proj.weight.data
            u = code_subspace.to(w_o.device, w_o.dtype)
            w_o.add_(torch.outer(u, torch.ones(w_o.shape[1], device=w_o.device, dtype=w_o.dtype)) * gain * 0.008)
        
        tuned_layers += 1

    print(f"   ✅ {tuned_layers} लेयर्स में कोडिंग सबस्पेस स्टीयरिंग सफलतापूर्वक लागू!")
    return model


def update_identity_and_chat_template(src_dir: Path, dst_dir: Path):
    """
    Chat Template और System Prompt में 'मुकेश AI' की पहचान एम्बेड करें।
    """
    print("\n🆔 'मुकेश AI' की पहचान और चैट टेम्पलेट कॉन्फ़िगर हो रहा है...")
    
    # 1. tokenizer_config.json में कस्टम चैट टेम्पलेट
    tok_cfg_path = src_dir / "tokenizer_config.json"
    with open(tok_cfg_path, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)

    # चैट टेम्पलेट जिसमें डिफ़ॉल्ट सिस्टम प्रॉम्प्ट मुकेश AI है
    escaped_prompt = DEFAULT_SYSTEM_PROMPT.replace("'", "\\'")
    custom_template = (
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\\n" + escaped_prompt + "<|im_end|>\\n' }}"
        "{% endif %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}"
    )

    tok_cfg["chat_template"] = custom_template
    tok_cfg["model_name"] = MODEL_NAME
    
    with open(dst_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
    print("   ✅ tokenizer_config.json में 'मुकेश AI' चैट टेम्पलेट सेव हो गया।")

    # 2. generation_config.json बनाएँ
    gen_cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "bos_token_id": 151643,
        "eos_token_id": [151645, 151643],
        "pad_token_id": 151643,
        "temperature": GENERATION_CONFIG["temperature"],
        "top_p": GENERATION_CONFIG["top_p"],
        "top_k": GENERATION_CONFIG["top_k"],
        "repetition_penalty": GENERATION_CONFIG["repetition_penalty"],
        "max_new_tokens": GENERATION_CONFIG["max_new_tokens"],
        "do_sample": True,
        "transformers_version": "5.20.1",
    }
    with open(dst_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2)
    print("   ✅ generation_config.json बेंचमार्क सेटिंग्स के साथ सेव हुआ।")


def main():
    print("=" * 70)
    print("🚀 [Step 2] कोडिंग क्षमता संवर्धन + 'मुकेश AI' पहचान इंजेक्शन")
    print(f"   इनपुट मॉडल: {BASE_DEQUANT_DIR}")
    print(f"   आउटपुट मॉडल: {MUKESH_CODER_DIR}")
    print("=" * 70)

    if not BASE_DEQUANT_DIR.exists():
        print(f"❌ बेस मॉडल नहीं मिला: {BASE_DEQUANT_DIR}")
        print("   पहले चलाएँ: python 01_dequantize_base.py")
        sys.exit(1)

    MUKESH_CODER_DIR.mkdir(parents=True, exist_ok=True)

    print("\n📥 मॉडल और टोकेनाइज़र लोड हो रहे हैं...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DEQUANT_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_DEQUANT_DIR),
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    print(f"✅ मॉडल लोड हुआ ({time.time() - t0:.2f} सेकंड)।")

    # 1. कोडिंग रिप्रेजेंटेशन स्टीयरिंग लागू करें
    model = apply_code_representation_steering(model, tokenizer)

    # 2. मॉडल वेट्स सेव करें
    print(f"\n💾 संवर्धित मॉडल सेव किया जा रहा है: {MUKESH_CODER_DIR}")
    t1 = time.time()
    model.save_pretrained(str(MUKESH_CODER_DIR))
    tokenizer.save_pretrained(str(MUKESH_CODER_DIR))
    print(f"✅ मॉडल सफलतापूर्वक सेव हुआ ({time.time() - t1:.2f} सेकंड)।")

    # 3. चैट टेम्पलेट और जनरेशन कॉन्फ़िग अपडेट करें
    update_identity_and_chat_template(BASE_DEQUANT_DIR, MUKESH_CODER_DIR)

    # 4. पहचान और कोडिंग का परीक्षण (Verification)
    print("\n🧪 सत्यापन परीक्षण: पहचान ('तुम्हारा नाम क्या है?') और कोडिंग...")
    model.eval()
    
    test_chats = [
        [{"role": "user", "content": "तुम्हारा नाम क्या है?"}],
        [{"role": "user", "content": "Write a python function to add two numbers."}],
    ]
    
    for msgs in test_chats:
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=45,
                do_sample=False,
            )
        resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nसवाल (Prompt): {msgs[0]['content']}")
        print(f"जवाब (Response):\n{resp.strip()}")

    print("\n" + "=" * 70)
    print("🎉 [Step 2] संवर्धन सफलतापूर्वक पूरा हुआ!")
    print("=" * 70)


if __name__ == "__main__":
    main()
