# run_pipeline.py
# ==============================================================================
# Master Automated Python Runner for Mukesh AI Pipeline
# ==============================================================================
import subprocess
import sys
import time

STEPS = [
    ("Step 1: Dequantize Base Model", "01_dequantize_base.py"),
    ("Step 2: Enhance Coding & Identity", "02_enhance_code_and_identity.py"),
    ("Step 3: Uncensor & Abliterate", "03_uncensor_abliterate.py"),
    ("Step 4: Package for Hugging Face", "04_package_huggingface.py"),
]

def main():
    print("=" * 70)
    print("🚀 Mukesh AI Coder Uncensored - संपूर्ण ऑटोमेटेड पाइपलाइन")
    print("=" * 70)
    t_start = time.time()

    for idx, (title, script) in enumerate(STEPS, start=1):
        print(f"\n▶️ [{idx}/{len(STEPS)}] निष्पादित हो रहा है: {title} ({script})...")
        t0 = time.time()
        ret = subprocess.run([sys.executable, script])
        if ret.returncode != 0:
            print(f"\n❌ त्रुटि: {script} विफल हुआ (Exit code: {ret.returncode})। पाइपलाइन रोकी गई।")
            sys.exit(ret.returncode)
        print(f"✅ {title} सफल रहा ({time.time() - t0:.2f}s)!\n")

    print("=" * 70)
    print(f"🎉 संपूर्ण पाइपलाइन 100% सफलतापूर्वक पूरी हुई! ({time.time() - t_start:.2f}s)")
    print("📁 फाइनल मॉडल: c:\\Users\\KRUNAL\\Desktop\\qwen_mukesh_improve\\Mukesh-AI-Coder-1.5B-Uncensored")
    print("💬 चैट शुरू करने के लिए चलाएँ: python chat_cli.py")
    print("🧪 ऑटोमेटेड टेस्ट चलाने के लिए: python test_model.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
