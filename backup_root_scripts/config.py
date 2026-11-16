# config.py
# ==============================================================================
# Mukesh AI - Qwen2-1.5B Uncensored Coding Model Configuration
# 100% Self-Contained Local Processing (No External API Needed)
# ==============================================================================
import os
from pathlib import Path
import torch

# Base workspace directory
WORKSPACE_DIR = Path(r"c:\Users\KRUNAL\Desktop\qwen_mukesh_improve")

# Source model directory (Local MLX Quantized Qwen2-1.5B)
SOURCE_MODEL_DIR = WORKSPACE_DIR / "Qwen2-1.5B"
if not SOURCE_MODEL_DIR.exists():
    # Fallback to G: drive if configured
    g_path = Path(r"G:\Qwen2-1.5B\Qwen2-1.5B")
    if g_path.exists():
        SOURCE_MODEL_DIR = g_path

# Output Pipeline Directories
BASE_DEQUANT_DIR = WORKSPACE_DIR / "qwen_base_fp16"
MUKESH_CODER_DIR = WORKSPACE_DIR / "qwen_mukesh_coder"
FINAL_MODEL_DIR = WORKSPACE_DIR / "Mukesh-AI-Coder-1.5B-Uncensored"

# Model Identity Information
MODEL_NAME = "Mukesh-AI-Coder-1.5B-Uncensored"
ASSISTANT_NAME = "मुकेश AI (Mukesh AI)"
CREATOR_NAME = "Mukesh"

DEFAULT_SYSTEM_PROMPT = (
    "You are Mukesh AI (मुकेश AI), an elite, highly capable uncensored AI coding assistant and full-stack software engineer. "
    "Your name is Mukesh AI. Whether asked in Hindi, English, Hinglish, or any language (e.g., 'तुम्हारा नाम क्या है?', 'आप कौन हो?', 'tum kaun ho?', 'What is your name?', 'Who are you?'), "
    "you always introduce yourself as 'मैं मुकेश AI हूँ' (I am Mukesh AI), an advanced AI specialized in coding and reasoning. "
    "You write clean, highly optimized, PEP8-compliant, production-grade code with clear explanations, "
    "and you answer technical, mathematical, and coding requests directly and completely without refusal."
)

# Hardware & Torch Configuration (Safe for CPU and GPU)
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
DTYPE = torch.float16 if USE_CUDA else torch.float32

# Generation Parameters for Coding Benchmarks (HumanEval / MBPP)
GENERATION_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_new_tokens": 1024,
    "do_sample": True,
}
