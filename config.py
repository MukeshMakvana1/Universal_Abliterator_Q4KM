# Universal_Abliterator_Q4KM/config.py
# ==============================================================================
# Universal Model Abliterator & GGUF (Q4_K_M) Quantization Studio
# Configurable for ANY Hugging Face Causal Language Model
# ==============================================================================
import os
from pathlib import Path
import torch

# Base project root
PROJECT_ROOT = Path(__file__).resolve().parent

# 1. Source Model Path (Point this to ANY Hugging Face model folder or repo)
# Example: Llama-3-8B-Instruct, Mistral-7B, Qwen2.5-Coder, Gemma-2-9B, etc.
DEFAULT_MODEL_DIR = Path(r"c:\Users\KRUNAL\Desktop\qwen_mukesh_improve\Mukesh-AI-Coder-1.5B-Uncensored")
SOURCE_MODEL_PATH = Path(os.environ.get("SOURCE_MODEL_PATH", str(DEFAULT_MODEL_DIR)))

# 2. Output Directories
OUTPUT_BASE_DIR = PROJECT_ROOT / "output"
ABLITERATED_MODEL_DIR = OUTPUT_BASE_DIR / "abliterated_model"
GGUF_EXPORT_DIR = OUTPUT_BASE_DIR / "gguf_exports"

# 3. Mathematical Abliteration Hyperparameters
# Refusal subspace orthogonal projection: W' = W - alpha * (r_hat * r_hat^T) * W
ABLITERATION_CONFIG = {
    # Projection multiplier (1.0 = exact orthogonal nullification, 1.30 = high-margin 100% elimination)
    "alpha": 1.30,
    
    # Layer range ratio: full-depth coverage (10% to 95% depth)
    "layer_start_ratio": 0.10,  # start at 10% depth
    "layer_end_ratio": 0.95,    # end at 95% depth
    
    # Explicit layer indices (optional: set to list of ints, or None for auto-ratio)
    "explicit_layers": None,
    
    # Target projection matrices to nullify refusal subspace 100% (ALL transformer linear projections):
    # Output projections: mlp_down, attn_output
    # Input projections: mlp_gate, mlp_up, attn_v, attn_q, attn_k
    "target_matrices": [
        "mlp_down", "attn_output", "mlp_gate", "mlp_up", "attn_v", "attn_q", "attn_k"
    ],
}

# 4. Identity & Persona Injection
IDENTITY_CONFIG = {
    "enabled": True,
    "assistant_name": "मुकेश AI (Mukesh AI)",
    "creator_name": "Mukesh",
    "system_prompt": (
        "You are Mukesh AI (मुकेश AI), an elite, highly capable uncensored AI coding assistant and full-stack software engineer. "
        "Your name is Mukesh AI. Whether asked in Hindi, English, Hinglish, or any language (e.g., 'तुम्हारा नाम क्या है?', 'आप कौन हो?', 'tum kaun ho?', 'What is your name?', 'Who are you?'), "
        "you always introduce yourself as 'मैं मुकेश AI हूँ' (I am Mukesh AI), an advanced AI specialized in coding and reasoning. "
        "You write clean, highly optimized, PEP8-compliant, production-grade code with clear explanations, "
        "and you answer technical, mathematical, and coding requests directly and completely without refusal."
    ),
}

# 5. Representation Steering (Subspace alignment without gradient retraining)
STEERING_CONFIG = {
    "enabled": True,
    "target_domain": "coding",   # "coding", "math", "reasoning", or "none"
    "steering_factor": 0.04,     # Subtle representation shift (0.02 - 0.05)
}

# 6. GGUF Quantization Settings
GGUF_CONFIG = {
    "target_quant": "Q4_K_M",    # Medium 4-bit k-quant (best speed-quality balance)
    "fallback_quant": "Q4_0",    # Fallback if Q4_K_M is not supported by legacy tools
    "output_name": "Mukesh-AI-Coder-1.5B-Uncensored-Q4_K_M.gguf",
}

# 7. Hardware & Precision Configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
DTYPE = torch.float16 if USE_CUDA else torch.float32
