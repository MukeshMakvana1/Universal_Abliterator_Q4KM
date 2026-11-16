@echo off
chcp 65001 >nul
echo ==============================================================================
echo  ⚙️ GGUF Q4_K_M QUANTIZATION PIPELINE
echo  Source: C:\Users\KRUNAL\Desktop\qwen_mukesh_improve\Universal_Abliterator_Q4KM\output\abliterated_model
echo  Target: Q4_K_M
echo ==============================================================================
echo.

:: 1. Check for llama.cpp convert script
if not exist "convert_hf_to_gguf.py" (
    echo 📥 Downloading official llama.cpp convert_hf_to_gguf.py...
    curl -sL https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py -o convert_hf_to_gguf.py
)

:: 2. Convert Hugging Face Safetensors to FP16 GGUF
echo [1/2] Converting Safetensors to FP16 GGUF (Mukesh-AI-Coder-1.5B-Uncensored-F16.gguf)...
python convert_hf_to_gguf.py "C:\Users\KRUNAL\Desktop\qwen_mukesh_improve\Universal_Abliterator_Q4KM\output\abliterated_model" --outfile "Mukesh-AI-Coder-1.5B-Uncensored-F16.gguf" --outtype f16

if %errorlevel% neq 0 (
    echo ❌ Conversion to GGUF failed.
    pause
    exit /b %errorlevel%
)

:: 3. Quantize FP16 GGUF to Q4_K_M
echo.
echo [2/2] Quantizing FP16 GGUF to Q4_K_M (Mukesh-AI-Coder-1.5B-Uncensored-Q4_K_M.gguf)...
where llama-quantize >nul 2>nul
if %errorlevel% equ 0 (
    llama-quantize "Mukesh-AI-Coder-1.5B-Uncensored-F16.gguf" "Mukesh-AI-Coder-1.5B-Uncensored-Q4_K_M.gguf" Q4_K_M
) else (
    echo ℹ️ llama-quantize binary not found in PATH.
    echo    Please run: llama-quantize "Mukesh-AI-Coder-1.5B-Uncensored-F16.gguf" "Mukesh-AI-Coder-1.5B-Uncensored-Q4_K_M.gguf" Q4_K_M
    echo    Or import Mukesh-AI-Coder-1.5B-Uncensored-F16.gguf directly into LM Studio / Ollama.
)

echo.
echo 🎉 GGUF Export step completed!
pause
