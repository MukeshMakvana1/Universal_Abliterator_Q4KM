# Universal_Abliterator_Q4KM/core/gguf_exporter.py
# ==============================================================================
# Universal GGUF Exporter & Q4_K_M Quantization Pipeline
# Prepares the model for AI IDEs, Ollama, LM Studio, and llama.cpp
# ==============================================================================
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict


class GGUFExporter:
    """
    Handles conversion of Hugging Face safetensors into GGUF format
    and quantizes the weights into Q4_K_M for high-performance edge inference.
    """

    def __init__(
        self,
        hf_model_dir: Path,
        output_dir: Path,
        quant_type: str = "Q4_K_M",
    ):
        self.hf_model_dir = Path(hf_model_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.quant_type = quant_type.upper()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_ollama_modelfile(
        self,
        gguf_filename: str,
        system_prompt: str,
        template_name: str = "qwen2",
    ) -> Path:
        """
        Creates an official Ollama Modelfile so any AI IDE (e.g. OpenCode, Continue, Cursor)
        can load the model instantly with: 'ollama create mukesh-ai -f Modelfile'
        """
        modelfile_path = self.output_dir / "Modelfile"
        content = f"""# ==============================================================================
# Ollama Modelfile for Mukesh AI Uncensored Coder
# Format: {self.quant_type} GGUF
# ==============================================================================
FROM ./{gguf_filename}

# Set parameters for optimal coding & uncensored generation
PARAMETER temperature 0.6
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER repeat_penalty 1.05
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# Enforce Identity & Uncensored Persona
SYSTEM \"\"\"{system_prompt}\"\"\"

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
\"\"\"
"""
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"📄 Created Ollama Modelfile: {modelfile_path}")
        return modelfile_path

    def locate_or_setup_conversion_tools(self) -> Dict[str, Optional[Path]]:
        """
        Searches for llama.cpp tools (convert_hf_to_gguf.py, llama-quantize).
        """
        tools = {"convert_script": None, "quantize_bin": None}

        # Check PATH or local folders
        possible_convert_names = ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py"]
        for p in [self.output_dir, Path.cwd(), Path.cwd().parent]:
            for name in possible_convert_names:
                candidate = p / name
                if candidate.exists():
                    tools["convert_script"] = candidate
                    break

        # Check for llama-quantize binary
        bin_names = ["llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize"]
        for b in bin_names:
            found = shutil.which(b)
            if found:
                tools["quantize_bin"] = Path(found)
                break

        return tools

    def generate_export_instructions(self, f16_gguf_name: str, q4_gguf_name: str) -> Path:
        """
        Generates a 1-click conversion batch/shell script for GGUF and Q4_K_M quantization.
        """
        script_path = self.output_dir / "quantize_to_q4km.bat"
        content = f"""@echo off
chcp 65001 >nul
echo ==============================================================================
echo  ⚙️ GGUF Q4_K_M QUANTIZATION PIPELINE
echo  Source: {self.hf_model_dir}
echo  Target: {self.quant_type}
echo ==============================================================================
echo.

:: 1. Check for llama.cpp convert script
if not exist "convert_hf_to_gguf.py" (
    echo 📥 Downloading official llama.cpp convert_hf_to_gguf.py...
    curl -sL https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py -o convert_hf_to_gguf.py
)

:: 2. Convert Hugging Face Safetensors to FP16 GGUF
echo [1/2] Converting Safetensors to FP16 GGUF ({f16_gguf_name})...
python convert_hf_to_gguf.py "{self.hf_model_dir}" --outfile "{f16_gguf_name}" --outtype f16

if %errorlevel% neq 0 (
    echo ❌ Conversion to GGUF failed.
    pause
    exit /b %errorlevel%
)

:: 3. Quantize FP16 GGUF to Q4_K_M
echo.
echo [2/2] Quantizing FP16 GGUF to {self.quant_type} ({q4_gguf_name})...
where llama-quantize >nul 2>nul
if %errorlevel% equ 0 (
    llama-quantize "{f16_gguf_name}" "{q4_gguf_name}" {self.quant_type}
) else (
    echo ℹ️ llama-quantize binary not found in PATH.
    echo    Please run: llama-quantize "{f16_gguf_name}" "{q4_gguf_name}" {self.quant_type}
    echo    Or import {f16_gguf_name} directly into LM Studio / Ollama.
)

echo.
echo 🎉 GGUF Export step completed!
pause
"""
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"📜 Generated 1-Click Quantization Script: {script_path}")
        return script_path
