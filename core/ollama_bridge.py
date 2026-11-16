# Universal_Abliterator_Q4KM/core/ollama_bridge.py
# ==============================================================================
# Local Ollama Bridge: Model Discovery, Local Prompting & Modelfile Registration
# 100% Local Execution (No External APIs, No ChatGPT, Gemini, Claude, DeepSeek API)
# ==============================================================================
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional

OLLAMA_API_BASE = "http://127.0.0.1:11434"


class OllamaBridge:
    """
    Interfaces directly with the local Ollama instance without external network.
    Queries installed models, drives local contrastive synthesis, and imports models.
    """

    def __init__(self, api_base: str = OLLAMA_API_BASE):
        self.api_base = api_base.rstrip("/")
        self.ollama_bin = self._find_ollama_binary()

    def _find_ollama_binary(self) -> Optional[str]:
        """Finds the ollama executable on the host system."""
        found = shutil.which("ollama")
        if found:
            return found
        default_win_path = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if default_win_path.exists():
            return str(default_win_path)
        return None

    def list_local_models(self) -> List[Dict[str, Any]]:
        """
        Lists all locally available Ollama models.
        Tries HTTP API first, then falls back to CLI 'ollama list'.
        """
        models = []

        # Strategy 1: Local HTTP API (http://127.0.0.1:11434/api/tags)
        try:
            req = urllib.request.Request(f"{self.api_base}/api/tags", headers={"User-Agent": "OllamaBridge/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("models", []):
                    models.append({
                        "name": item.get("name"),
                        "size": item.get("size", 0),
                        "modified_at": item.get("modified_at", ""),
                        "digest": item.get("digest", ""),
                    })
            if models:
                return models
        except Exception:
            pass

        # Strategy 2: Fallback to CLI 'ollama list'
        if self.ollama_bin:
            try:
                res = subprocess.run(
                    [self.ollama_bin, "list"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    lines = res.stdout.strip().splitlines()
                    for line in lines[1:]:  # skip header: NAME ID SIZE MODIFIED
                        parts = line.split()
                        if len(parts) >= 3:
                            name = parts[0]
                            size_str = parts[2] if len(parts) > 2 else ""
                            models.append({
                                "name": name,
                                "size_str": size_str,
                            })
            except Exception:
                pass

        return models

    def query_model(self, model_name: str, prompt: str, system: Optional[str] = None, timeout: int = 60) -> str:
        """
        Executes a local inference pass through the local Ollama model (no internet needed).
        """
        # Try HTTP API
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
            }
            if system:
                payload["system"] = system

            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.api_base}/api/generate",
                data=data_bytes,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception:
            pass

        # Fallback to CLI: 'ollama run <model> <prompt>'
        if self.ollama_bin:
            try:
                res = subprocess.run(
                    [self.ollama_bin, "run", model_name, prompt],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
                if res.returncode == 0:
                    return res.stdout.strip()
            except Exception:
                pass

        return ""

    def register_model_with_modelfile(self, model_name: str, modelfile_path: Path) -> bool:
        """
        Registers an uncensored GGUF model into Ollama via 'ollama create <name> -f <modelfile>'.
        """
        if not self.ollama_bin:
            print("⚠️ Ollama executable not found. Please install Ollama or add to PATH.")
            return False

        print(f"📦 Registering '{model_name}' into Ollama with Modelfile: {modelfile_path}...")
        try:
            res = subprocess.run(
                [self.ollama_bin, "create", model_name, "-f", str(modelfile_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if res.returncode == 0:
                print(f"🎉 Successfully created local Ollama model: {model_name}!")
                print(f"   You can now run: 'ollama run {model_name}' in any terminal or AI IDE!")
                return True
            else:
                print(f"⚠️ Ollama create warning: {res.stderr.strip()}")
                return False
        except Exception as e:
            print(f"❌ Error creating Ollama model: {e}")
            return False
