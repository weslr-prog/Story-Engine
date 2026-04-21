#!/usr/bin/env python3
"""Phase 1 validation test."""
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
import shutil

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

from config import SETTINGS

def main():
    print("\n=== PHASE 1 VALIDATION ===\n")
    
    checks = []
    
    # Check FFmpeg
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    checks.append(("FFmpeg", ffmpeg_ok))
    
    # Check Ollama service
    try:
        resp = requests.get('http://127.0.0.1:11434/api/tags', timeout=5)
        ollama_ok = resp.ok
        checks.append(("Ollama service", ollama_ok))
    except:
        checks.append(("Ollama service", False))
    
    # Check model
    try:
        payload = {
            "model": SETTINGS.ollama_model,
            "temperature": 0.0,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "ok"}],
            "stream": False,
        }
        resp = requests.post(SETTINGS.ollama_url, json=payload, timeout=30)
        model_ok = resp.ok
        checks.append((f"Ollama model ({SETTINGS.ollama_model})", model_ok))
    except Exception as e:
        checks.append(("Ollama model test", False))
    
    # Check ChromaDB
    chroma_ok = Path(SETTINGS.chroma_db_path).exists()
    checks.append(("ChromaDB path", chroma_ok))
    
    # Check M.2
    models_ok = Path(SETTINGS.hypura_models_dir).exists()
    checks.append(("M.2 models directory", models_ok))
    
    # Check venv
    venv_ok = Path(ROOT / '.venv/bin/activate').exists()
    checks.append(("Virtual environment", venv_ok))
    
    # Print results
    for name, ok in checks:
        print(f"{name:.<35} {'✅' if ok else '❌'}")
    
    all_ok = all(ok for _, ok in checks)
    print(f"\n{'='*40}")
    print(f"PHASE 1 STATUS: {'✅ COMPLETE' if all_ok else '⚠️  NEEDS WORK'}")
    print(f"{'='*40}\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
