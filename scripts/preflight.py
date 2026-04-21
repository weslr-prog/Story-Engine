import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from config import SETTINGS  # noqa: E402
from engine.chatterbox_http import discover_chatterbox  # noqa: E402


def check_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _root_url(target: str) -> str:
    parts = urlsplit(target)
    if not parts.scheme or not parts.netloc:
        return target
    return f"{parts.scheme}://{parts.netloc}"


def check_ollama() -> tuple[bool, str]:
    try:
        resp = requests.get(f"{_root_url(SETTINGS.ollama_url)}/api/tags", timeout=4)
        if not resp.ok:
            return False, f"HTTP {resp.status_code}"
        return True, "reachable"
    except Exception as exc:
        return False, str(exc)


def check_hypura() -> tuple[bool, str]:
    try:
        resp = requests.get(_root_url(SETTINGS.hypura_url), timeout=4)
        return resp.ok, f"HTTP {resp.status_code}"
    except Exception as exc:
        return False, str(exc)


def hypura_model_inventory() -> tuple[bool, dict]:
    endpoint = f"{_root_url(SETTINGS.hypura_url)}/api/tags"
    try:
        resp = requests.get(endpoint, timeout=6)
        if not resp.ok:
            return False, {"endpoint": endpoint, "error": f"HTTP {resp.status_code}", "models": []}
        payload = resp.json()
        models = [m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)]
        return True, {
            "endpoint": endpoint,
            "models": models,
            "configured_model": SETTINGS.hypura_model,
            "configured_present": SETTINGS.hypura_model in models,
        }
    except Exception as exc:
        return False, {"endpoint": endpoint, "error": str(exc), "models": []}


def check_chatterbox() -> tuple[bool, str]:
    try:
        resp = requests.get(SETTINGS.chatterbox_url, timeout=4)
        return resp.ok, f"HTTP {resp.status_code}"
    except Exception as exc:
        return False, str(exc)


def check_local_disk_kv() -> tuple[bool, str]:
    target = SETTINGS.ollama_url.strip()
    try:
        parts = urlsplit(target)
        if not parts.scheme or not parts.netloc:
            return False, "invalid OLLAMA_URL"
        probe = f"{parts.scheme}://{parts.netloc}/"
        resp = requests.get(probe, timeout=4)
        return True, f"{probe} HTTP {resp.status_code}"
    except Exception as exc:
        return False, str(exc)


def probe_local_disk_kv_model() -> tuple[bool, str]:
    probe_timeout = int(os.getenv("PREFLIGHT_PROBE_TIMEOUT", "120"))
    payload = {
        "model": SETTINGS.ollama_model,
        "temperature": 0.0,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    try:
        resp = requests.post(SETTINGS.ollama_url, json=payload, timeout=probe_timeout)
        if not resp.ok:
            body = (resp.text or "").strip().replace("\n", " ")
            return False, f"HTTP {resp.status_code} {body[:220]}"
        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "") or msg.get("reasoning", "")
        if isinstance(content, str):
            return True, f"ok ({SETTINGS.ollama_model})"
        return True, f"ok ({SETTINGS.ollama_model})"
    except Exception as exc:
        return False, str(exc)


def check_chroma() -> tuple[bool, str]:
    try:
        path = Path(SETTINGS.chroma_db_path)
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return True, str(path)
    except Exception as exc:
        return False, str(exc)


def check_models_dir() -> tuple[bool, str]:
    path = Path(SETTINGS.hypura_models_dir)
    if not path.exists():
        return False, f"missing: {path}"
    if not path.is_dir():
        return False, f"not a directory: {path}"
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    return True, f"{path} free={free_gb:.1f}GB"


def discover_api_names() -> list[str]:
    try:
        discovery = discover_chatterbox(SETTINGS.chatterbox_url)
        return discovery.get("endpoints", [])
    except Exception:
        return []


def check_ffmpeg() -> tuple[bool, str]:
    if not check_binary("ffmpeg"):
        return False, "not found in PATH"
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True
        )
        first_line = proc.stdout.splitlines()[0] if proc.stdout else "ok"
        return True, first_line
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    report = {
        "python": sys.version.split()[0],
        "ollama_model": SETTINGS.ollama_model,
        "hypura_model": SETTINGS.hypura_model,
        "ffmpeg": {},
        "ollama": {},
        "hypura": {},
        "hypura_inventory": {},
        "kv_confidence": "configured-only",
        "local_disk_kv": {},
        "local_disk_kv_model_probe": {},
        "chroma": {},
        "models_dir": {},
        "chatterbox_webui": {},
        "gradio_endpoints": [],
        "next_action": "",
    }

    ffmpeg_ok, ffmpeg_msg = check_ffmpeg()
    report["ffmpeg"] = {"ok": ffmpeg_ok, "detail": ffmpeg_msg}

    ollama_ok, ollama_msg = check_ollama()
    report["ollama"] = {"ok": ollama_ok, "detail": ollama_msg}

    hypura_ok, hypura_msg = check_hypura()
    report["hypura"] = {"ok": hypura_ok, "detail": hypura_msg}
    inv_ok, inv = hypura_model_inventory()
    report["hypura_inventory"] = {"ok": inv_ok, "detail": inv}
    if inv_ok and inv.get("configured_present"):
        report["kv_confidence"] = "partially-verified"

    local_kv_ok, local_kv_msg = check_local_disk_kv()
    report["local_disk_kv"] = {"ok": local_kv_ok, "detail": local_kv_msg}
    if local_kv_ok:
        model_ok, model_msg = probe_local_disk_kv_model()
        report["local_disk_kv_model_probe"] = {"ok": model_ok, "detail": model_msg}
    else:
        report["local_disk_kv_model_probe"] = {"ok": False, "detail": "skipped"}

    chroma_ok, chroma_msg = check_chroma()
    report["chroma"] = {"ok": chroma_ok, "detail": chroma_msg}

    models_ok, models_msg = check_models_dir()
    report["models_dir"] = {"ok": models_ok, "detail": models_msg}

    chatterbox_ok, chatterbox_msg = check_chatterbox()
    report["chatterbox_webui"] = {"ok": chatterbox_ok, "detail": chatterbox_msg}

    if chatterbox_ok:
        report["gradio_endpoints"] = discover_api_names()

    if not ffmpeg_ok:
        report["next_action"] = "Install ffmpeg before running TTS stitching."
    elif not models_ok:
        report["next_action"] = "Set HYPURA_MODELS_DIR to the external M.2 model directory, then rerun preflight."
    elif not ollama_ok:
        report["next_action"] = "Start Ollama and pull the Phi-3.5 Mini model before running the engine."
    elif not report["local_disk_kv_model_probe"].get("ok", False):
        report["next_action"] = "Ollama is reachable, but the configured structural model did not answer. Pull the model and rerun preflight."
    elif not hypura_ok:
        report["next_action"] = "Build or start Hypura on port 11435 before attempting prose generation."
    elif not chroma_ok:
        report["next_action"] = "Fix CHROMA_DB_PATH permissions so long-term memory can initialize."
    elif not chatterbox_ok:
        report["next_action"] = "Chatterbox is still optional at this stage. Start it later when you reach Phase 13."
    elif not report["gradio_endpoints"]:
        report["next_action"] = "Core stack is ready. Discover Chatterbox endpoints later during Phase 13."
    else:
        report["next_action"] = "Core stack checks passed. Continue to database, RAG, and orchestrator smoke tests."

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
