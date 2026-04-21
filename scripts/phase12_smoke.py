#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.inference_router import InferenceRouter
from engine.local_llm import HypuraClient, OllamaClient


def _safe_chat(fn):
    try:
        return {"ok": True, "response": fn()[:200]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    structural = OllamaClient()
    prose = HypuraClient()
    router = InferenceRouter(structural=structural, prose=prose, fallback=OllamaClient())

    result = {
        "structural_ping": structural.ping(),
        "prose_ping": prose.ping(),
        "structural_chat": _safe_chat(lambda: router.route_to_structural("Return one short sentence: structural lane online.", max_tokens=32, temperature=0.1)),
        "prose_chat": _safe_chat(lambda: router.route_to_prose("Return one short sentence: prose lane online.", max_tokens=32, temperature=0.3)),
        "pipeline_entrypoint": (ROOT / "pipeline_novel.py").exists(),
    }

    print(json.dumps(result, indent=2))
    ok = result["structural_ping"] and result["prose_ping"] and result["pipeline_entrypoint"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
