#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = Path(os.getenv("HYPURA_MODELS_DIR", "/Volumes/256 M.2/story-engine-models"))


def _find_llama_bin(name: str) -> Path | None:
    candidates = [
        ROOT / "third_party" / "hypura" / "target" / "release" / "build",
        ROOT / "third_party" / "hypura" / "target" / "debug" / "build",
    ]
    for base in candidates:
        if not base.exists():
            continue
        for out in base.glob("hypura-sys-*/out/bin"):
            p = out / name
            if p.exists():
                return p
    found = shutil.which(name)
    return Path(found) if found else None


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip()


def smoke(models_dir: Path) -> dict:
    report = {
        "models_dir": str(models_dir),
        "has_models_dir": models_dir.exists() and models_dir.is_dir(),
        "llama_imatrix": "",
        "llama_quantize": "",
        "llama_perplexity": "",
        "gguf_count": 0,
        "status": "",
    }
    for tool in ("llama-imatrix", "llama-quantize", "llama-perplexity"):
        p = _find_llama_bin(tool)
        report[tool.replace("-", "_")] = str(p) if p else ""

    ggufs = list(models_dir.rglob("*.gguf")) if models_dir.exists() else []
    report["gguf_count"] = len(ggufs)

    missing = [k for k in ("llama_imatrix", "llama_quantize") if not report[k]]
    if missing:
        report["status"] = f"missing tools: {', '.join(missing)}"
        return report

    report["status"] = "ready"
    return report


def write_plan(models_dir: Path, input_model: Path, output_model: Path, calibration: Path) -> dict:
    imatrix = _find_llama_bin("llama-imatrix")
    quantize = _find_llama_bin("llama-quantize")
    perplexity = _find_llama_bin("llama-perplexity")

    if not imatrix or not quantize:
        raise RuntimeError("llama-imatrix/llama-quantize not found. Build third_party/hypura first.")

    plan = {
        "models_dir": str(models_dir),
        "input_model": str(input_model),
        "output_model": str(output_model),
        "calibration": str(calibration),
        "commands": [
            [str(imatrix), "-m", str(input_model), "-f", str(calibration), "-o", str(output_model) + ".imatrix.dat", "--chunks", "64", "-ngl", "99"],
            [str(quantize), "--imatrix", str(output_model) + ".imatrix.dat", str(input_model), str(output_model), "Q4_K_M"],
        ],
    }
    if perplexity:
        plan["commands"].append([str(perplexity), "-m", str(output_model), "-f", str(calibration), "-ngl", "99"])
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 TurboQuant helper")
    parser.add_argument("--smoke", action="store_true", help="Run readiness smoke check only")
    parser.add_argument("--write-plan", action="store_true", help="Write quantization command plan JSON")
    parser.add_argument("--input-model", default="", help="Input GGUF model path")
    parser.add_argument("--output-model", default="", help="Output GGUF model path")
    parser.add_argument("--calibration", default="", help="Calibration text file path")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--plan-file", default=str(ROOT / "docs" / "implementation" / "phase3_turboquant_plan.json"))
    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    if args.smoke:
        print(json.dumps(smoke(models_dir), indent=2))
        return 0

    if args.write_plan:
        if not args.input_model or not args.output_model or not args.calibration:
            raise SystemExit("--input-model, --output-model, and --calibration are required with --write-plan")
        plan = write_plan(
            models_dir,
            Path(args.input_model),
            Path(args.output_model),
            Path(args.calibration),
        )
        out = Path(args.plan_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote plan: {out}")
        return 0

    raise SystemExit("Use --smoke or --write-plan")


if __name__ == "__main__":
    raise SystemExit(main())
