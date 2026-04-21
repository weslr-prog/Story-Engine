#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from convert_story_engine import Inputs, convert_rule
from validate_phase_docs import PHASE_FILE_CANDIDATES, find_first_existing, read_text, validate_phase3_labels


def resolve_phase_files(source_dir: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for phase, names in PHASE_FILE_CANDIDATES.items():
        path = find_first_existing(source_dir, names)
        if path is not None:
            resolved[phase] = path
    return resolved


def validate_required(resolved: dict[str, Path], require_phase4: bool) -> list[str]:
    errors: list[str] = []
    for key in ["phase1", "phase2", "phase3"]:
        if key not in resolved:
            errors.append(f"Missing {key} source file.")
    if require_phase4 and "phase4" not in resolved:
        errors.append("Missing phase4 source file.")

    if "phase3" in resolved:
        notes: list[str] = []
        validate_phase3_labels(resolved["phase3"], errors, notes)

    for key, path in resolved.items():
        if not read_text(path).strip():
            errors.append(f"{path.name} is empty ({key}).")
    return errors


def copy_inputs(resolved: dict[str, Path], input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "phase1": "Story DNA Summary.txt",
        "phase2": "Story Bible.txt",
        "phase3": "Chapter Blueprint.txt",
        "phase4": "style_guide.txt",
    }
    for phase, target_name in mapping.items():
        src = resolved.get(phase)
        if src is None:
            continue
        shutil.copyfile(src, input_dir / target_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest and convert Phase 1-4 prompt outputs into Story-Engine artifacts.")
    parser.add_argument("--source-dir", required=True, help="Directory containing phase output text files.")
    parser.add_argument("--out-dir", default=".", help="Directory where converted JSON and guides are written.")
    parser.add_argument(
        "--project-input-dir",
        default="",
        help="Optional directory to copy normalized source docs (Story DNA Summary.txt, Story Bible.txt, Chapter Blueprint.txt, style_guide.txt).",
    )
    parser.add_argument("--require-phase4", action="store_true", help="Require phase 4 file to exist.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    project_input_dir = Path(args.project_input_dir).resolve() if args.project_input_dir else None

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"[ERROR] Source directory not found: {source_dir}")
        return 2

    resolved = resolve_phase_files(source_dir)
    errors = validate_required(resolved, args.require_phase4)
    if errors:
        print("[FAIL] Ingest validation failed:")
        for item in errors:
            print(f"- {item}")
        return 1

    if project_input_dir is not None:
        copy_inputs(resolved, project_input_dir)
        print(f"[OK] Normalized phase docs copied to: {project_input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    convert_rule(
        Inputs(
            dna=resolved["phase1"],
            bible=resolved["phase2"],
            blueprint=resolved["phase3"],
            out_dir=out_dir,
        )
    )

    if "phase4" in resolved:
        shutil.copyfile(resolved["phase4"], out_dir / "style_guide.txt")
        print("[OK] style_guide.txt overridden from phase 4 writing prompts")

    print(f"[OK] story_bible.json -> {out_dir / 'story_bible.json'}")
    print(f"[OK] characters.json -> {out_dir / 'characters.json'}")
    print(f"[OK] chapter_briefs.json -> {out_dir / 'chapter_briefs.json'}")
    print(f"[OK] consistency_checklist.txt -> {out_dir / 'consistency_checklist.txt'}")
    print(f"[OK] master_system_prompt.md -> {out_dir / 'master_system_prompt.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
