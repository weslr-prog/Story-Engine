#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PHASE_FILE_CANDIDATES = {
    "phase1": ["Phase 1 - Story DNA Summary.txt", "Story DNA Summary.txt", "Story DNA.txt"],
    "phase2": ["Phase 2 - Story Bible.txt", "Story Bible.txt"],
    "phase3": ["Phase 3 - Chapter Blueprint.txt", "Chapter Blueprint.txt"],
    "phase4": ["Phase 4 - Writing Prompts.txt", "Writing Prompts.txt", "style_guide.txt"],
}

REQUIRED_PHASE3_LABELS = [
    "Word target",
    "POV",
    "SCENE ZERO",
    "SCENE BREAKDOWN",
    "CHARACTER BEAT",
    "ACTION BEAT",
    "EMOTIONAL BEAT",
    "INTERIORITY BEAT",
    "CLIFFHANGER",
]


def find_first_existing(base_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = base_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def validate_non_empty(label: str, path: Path | None, errors: list[str], notes: list[str]) -> None:
    if path is None:
        errors.append(f"Missing required file for {label}.")
        return
    text = read_text(path).strip()
    if not text:
        errors.append(f"{path.name} is empty ({label}).")
        return
    notes.append(f"[OK] {label}: {path.name}")


def validate_phase3_labels(path: Path, errors: list[str], notes: list[str]) -> None:
    text = read_text(path)
    missing: list[str] = []
    for label in REQUIRED_PHASE3_LABELS:
        pattern = re.compile(rf"(^|\n)\s*{re.escape(label)}\s*:?", re.IGNORECASE)
        if not pattern.search(text):
            missing.append(label)
    if missing:
        errors.append(
            "Phase 3 label check failed. Missing labels: " + ", ".join(missing)
        )
    else:
        notes.append("[OK] Phase 3 labels: all required labels found")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Phase 1-4 prompt output docs before Story-Engine conversion."
    )
    parser.add_argument(
        "--source-dir",
        default="The Last Signal",
        help="Directory that contains phase output text files (default: ./The Last Signal)",
    )
    parser.add_argument(
        "--require-phase4",
        action="store_true",
        help="Treat Phase 4 writing prompts/style guide as required.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"[ERROR] Source directory not found: {source_dir}")
        return 2

    errors: list[str] = []
    notes: list[str] = []

    phase1 = find_first_existing(source_dir, PHASE_FILE_CANDIDATES["phase1"])
    phase2 = find_first_existing(source_dir, PHASE_FILE_CANDIDATES["phase2"])
    phase3 = find_first_existing(source_dir, PHASE_FILE_CANDIDATES["phase3"])
    phase4 = find_first_existing(source_dir, PHASE_FILE_CANDIDATES["phase4"])

    validate_non_empty("Phase 1", phase1, errors, notes)
    validate_non_empty("Phase 2", phase2, errors, notes)
    validate_non_empty("Phase 3", phase3, errors, notes)

    if args.require_phase4:
        validate_non_empty("Phase 4", phase4, errors, notes)
    elif phase4 is not None and read_text(phase4).strip():
        notes.append(f"[OK] Phase 4: {phase4.name}")
    else:
        notes.append("[WARN] Phase 4 is optional for this check; no file found")

    if phase3 is not None and phase3.exists():
        validate_phase3_labels(phase3, errors, notes)

    print("Phase docs validation report")
    print(f"source_dir: {source_dir}")
    print()
    for line in notes:
        print(line)

    if errors:
        print()
        print("[FAIL] Validation errors:")
        for item in errors:
            print(f"- {item}")
        return 1

    print()
    print("[PASS] Phase docs look compatible with Story-Engine conversion expectations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
