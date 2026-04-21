from __future__ import annotations

import argparse
import os
from pathlib import Path

from engine.config import SETTINGS
from engine.output_pipeline import export_manuscript
from engine.tts_engine import narrate_chapter
from pipeline_novel import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Story-Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a project pipeline")
    run_parser.add_argument("--project", required=True)
    run_parser.add_argument("--chapter-limit", type=int)
    run_parser.add_argument("--dry-run", action="store_true")

    resume_parser = subparsers.add_parser("resume", help="Resume a project pipeline")
    resume_parser.add_argument("--project", required=True)

    export_parser = subparsers.add_parser("export", help="Export generated outputs")
    export_parser.add_argument("--project", required=True)
    export_parser.add_argument("--format", choices=["epub", "docx", "md"], default="md")

    narrate_parser = subparsers.add_parser("narrate", help="Narrate a chapter once TTS is enabled")
    narrate_parser.add_argument("--project", required=True)
    narrate_parser.add_argument("--chapter", type=int, required=True)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print(f"Story-Engine loaded from {Path.cwd()}")
    print(f"Projects root: {SETTINGS.projects_root}")

    if args.command in {"run", "resume"}:
        if args.chapter_limit:
            os.environ["CHAPTER_LAST"] = str(int(args.chapter_limit))
        os.environ["PROJECT_NAME"] = str(args.project)
        return run_pipeline(project_name=args.project, dry_run=getattr(args, "dry_run", False))

    if args.command == "export":
        manuscript = Path("chapters") / "manuscript.md"
        if not manuscript.exists():
            print("No manuscript found. Run pipeline first.")
            return 1
        out = export_manuscript(manuscript, args.format)
        print(f"Exported: {out}")
        return 0

    if args.command == "narrate":
        ch = int(args.chapter)
        tts_path = Path("chapters") / f"ch{ch:02d}_tts.txt"
        audio_path = Path("audio") / f"ch{ch:02d}_narration.wav"
        if not tts_path.exists():
            print(f"Missing TTS text: {tts_path}")
            return 1
        voice = SETTINGS.voice_sample or str(Path("voices") / "p233_023.wav")
        narrate_chapter(tts_path.read_text(encoding="utf-8"), voice, str(audio_path), chapter_num=ch)
        print(f"Narration complete: {audio_path}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
