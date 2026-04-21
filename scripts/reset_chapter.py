#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _chapter_tag(chapter: int) -> str:
    if chapter < 1:
        raise ValueError("--chapter must be >= 1")
    return f"ch{chapter:02d}"


def _targets(ch: str) -> list[Path]:
    return [
        ROOT / "chapters" / f"{ch}_draft.txt",
        ROOT / "chapters" / f"{ch}_edited.txt",
        ROOT / "chapters" / f"{ch}_final.txt",
        ROOT / "chapters" / f"{ch}_tts.txt",
        ROOT / "chapters" / "scenes" / ch,
        ROOT / "summaries" / f"{ch}_summary.txt",
        ROOT / "audio" / f"{ch}_narration.wav",
        ROOT / "audio" / "segments" / ch,
        ROOT / "reviews" / f"{ch}_local_critic.md",
        ROOT / "reviews" / f"{ch}_external_critic.md",
        ROOT / "reviews" / f"{ch}_external_critic_prompt.md",
        ROOT / "reviews" / f"{ch}_edited_for_external.txt",
        ROOT / "reviews" / f"{ch}_pre_narration_review.md",
        ROOT / "reviews" / f"{ch}_post_chapter_review.md",
        ROOT / "reviews" / f"{ch}_pre_narration.approved",
        ROOT / "reviews" / f"{ch}_post_chapter.approved",
    ]


def _delete(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        print(f"SKIP missing: {path.relative_to(ROOT)}")
        return False

    if path.is_dir():
        print(f"DELETE dir: {path.relative_to(ROOT)}")
        if not dry_run:
            shutil.rmtree(path)
        return True

    print(f"DELETE file: {path.relative_to(ROOT)}")
    if not dry_run:
        path.unlink()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete generated artifacts for one chapter.")
    parser.add_argument("--chapter", type=int, required=True, help="Chapter number, e.g. 1 for ch01")
    parser.add_argument("--dry-run", action="store_true", help="Show deletions without deleting files")
    args = parser.parse_args()

    ch = _chapter_tag(args.chapter)
    print(f"Chapter reset target: {ch}")

    deleted = 0
    for p in _targets(ch):
        if _delete(p, args.dry_run):
            deleted += 1

    if args.dry_run:
        print(f"Dry run complete. {deleted} items would be removed.")
    else:
        print(f"Reset complete. {deleted} items removed.")


if __name__ == "__main__":
    main()
