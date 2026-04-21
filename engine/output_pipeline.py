from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def stitch_chapter(output_dir: Path, chapter_number: int, scenes: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    chapter_path = output_dir / f"ch{chapter_number:02d}.md"
    chapter_path.write_text("\n\n".join(scenes).strip() + "\n", encoding="utf-8")
    return chapter_path


def stitch_novel(output_dir: Path, chapter_paths: list[Path]) -> Path:
    novel_path = output_dir / "manuscript.md"
    content = []
    for chapter_path in chapter_paths:
        content.append(chapter_path.read_text(encoding="utf-8").strip())
    novel_path.write_text("\n\n".join(part for part in content if part) + "\n", encoding="utf-8")
    return novel_path


def export_manuscript(manuscript_path: Path, export_format: str = "md") -> Path:
    export_format = export_format.lower().strip()
    if export_format == "md":
        return manuscript_path

    if export_format not in {"docx", "epub"}:
        raise ValueError(f"Unsupported export format: {export_format}")

    out = manuscript_path.with_suffix(f".{export_format}")
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError("pandoc is required for docx/epub export")

    subprocess.run(
        [pandoc, str(manuscript_path), "-o", str(out)],
        check=True,
        capture_output=True,
    )
    return out
