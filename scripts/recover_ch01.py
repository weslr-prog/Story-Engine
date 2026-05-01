#!/usr/bin/env python3
"""
Surgical recovery for chapter 1 after lint-fail checkpoint.

Picks up from the already-generated ch01_edited.txt, runs lint with the
fixed LintSettings, writes final/tts/summary artifacts, narrates the
chapter, and updates the checkpoint. No inference re-runs.

Usage:
    python scripts/recover_ch01.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from config import SETTINGS
from engine.agents import (
    MemoryManagerAgent,
    _extract_prose_only,
)
from engine.inference_router import InferenceRouter
from engine.local_llm import HypuraClient, OllamaClient
from engine.orchestrator import PipelineContext
from engine.output_pipeline import stitch_chapter
from engine.story_lint import LintSettings, lint_chapter, to_markdown
from engine.tts_engine import narrate_chapter
from pipeline_novel import (  # noqa: E402
    _cap_repeated_sentences,
    _chapter_artifacts,
    _load_json,
    _save_json,
    _with_chapter_heading,
    _target_min_words,
    _word_count,
    _write,
    _save_checkpoint,
)


def main() -> int:
    chapter_num = 1
    project_name = "The-Last-Signal"

    artifacts = _chapter_artifacts(chapter_num)

    if not artifacts.edited.exists():
        print(f"[FAIL] ch01_edited.txt not found at {artifacts.edited}")
        return 1

    briefs = _load_json(ROOT / "chapter_briefs.json", [])
    if not briefs:
        print("[FAIL] chapter_briefs.json missing or empty")
        return 1

    brief = briefs[chapter_num - 1]

    # --- 1. Read existing edited prose ---
    edited_text = artifacts.edited.read_text(encoding="utf-8")
    print(f"[INFO] Edited prose loaded: {_word_count(edited_text)} words")

    # --- 2. Cap repeated sentences + lint ---
    lint_settings = LintSettings()
    final_text = _cap_repeated_sentences(edited_text, lint_settings.max_sentence_repeat)
    final_text = _with_chapter_heading(chapter_num, brief, final_text)

    chapter_min_words = _target_min_words(brief)
    chapter_words = _word_count(final_text)
    print(f"[INFO] Word count: {chapter_words} (min: {chapter_min_words})")

    if chapter_words < chapter_min_words:
        print(f"[FAIL] Chapter still undersized: {chapter_words} < {chapter_min_words}")
        return 1

    lint_report = lint_chapter(final_text, chapter_num, brief, lint_settings)
    _save_json(artifacts.lint_json, lint_report)
    _write(artifacts.lint_md, to_markdown(lint_report))

    if not lint_report.get("passed", False):
        print(f"[FAIL] Lint still failing after recovery patch:")
        for c in lint_report["checks"]:
            if not c["passed"]:
                print(f"  [{c['name']}] {c['violations']}")
        return 1

    print("[OK] Lint passed")
    for c in lint_report["checks"]:
        print(f"  ✅ {c['name']}")

    # --- 3. Write final and tts artifacts ---
    _write(artifacts.final, final_text)
    print(f"[OK] Written: {artifacts.final.relative_to(ROOT)}")

    tts_text = _extract_prose_only(final_text)
    if not tts_text:
        tts_text = final_text
    _write(artifacts.tts, tts_text)
    tts_words = _word_count(tts_text)
    est_mins = tts_words / 150
    print(f"[OK] Written TTS text: {artifacts.tts.relative_to(ROOT)} ({tts_words} words, ~{est_mins:.1f} min est.)")

    # --- 4. Summary via MemoryManagerAgent ---
    print("[INFO] Building chapter summary via MemoryManagerAgent...")
    try:
        router = InferenceRouter(
            prose_client=HypuraClient(
                model=SETTINGS.hypura_model,
                base_url=SETTINGS.hypura_url,
                timeout=int(SETTINGS.llm_call_timeout_seconds),
                num_ctx=SETTINGS.llm_num_ctx,
            ),
            structural_client=OllamaClient(
                model=SETTINGS.ollama_model,
                base_url=SETTINGS.ollama_url,
                timeout=int(SETTINGS.llm_call_timeout_seconds),
                num_ctx=SETTINGS.llm_num_ctx,
            ),
        )
        memory_manager = MemoryManagerAgent(router)
        context = PipelineContext(project_name=project_name, current_chapter=chapter_num)
        summary = memory_manager.run(
            context,
            {
                "task": "Summarize chapter in 120-180 words with unresolved threads.",
                "chapter_text": final_text,
                "brief": brief,
            },
        ).content
    except Exception as exc:
        print(f"[WARN] Summary inference failed ({exc}); using fallback summary")
        summary = (
            f"Chapter {chapter_num}: Aris discovers an anomalous signal on channel 734-Theta. "
            "Despite Commander Vann's orders to log and ignore it, he returns to listen again. "
            "Unresolved: the signal's origin and intentional structure remain unexplained."
        )

    _write(artifacts.summary, summary)
    print(f"[OK] Written: {artifacts.summary.relative_to(ROOT)}")

    # --- 5. Stitch chapter for manuscript ---
    stitch_chapter(ROOT / "chapters", chapter_num, [artifacts.final.read_text(encoding="utf-8")])
    print(f"[OK] Chapter stitched into chapters/ directory")

    # --- 6. Narrate ---
    voice_sample = SETTINGS.voice_sample or str(ROOT / "voices" / "p233_023.wav")
    if Path(voice_sample).exists():
        print(f"[INFO] Narrating chapter 1 with voice sample: {voice_sample}")
        narrate_chapter(
            text=tts_text,
            voice_sample=voice_sample,
            output_path=str(artifacts.audio),
            chapter_num=chapter_num,
            resume=True,
        )
        if artifacts.audio.exists():
            size_mb = artifacts.audio.stat().st_size / (1024 * 1024)
            print(f"[OK] Narration written: {artifacts.audio.relative_to(ROOT)} ({size_mb:.1f} MB)")
            _save_checkpoint(project_name, 2, "complete", reason="chapter_complete")
            print(f"[OK] Checkpoint updated: next_chapter=2, state=complete")
        else:
            print(f"[WARN] Audio file not found after narration; checkpoint not advanced")
    else:
        print(f"[WARN] Voice sample not found: {voice_sample}")
        print(f"[INFO] Skipping narration; updating checkpoint to next chapter")
        _save_checkpoint(project_name, 2, "complete", reason="chapter_complete_no_audio")

    print()
    print("=== Chapter 1 Recovery Complete ===")
    print(f"  Final prose:    {artifacts.final.relative_to(ROOT)} ({_word_count(final_text)} words)")
    print(f"  TTS text:       {artifacts.tts.relative_to(ROOT)} ({tts_words} words, ~{est_mins:.1f} min narration)")
    print(f"  Summary:        {artifacts.summary.relative_to(ROOT)}")
    print(f"  Lint:           ✅ passed")
    if artifacts.audio.exists():
        print(f"  Narration WAV:  {artifacts.audio.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
