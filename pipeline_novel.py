#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
import gc
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from config import SETTINGS
from engine.agents import ArchitectAgent, EditorAgent, MemoryManagerAgent, PlannerAgent, WriterAgent
from engine.genre_pack import load_genre_pack
from engine.inference_router import InferenceRouter
from engine.local_llm import HypuraClient, OllamaClient
from engine.memory_monitor import MemoryAction, MemoryMonitor
from engine.orchestrator import FSMState, Orchestrator, PipelineContext
from engine.output_pipeline import export_manuscript, stitch_chapter, stitch_novel
from engine.rag_memory import StoryMemory
from engine.story_bible_db import Project, WorldFact, make_session_factory
from engine.story_lint import LintSettings, lint_chapter, to_markdown
from engine.tts_engine import narrate_chapter


@dataclass
class ChapterArtifacts:
    draft: Path
    edited: Path
    final: Path
    tts: Path
    summary: Path
    audio: Path
    lint_json: Path
    lint_md: Path


def _mkdirs() -> None:
    for rel in [
        "chapters",
        "chapters/scenes",
        "summaries",
        "audio",
        "audio/segments",
        SETTINGS.reviews_dir,
        SETTINGS.checkpoint_dir,
        SETTINGS.diagnostics_dir,
        SETTINGS.diagnostics_dir / "runs",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seed_story_inputs(chapter_count: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    story_bible_path = ROOT / "story_bible.json"
    characters_path = ROOT / "characters.json"
    briefs_path = ROOT / "chapter_briefs.json"

    if not story_bible_path.exists():
        _save_json(
            story_bible_path,
            {
                "title": "Glass Meridian",
                "logline": "A forensic archivist uncovers a city-wide memory rewrite tied to her vanished sister.",
                "themes": ["truth", "identity", "cost of certainty"],
                "world_facts": [
                    "The city of Meridian records civic memory snapshots nightly.",
                    "Unauthorized memory edits are treated as terrorism.",
                ],
            },
        )

    if not characters_path.exists():
        _save_json(
            characters_path,
            [
                {
                    "name": "Mara Quill",
                    "role": "protagonist",
                    "core_wound": "She failed to protect her sister during a prior data purge.",
                    "flawed_belief": "Control is safer than trust.",
                    "voice_style": "precise, emotionally guarded",
                },
                {
                    "name": "Ivo Vale",
                    "role": "ally",
                    "core_wound": "Expelled for exposing corruption.",
                    "flawed_belief": "Truth alone saves people.",
                    "voice_style": "dry wit, blunt honesty",
                },
            ],
        )

    if not briefs_path.exists():
        briefs = []
        for ch in range(1, chapter_count + 1):
            briefs.append(
                {
                    "chapter": ch,
                    "title": f"Chapter {ch}: Turning Point",
                    "goal": "Advance the central mystery while forcing a character choice.",
                    "key_events": [
                        "Mara receives contradictory evidence about her sister.",
                        "Ivo reveals a risky lead.",
                        "A decision closes one path and opens a costlier one.",
                    ],
                    "cliffhanger": "End on a concrete irreversible decision.",
                }
            )
        _save_json(briefs_path, briefs)

    return (
        _load_json(story_bible_path, {}),
        _load_json(characters_path, []),
        _load_json(briefs_path, []),
    )


def _chapter_artifacts(chapter_num: int) -> ChapterArtifacts:
    ch = f"ch{chapter_num:02d}"
    return ChapterArtifacts(
        draft=ROOT / "chapters" / f"{ch}_draft.txt",
        edited=ROOT / "chapters" / f"{ch}_edited.txt",
        final=ROOT / "chapters" / f"{ch}_final.txt",
        tts=ROOT / "chapters" / f"{ch}_tts.txt",
        summary=ROOT / "summaries" / f"{ch}_summary.txt",
        audio=ROOT / "audio" / f"{ch}_narration.wav",
        lint_json=ROOT / SETTINGS.reviews_dir / f"{ch}_lint.json",
        lint_md=ROOT / SETTINGS.reviews_dir / f"{ch}_lint.md",
    )


def _pre_narration_review_path(chapter_num: int) -> Path:
    return ROOT / SETTINGS.reviews_dir / f"ch{chapter_num:02d}_pre_narration_review.md"


def _pre_narration_marker_path(chapter_num: int) -> Path:
    return ROOT / SETTINGS.reviews_dir / f"ch{chapter_num:02d}_pre_narration.approved"


def _write_pre_narration_review(
    chapter_num: int,
    brief: dict[str, Any],
    artifacts: ChapterArtifacts,
    voice_sample: str,
) -> Path:
    preview_sentences = [part.strip() for part in artifacts.tts.read_text(encoding="utf-8").split(".") if part.strip()]
    preview = ". ".join(preview_sentences[:3]).strip()
    if preview and not preview.endswith("."):
        preview += "."

    review_text = f"""# Pre-Narration Review: Chapter {chapter_num:02d}

## Goal
- Tune narration voice in the repo-local Gradio UI before batch narration runs.

## Chapter
- Title: {brief.get('title') or brief.get('goal') or f'Chapter {chapter_num}'}
- Voice sample: {voice_sample}
- TTS text file: {artifacts.tts}
- Target narration file: {artifacts.audio}

## Launch The Local Gradio UI
```bash
bash scripts/start_chatterbox_tts_ui.sh
```

## Suggested Starting Controls
- Exaggeration: {SETTINGS.exaggeration}
- CFG/Pace: {SETTINGS.cfg_weight}
- Temperature: {SETTINGS.temperature}
- Min P: 0.05
- Top P: 1.0
- Repetition penalty: 1.2

## Copy/Paste Preview Text
{preview or artifacts.tts.read_text(encoding='utf-8')[:500]}

## Approval Step
When the voice sounds right, approve this chapter so the pipeline can continue:
```bash
touch reviews/ch{chapter_num:02d}_pre_narration.approved
```

Then rerun the pipeline or chapter narration command.
"""
    review_path = _pre_narration_review_path(chapter_num)
    _write(review_path, review_text)
    return review_path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((text or "").strip() + "\n", encoding="utf-8")


def _word_count(text: str) -> int:
    return len((text or "").split())


def _chapter_heading(chapter_num: int, brief: dict[str, Any]) -> str:
    raw_title = str(brief.get("title") or "").strip()
    if raw_title:
        lowered = raw_title.lower()
        if lowered.startswith("chapter"):
            return raw_title
        return f"Chapter {chapter_num}: {raw_title}"
    return f"Chapter {chapter_num}"


def _with_chapter_heading(chapter_num: int, brief: dict[str, Any], text: str) -> str:
    body = (text or "").strip()
    if not body:
        return body
    heading = _chapter_heading(chapter_num, brief)
    if body.lower().startswith(heading.lower()):
        return body
    return f"{heading}\n\n{body}"


def _target_min_words(brief: dict[str, Any]) -> int:
    env_target = int(os.getenv("WORD_TARGET_MIN", "0") or "0")
    if env_target > 0:
        return env_target
    brief_target = int(brief.get("word_target", 0) or 0)
    if brief_target > 0:
        return max(800, int(brief_target * 0.8))
    return 1800


def _is_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _chapter_range(max_chapters: int) -> tuple[int, int]:
    start = int(os.getenv("CHAPTER_START", "1") or "1")
    last = int(os.getenv("CHAPTER_LAST", os.getenv("CHAPTER_COUNT", str(max_chapters))) or str(max_chapters))
    start = max(1, start)
    last = max(start, min(last, max_chapters))
    return start, last


def _load_checkpoint(project: str) -> int:
    path = ROOT / SETTINGS.checkpoint_dir / f"{project}.json"
    if not path.exists():
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return max(1, int(data.get("next_chapter", 1)))
    except Exception:
        return 1


def _save_checkpoint(project: str, next_chapter: int, state: str, reason: str = "") -> None:
    path = ROOT / SETTINGS.checkpoint_dir / f"{project}.json"
    _save_json(
        path,
        {
            "project": project,
            "next_chapter": next_chapter,
            "state": state,
            "reason": reason,
            "ts": time.time(),
        },
    )


def _write_chapter_inference_report(
    run_id: str,
    chapter_num: int,
    context: PipelineContext,
    fallback_count: int,
    memory_events: list[dict[str, Any]],
) -> None:
    report = {
        "run_id": run_id,
        "chapter": chapter_num,
        "fallback_count": fallback_count,
        "inference": context.inference_log,
        "memory_events": memory_events,
        "timestamp": time.time(),
    }
    path = ROOT / SETTINGS.reviews_dir / f"ch{chapter_num:02d}_inference_report.json"
    _save_json(path, report)


def _apply_memory_action(action: str, chapter_num: int) -> tuple[bool, str]:
    if action == MemoryAction.EMERGENCY:
        return True, f"memory emergency threshold reached at chapter {chapter_num}"
    if action == MemoryAction.PAUSE:
        return True, f"memory pause threshold reached at chapter {chapter_num}"
    if action == MemoryAction.THROTTLE:
        delay_s = max(0.2, SETTINGS.request_delay * SETTINGS.memory_throttle_request_delay_scale)
        print(f"[MEMORY] throttle active for chapter {chapter_num}; sleeping {delay_s:.2f}s")
        time.sleep(delay_s)
        return False, ""
    if action == MemoryAction.WARN:
        print(f"[MEMORY] warning threshold reached at chapter {chapter_num}")
    return False, ""


def _build_router() -> InferenceRouter:
    structural = OllamaClient()
    prose = HypuraClient()
    fallback = OllamaClient() if SETTINGS.allow_model_fallback else None
    return InferenceRouter(structural=structural, prose=prose, fallback=fallback)


def run_pipeline(project_name: str, dry_run: bool = False) -> int:
    _mkdirs()
    run_id = uuid.uuid4().hex[:12]
    monitor = MemoryMonitor(ROOT / SETTINGS.diagnostics_dir)

    max_chapters = SETTINGS.chapter_count
    story_bible, characters, briefs = _seed_story_inputs(max_chapters)
    if not briefs:
        raise RuntimeError("chapter_briefs.json is empty")

    start_ch, last_ch = _chapter_range(min(max_chapters, len(briefs)))
    checkpoint_start = _load_checkpoint(project_name)
    start_ch = max(start_ch, checkpoint_start)

    session_factory = make_session_factory(str(ROOT / SETTINGS.story_db_filename))
    memory = StoryMemory()

    router = _build_router()
    router_health = router.health_check()
    _save_json(
        ROOT / SETTINGS.diagnostics_dir / "runs" / f"{project_name}_{run_id}_startup.json",
        {
            "project": project_name,
            "run_id": run_id,
            "router_health": router_health,
            "llm_num_ctx": SETTINGS.llm_num_ctx,
            "timestamp": time.time(),
        },
    )
    genre = load_genre_pack(SETTINGS.default_genre_pack)

    prose_client = router.prose
    if not router_health.get("prose_ok") and router.fallback is not None:
        print("[ROUTER] Prose lane unavailable at startup; using fallback structural model for prose generation.")
        prose_client = router.fallback

    planner = PlannerAgent(client=router.structural)
    writer = WriterAgent(client=prose_client)
    editor = EditorAgent(client=prose_client)
    memory_manager = MemoryManagerAgent(client=router.structural)
    architect = ArchitectAgent(client=router.structural)

    context = PipelineContext(project_name=project_name, current_chapter=start_ch, chapter_limit=last_ch)
    fsm = Orchestrator()

    with session_factory() as session:
        project = session.query(Project).filter(Project.name == project_name).one_or_none()
        if project is None:
            project = Project(name=project_name, genre_pack=genre.genre_name, status="running")
            session.add(project)
            session.flush()

        for idx, fact in enumerate(story_bible.get("world_facts", []), start=1):
            key = f"seed_fact_{idx}"
            existing = session.query(WorldFact).filter(WorldFact.project_id == project.id, WorldFact.key == key).one_or_none()
            if existing is None:
                session.add(WorldFact(project_id=project.id, key=key, value=str(fact), scope="global"))
                memory.add_world_fact(project_name, key, str(fact))

        session.commit()

    if dry_run:
        architecture_note = "Dry-run architecture note generated without model calls."
    else:
        architecture_note = architect.run(
            context,
            {
                "genre": genre.genre_name,
                "logline": story_bible.get("logline", ""),
                "themes": story_bible.get("themes", []),
                "characters": [c.get("name", "") for c in characters],
            },
        ).content
    _write(ROOT / SETTINGS.reviews_dir / "architecture_note.md", architecture_note)

    chapter_paths: list[Path] = []
    voice_sample = SETTINGS.voice_sample or str(ROOT / "voices" / "p233_023.wav")
    fallback_count = 0

    for chapter_num in range(start_ch, last_ch + 1):
        brief = briefs[chapter_num - 1]
        context.current_chapter = chapter_num
        context.current_scene = 1
        context.state = FSMState.INIT
        context.inference_log = {}
        memory_events: list[dict[str, Any]] = []

        chapter_start = monitor.snapshot(chapter_num, "chapter_start")
        start_action = monitor.classify(chapter_start)
        monitor.write(run_id, chapter_start, start_action)
        memory_events.append({"label": chapter_start.label, "action": start_action, "rss_mb": chapter_start.rss_mb})
        should_stop, reason = _apply_memory_action(start_action, chapter_num)
        if should_stop:
            _save_checkpoint(project_name, chapter_num, "memory_pause", reason=reason)
            print(f"[PAUSE] {reason}")
            return 0

        artifacts = _chapter_artifacts(chapter_num)
        pre_narration_marker = _pre_narration_marker_path(chapter_num)
        if artifacts.final.exists() and artifacts.audio.exists():
            chapter_paths.append(stitch_chapter(ROOT / "chapters", chapter_num, [artifacts.final.read_text(encoding="utf-8")]))
            _save_checkpoint(project_name, chapter_num + 1, FSMState.COMPLETE.value, reason="already_complete")
            continue

        if artifacts.final.exists() and artifacts.tts.exists() and artifacts.summary.exists():
            chapter_path = stitch_chapter(ROOT / "chapters", chapter_num, [artifacts.final.read_text(encoding="utf-8")])
            chapter_paths.append(chapter_path)

            if dry_run:
                continue

            if SETTINGS.pause_before_narration_review and not artifacts.audio.exists() and not pre_narration_marker.exists():
                review_path = _write_pre_narration_review(chapter_num, brief, artifacts, voice_sample)
                _save_checkpoint(project_name, chapter_num, "pre_narration_review", reason="manual_voice_tuning")
                print(f"[PAUSE] Chapter {chapter_num} ready for manual voice tuning in Gradio UI: {review_path}")
                return 0

            if not artifacts.audio.exists():
                if Path(voice_sample).exists():
                    pre_tts = monitor.snapshot(chapter_num, "pre_tts_resume")
                    pre_tts_action = monitor.classify(pre_tts)
                    monitor.write(run_id, pre_tts, pre_tts_action)
                    memory_events.append({"label": pre_tts.label, "action": pre_tts_action, "rss_mb": pre_tts.rss_mb})
                    should_stop, reason = _apply_memory_action(pre_tts_action, chapter_num)
                    if should_stop:
                        _save_checkpoint(project_name, chapter_num, "memory_pause", reason=reason)
                        print(f"[PAUSE] {reason}")
                        return 0
                    narrate_chapter(
                        text=artifacts.tts.read_text(encoding="utf-8"),
                        voice_sample=voice_sample,
                        output_path=str(artifacts.audio),
                        chapter_num=chapter_num,
                        resume=True,
                    )
                else:
                    print(f"[WARN] voice sample not found: {voice_sample}")

            if artifacts.audio.exists():
                _save_checkpoint(project_name, chapter_num + 1, FSMState.COMPLETE.value, reason="resume_branch_complete")
            continue

        plan_vars = {
            "writer_prefix": genre.writer_prefix,
            "rules": genre.content_rules,
            "brief": brief,
            "characters": characters,
            "previous_summary": (ROOT / "summaries" / f"ch{chapter_num - 1:02d}_summary.txt").read_text(encoding="utf-8").strip()
            if chapter_num > 1 and (ROOT / "summaries" / f"ch{chapter_num - 1:02d}_summary.txt").exists()
            else "",
        }

        while context.state not in {FSMState.COMPLETE, FSMState.FAILED}:
            if context.state == FSMState.INIT:
                context = fsm.advance(context)
            elif context.state == FSMState.OUTLINE:
                if dry_run:
                    context.metadata["chapter_outline"] = f"Dry-run outline for chapter {chapter_num}."
                else:
                    context.metadata["chapter_outline"] = planner.run(context, plan_vars).content
                context = fsm.advance(context)
            elif context.state == FSMState.CHAPTER_PLAN:
                if dry_run:
                    context.metadata["scene_plan"] = f"Dry-run scene plan for chapter {chapter_num}."
                else:
                    context.metadata["scene_plan"] = planner.run(
                        context,
                        {"task": "build scene plan", "outline": context.metadata.get("chapter_outline", ""), "brief": brief},
                    ).content
                _write(ROOT / SETTINGS.reviews_dir / f"ch{chapter_num:02d}_scene_plan.md", context.metadata["scene_plan"])
                context = fsm.advance(context)
            elif context.state == FSMState.SCENE_WRITE:
                before_writer = monitor.snapshot(chapter_num, "pre_writer")
                before_writer_action = monitor.classify(before_writer)
                monitor.write(run_id, before_writer, before_writer_action)
                memory_events.append({"label": before_writer.label, "action": before_writer_action, "rss_mb": before_writer.rss_mb})
                should_stop, reason = _apply_memory_action(before_writer_action, chapter_num)
                if should_stop:
                    _save_checkpoint(project_name, chapter_num, "memory_pause", reason=reason)
                    print(f"[PAUSE] {reason}")
                    return 0
                if dry_run:
                    draft = (
                        f"Chapter {chapter_num} draft placeholder.\n"
                        "The protagonist faces a turning-point choice and uncovers a new clue."
                    )
                else:
                    draft = writer.run(
                        context,
                        {
                            "instruction": "Write a polished chapter in past tense unless the brief explicitly requires another tense.",
                            "style": genre.writer_prefix,
                            "brief": brief,
                            "characters": characters,
                            "previous_summary": plan_vars.get("previous_summary", ""),
                            "scene_plan": context.metadata.get("scene_plan", ""),
                        },
                    ).content
                    diag = context.inference_log.get("agents", {}).get("writer", {}).get("diagnostics", {})
                    if diag.get("fallback_used"):
                        fallback_count += 1
                _write(artifacts.draft, draft)
                context.metadata["draft"] = draft
                context = fsm.advance(context)
            elif context.state == FSMState.SCENE_EDIT:
                before_editor = monitor.snapshot(chapter_num, "pre_editor")
                before_editor_action = monitor.classify(before_editor)
                monitor.write(run_id, before_editor, before_editor_action)
                memory_events.append({"label": before_editor.label, "action": before_editor_action, "rss_mb": before_editor.rss_mb})
                should_stop, reason = _apply_memory_action(before_editor_action, chapter_num)
                if should_stop:
                    _save_checkpoint(project_name, chapter_num, "memory_pause", reason=reason)
                    print(f"[PAUSE] {reason}")
                    return 0
                if dry_run:
                    edited = context.metadata.get("draft", "")
                else:
                    edited = editor.run(
                        context,
                        {
                            "instruction": "Tighten prose, remove repetition, preserve plot facts.",
                            "editor_prefix": genre.editor_prefix,
                            "draft": context.metadata.get("draft", ""),
                        },
                    ).content
                    diag = context.inference_log.get("agents", {}).get("editor", {}).get("diagnostics", {})
                    if diag.get("fallback_used"):
                        fallback_count += 1
                _write(artifacts.edited, edited)

                final_text = edited

                if not dry_run:
                    min_words = _target_min_words(brief)
                    expansion_passes = max(0, int(os.getenv("EXPANSION_PASSES", "1") or "1"))
                    pass_idx = 0
                    while _word_count(final_text) < min_words and pass_idx < expansion_passes:
                        pass_idx += 1
                        expanded = writer.run(
                            context,
                            {
                                "instruction": (
                                    f"Expand this chapter to at least {min_words} words while preserving canon, tone, "
                                    "plot facts, and chapter continuity. Return only chapter prose."
                                ),
                                "style": genre.writer_prefix,
                                "brief": brief,
                                "characters": characters,
                                "previous_summary": plan_vars.get("previous_summary", ""),
                                "scene_plan": context.metadata.get("scene_plan", ""),
                                "draft": final_text,
                            },
                        ).content
                        final_text = editor.run(
                            context,
                            {
                                "instruction": "Polish this expanded chapter while preserving plot and continuity.",
                                "editor_prefix": genre.editor_prefix,
                                "draft": expanded,
                            },
                        ).content
                        diag = context.inference_log.get("agents", {}).get("editor", {}).get("diagnostics", {})
                        if diag.get("fallback_used"):
                            fallback_count += 1

                final_text = _with_chapter_heading(chapter_num, brief, final_text)
                lint_report = lint_chapter(final_text, chapter_num, brief, LintSettings())
                _save_json(artifacts.lint_json, lint_report)
                _write(artifacts.lint_md, to_markdown(lint_report))
                _write(artifacts.final, final_text)
                _write(artifacts.tts, final_text)

                if dry_run:
                    summary = f"Dry-run summary for chapter {chapter_num}."
                else:
                    summary = memory_manager.run(
                        context,
                        {
                            "task": "Summarize chapter in 120-180 words with unresolved threads.",
                            "chapter_text": final_text,
                            "brief": brief,
                        },
                    ).content
                _write(artifacts.summary, summary)
                context.metadata["summary"] = summary
                context = fsm.advance(context)
            elif context.state == FSMState.MEMORY_UPDATE:
                memory.add_scene(project_name, chapter_num, 1, artifacts.final.read_text(encoding="utf-8"))
                memory.update_character(project_name, "Mara Quill", context.metadata.get("summary", ""))
                context = fsm.advance(context)
            elif context.state == FSMState.CHECKPOINT:
                _save_checkpoint(project_name, chapter_num + 1, context.state.value, reason="chapter_checkpoint")
                context = fsm.advance(context)
            elif context.state == FSMState.NEXT_SCENE:
                context.state = FSMState.COMPLETE
            else:
                context.state = FSMState.FAILED

        chapter_path = stitch_chapter(ROOT / "chapters", chapter_num, [artifacts.final.read_text(encoding="utf-8")])
        chapter_paths.append(chapter_path)

        if dry_run:
            continue

        if SETTINGS.pause_before_narration_review and not pre_narration_marker.exists():
            review_path = _write_pre_narration_review(chapter_num, brief, artifacts, voice_sample)
            _save_checkpoint(project_name, chapter_num, "pre_narration_review", reason="manual_voice_tuning")
            print(f"[PAUSE] Chapter {chapter_num} ready for manual voice tuning in Gradio UI: {review_path}")
            return 0

        if Path(voice_sample).exists():
            pre_tts = monitor.snapshot(chapter_num, "pre_tts")
            pre_tts_action = monitor.classify(pre_tts)
            monitor.write(run_id, pre_tts, pre_tts_action)
            memory_events.append({"label": pre_tts.label, "action": pre_tts_action, "rss_mb": pre_tts.rss_mb})
            should_stop, reason = _apply_memory_action(pre_tts_action, chapter_num)
            if should_stop:
                _save_checkpoint(project_name, chapter_num, "memory_pause", reason=reason)
                print(f"[PAUSE] {reason}")
                return 0
            narrate_chapter(
                text=artifacts.tts.read_text(encoding="utf-8"),
                voice_sample=voice_sample,
                output_path=str(artifacts.audio),
                chapter_num=chapter_num,
                resume=True,
            )
        else:
            print(f"[WARN] voice sample not found: {voice_sample}")

        if artifacts.audio.exists():
            _save_checkpoint(project_name, chapter_num + 1, FSMState.COMPLETE.value, reason="chapter_complete")

        gc_started = time.time()
        gc.collect()
        gc_elapsed = round(time.time() - gc_started, 3)
        post_gc = monitor.snapshot(chapter_num, "chapter_complete")
        post_gc_action = monitor.classify(post_gc)
        monitor.write(run_id, post_gc, post_gc_action)
        memory_events.append(
            {
                "label": post_gc.label,
                "action": post_gc_action,
                "rss_mb": post_gc.rss_mb,
                "gc_elapsed_s": gc_elapsed,
            }
        )
        _write_chapter_inference_report(run_id, chapter_num, context, fallback_count, memory_events)

    manuscript = stitch_novel(ROOT / "chapters", chapter_paths)
    for fmt in os.getenv("EXPORT_FORMATS", "md").split(","):
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        try:
            exported = export_manuscript(manuscript, fmt)
            print(f"[OK] Export ({fmt}): {exported}")
        except Exception as exc:
            print(f"[WARN] Export skipped ({fmt}): {exc}")
    print(f"[OK] Manuscript: {manuscript}")
    return 0


def main() -> int:
    dry_run = _is_truthy("DRY_RUN", default=False)
    project_name = os.getenv("PROJECT_NAME", "story_engine_run")
    return run_pipeline(project_name=project_name, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
