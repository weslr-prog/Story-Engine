from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import wave
import hashlib
import re
from contextlib import closing
from pathlib import Path
from typing import Any

from scripts.convert_story_engine import Inputs, convert_rule, write_prompt
from scripts.preflight import (
    check_chatterbox,
    check_ffmpeg,
    check_local_disk_kv,
    check_ollama,
    discover_api_names,
)
from config import SETTINGS
from ui.session_manager import (
    ROOT,
    INPUT_FILES,
    get_active_project,
    initialize_project,
    input_path,
    list_projects,
    project_paths,
    update_session,
    set_active_project,
)

REQUIRED_SOURCE_KEYS = ["dna", "bible", "blueprint"]
REQUIRED_CONVERSION_KEYS = ["dna", "bible", "blueprint", "style_guide"]
REQUIRED_GUIDE_KEYS = ["style_guide", "consistency"]
SUPPORTED_VOICE_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
MODEL_PROFILE_QWEN35 = "Qwen3.5-9B Non-thinking (MLX)"
MODEL_PROFILE_QWEN25_Q5 = "Qwen2.5-7B-Instruct-Q5 (Ollama)"
MODEL_PROFILE_CHOICES = [MODEL_PROFILE_QWEN35, MODEL_PROFILE_QWEN25_Q5]
LAST_SIGNAL_FILE_CANDIDATES = {
    "dna": ["Phase 1 - Story DNA Summary.txt", "Story DNA Summary.txt", "Story DNA.txt"],
    "bible": ["Phase 2 - Story Bible.txt", "Story Bible.txt"],
    "blueprint": ["Phase 3 - Chapter Blueprint.txt", "Chapter Blueprint.txt"],
    "style_guide": [
        "style_guide.txt",
        "Phase 4 - Writing Prompts.txt",
        "Writing Prompts.txt",
    ],
    "consistency": ["consistency_checklist.txt"],
}


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _model_profile_runtime(model_profile: str | None) -> tuple[str, str]:
    chosen = (model_profile or "").strip()
    if chosen == MODEL_PROFILE_QWEN35:
        return (
            os.getenv("QWEN35_MLX_URL", "http://127.0.0.1:8080/v1/chat/completions"),
            os.getenv("QWEN35_MLX_MODEL", "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit"),
        )
    # Default to the lower-memory profile.
    return (
        os.getenv("QWEN25_OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions"),
        os.getenv("QWEN25_OLLAMA_MODEL", "qwen2.5:7b-instruct-q5_K_M"),
    )

STYLE_GUIDE_TEMPLATE = """# Style Guide\n\n## Narrative POV and Tense\n- Use third person limited (primarily protagonist).\n- Use past tense consistently.\n\n## Voice and Diction\n- Prefer concrete verbs and precise nouns over abstract phrasing.\n- Keep dialogue natural and subtext-forward; avoid exposition dumps.\n- Avoid meta commentary about writing process.\n\n## Scene Construction\n- Every scene must do at least one: advance plot, deepen character, or escalate tension.\n- Keep transitions clear in time/place without long setup paragraphs.\n- End chapters on consequence-driven forward pull.\n\n## Prohibited Patterns\n- No repeated paragraph loops.\n- No out-of-world references (AI/model/prompt/author language).\n- Avoid early reveal leaks from future chapters.\n"""

CONSISTENCY_TEMPLATE = """# Consistency Checklist\n\n## Canon Facts\n- Character names, ages, and roles match `characters.json`.\n- Setting facts and world rules match `story_bible.json`.\n\n## Chapter Continuity\n- Chapter events align with `chapter_briefs.json` for this chapter.\n- Cause-and-effect chain is preserved from previous chapter.\n- No contradiction in timeline, injuries, possessions, or locations.\n\n## Character Integrity\n- Motivations reflect established core wounds and flawed beliefs.\n- Speech patterns stay consistent for each recurring character.\n\n## Reveal and Stakes Control\n- No premature major reveal before planned chapter.\n- Cliffhanger emerges from in-chapter character choices.\n\n## Output Quality\n- No duplicated paragraphs/sentences.\n- No meta narration about process/tools.\n- Chapter ending sets up next chapter clearly.\n"""



def create_project(project_name: str) -> tuple[list[str], str, str]:
    if not project_name.strip():
        return list_projects(), get_active_project(), "Project name cannot be empty."
    paths = initialize_project(project_name)
    active = set_active_project(paths.name)
    return list_projects(), active, f"Created and selected project: {active}"



def refresh_projects() -> tuple[list[str], str]:
    names = list_projects()
    active = get_active_project()
    return names, active



def select_project(project_name: str) -> str:
    if not project_name:
        return "Choose a project first."
    active = set_active_project(project_name)
    return f"Active project set to: {active}"



def project_overview(project_name: str) -> str:
    if not project_name:
        return "No active project."
    paths = initialize_project(project_name)
    existing_inputs = []
    for key, file_name in INPUT_FILES.items():
        p = paths.inputs_dir / file_name
        if p.exists() and p.stat().st_size > 0:
            existing_inputs.append(f"- {key}: {file_name}")

    existing_json = []
    for file_name in ["story_bible.json", "characters.json", "chapter_briefs.json"]:
        p = paths.json_dir / file_name
        if p.exists() and p.stat().st_size > 0:
            existing_json.append(f"- {file_name}")

    lines = [
        f"Project: {paths.name}",
        f"Root: {paths.root}",
        "",
        "Inputs:",
        *(existing_inputs or ["- none yet"]),
        "",
        "Converted JSON:",
        *(existing_json or ["- none yet"]),
    ]
    return "\n".join(lines)



def create_guide_template(project_name: str, input_key: str) -> tuple[str, str]:
    if not project_name:
        return "No active project selected.", ""
    if input_key == "style_guide":
        template = STYLE_GUIDE_TEMPLATE
    elif input_key == "consistency":
        template = CONSISTENCY_TEMPLATE
    else:
        return "Template is only available for Style Guide and Consistency Checklist.", ""

    path = input_path(project_name, input_key)
    path.write_text(template, encoding="utf-8")
    return f"Template created: {path.name}. Edit and save as needed.", template



def save_input_text(project_name: str, input_key: str, text: str) -> str:
    if not project_name:
        return "No active project selected."
    path = input_path(project_name, input_key)
    path.write_text(text or "", encoding="utf-8")
    return f"Saved: {path.name}"



def load_input_text(project_name: str, input_key: str) -> str:
    if not project_name:
        return ""
    path = input_path(project_name, input_key)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")



def import_uploaded_file(project_name: str, input_key: str, uploaded_path: str | None) -> str:
    if not project_name:
        return "No active project selected."
    if not uploaded_path:
        return "Choose a file to upload."

    target = input_path(project_name, input_key)
    src = Path(uploaded_path)
    shutil.copyfile(src, target)
    return f"Imported {src.name} -> {target.name}"


def import_last_signal_sources(project_name: str) -> str:
    if not project_name:
        return "No active project selected."

    source_dir = ROOT / "The Last Signal"
    if not source_dir.exists() or not source_dir.is_dir():
        return f"Source folder not found: {source_dir}"

    copied: list[str] = []
    missing: list[str] = []

    for key, candidates in LAST_SIGNAL_FILE_CANDIDATES.items():
        src: Path | None = None
        for name in candidates:
            candidate = source_dir / name
            if candidate.exists() and candidate.is_file():
                src = candidate
                break
        if src is None:
            if key in REQUIRED_SOURCE_KEYS:
                missing.append(INPUT_FILES[key])
            continue

        dst = input_path(project_name, key)
        shutil.copyfile(src, dst)
        copied.append(f"{src.name} -> {dst.name}")

    if missing:
        return "Imported partial source set from The Last Signal. Missing required files:\n- " + "\n- ".join(missing)
    if not copied:
        return "No matching source files found in The Last Signal folder."
    return "Imported source files from The Last Signal:\n- " + "\n- ".join(copied)



def _validate_required_sources(project_name: str) -> list[str]:
    missing = []
    for key in REQUIRED_SOURCE_KEYS:
        p = input_path(project_name, key)
        if not p.exists() or not p.read_text(encoding="utf-8", errors="replace").strip():
            missing.append(INPUT_FILES[key])
    return missing


def _non_empty_text(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    return bool(path.read_text(encoding="utf-8", errors="replace").strip())


def _validate_required_conversion_inputs(project_name: str) -> list[str]:
    missing = []
    for key in REQUIRED_CONVERSION_KEYS:
        p = input_path(project_name, key)
        if not _non_empty_text(p):
            missing.append(INPUT_FILES[key])
    return missing


def _validate_converted_json_outputs(json_dir: Path) -> list[str]:
    issues: list[str] = []

    story_bible_path = json_dir / "story_bible.json"
    characters_path = json_dir / "characters.json"
    briefs_path = json_dir / "chapter_briefs.json"

    def _load_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return None

    story_bible = _load_json(story_bible_path)
    if not isinstance(story_bible, dict) or not story_bible:
        issues.append("story_bible.json is missing or empty object")

    characters = _load_json(characters_path)
    if not isinstance(characters, list) or len(characters) == 0:
        issues.append("characters.json is missing or empty list")

    briefs = _load_json(briefs_path)
    if not isinstance(briefs, list) or len(briefs) == 0:
        issues.append("chapter_briefs.json is missing or empty list")

    return issues


def get_required_input_windows(project_name: str) -> tuple[str, str, str, str, str]:
    if not project_name:
        return "", "", "", "", "No active project selected."

    def _load_slot(key: str) -> tuple[str, bool]:
        p = input_path(project_name, key)
        if not _non_empty_text(p):
            return "", False
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        preview = text if len(text) <= 1800 else text[:1800] + "\n\n...[truncated preview]"
        return preview, True

    dna_text, dna_ok = _load_slot("dna")
    bible_text, bible_ok = _load_slot("bible")
    blueprint_text, blueprint_ok = _load_slot("blueprint")
    style_text, style_ok = _load_slot("style_guide")

    ready = dna_ok and bible_ok and blueprint_ok and style_ok
    status = (
        "Required source windows loaded. Conversion is unlocked."
        if ready
        else "Required source windows loaded. Conversion remains locked until all 4 source docs are non-empty."
    )
    return dna_text, bible_text, blueprint_text, style_text, status



def _validate_required_guides(project_name: str) -> list[str]:
    missing = []
    for key in REQUIRED_GUIDE_KEYS:
        p_input = input_path(project_name, key)
        file_name = INPUT_FILES[key]
        p_generated = initialize_project(project_name).json_dir / file_name
        has_input = p_input.exists() and p_input.read_text(encoding="utf-8", errors="replace").strip()
        has_generated = p_generated.exists() and p_generated.read_text(encoding="utf-8", errors="replace").strip()
        if not has_input and not has_generated:
            missing.append(INPUT_FILES[key])
    return missing



def get_readiness_report(project_name: str) -> str:
    if not project_name:
        return "No active project selected."

    paths = initialize_project(project_name)
    source_missing = _validate_required_sources(project_name)
    conversion_missing = _validate_required_conversion_inputs(project_name)
    guide_missing = _validate_required_guides(project_name)

    checks: list[tuple[str, bool, str]] = []
    checks.append(("Story source docs", not source_missing, "Missing: " + ", ".join(source_missing) if source_missing else "Ready"))
    checks.append(("Conversion lock (4 docs)", not conversion_missing, "Missing: " + ", ".join(conversion_missing) if conversion_missing else "Ready"))
    checks.append(("Guide docs", not guide_missing, "Missing: " + ", ".join(guide_missing) if guide_missing else "Ready"))

    json_outputs = ["story_bible.json", "characters.json", "chapter_briefs.json"]
    json_missing = [name for name in json_outputs if not (paths.json_dir / name).exists()]
    json_issues = _validate_converted_json_outputs(paths.json_dir) if not json_missing else []
    json_ok = (not json_missing) and (not json_issues)
    json_detail = "Missing: " + ", ".join(json_missing) if json_missing else ("Invalid: " + "; ".join(json_issues) if json_issues else "Ready")
    checks.append(("Converted JSON", json_ok, json_detail))

    has_active_voice = bool(list_project_voices(project_name))
    checks.append(("Project voices", has_active_voice, "Upload a voice sample in Voice tab" if not has_active_voice else "Ready"))

    lines = [f"Readiness report for project '{project_name}':", ""]
    for title, ok, detail in checks:
        icon = "[OK]" if ok else "[MISSING]"
        lines.append(f"{icon} {title}: {detail}")

    lines.extend(
        [
            "",
            "Guide file lifecycle:",
            "1. Converter generates style_guide.txt and consistency_checklist.txt from source docs.",
            "2. You can optionally override by editing guide files in Inputs tab.",
            "3. Sync to root pipeline files before running pipeline_novel.py.",
            "4. Pipeline reads these files during generation.",
        ]
    )
    return "\n".join(lines)



def run_conversion(project_name: str, mode: str) -> str:
    if not project_name:
        return "No active project selected."

    missing = _validate_required_conversion_inputs(project_name)
    if missing:
        return (
            "Conversion locked. Missing required source text files:\n- "
            + "\n- ".join(missing)
            + "\n\nRequired for conversion lock: Story DNA Summary, Story Bible, Chapter Blueprint, Style Guide (or Phase 4 Writing Prompts)."
        )

    paths = initialize_project(project_name)
    inputs = Inputs(
        dna=input_path(project_name, "dna"),
        bible=input_path(project_name, "bible"),
        blueprint=input_path(project_name, "blueprint"),
        out_dir=paths.json_dir,
    )

    logs: list[str] = [f"Running conversion in '{mode}' mode for project '{project_name}'..."]

    if mode in {"rule", "hybrid"}:
        convert_rule(inputs)
        logs.append(f"[OK] {paths.json_dir / 'story_bible.json'}")
        logs.append(f"[OK] {paths.json_dir / 'characters.json'}")
        logs.append(f"[OK] {paths.json_dir / 'chapter_briefs.json'}")
        logs.append(f"[OK] {paths.json_dir / 'style_guide.txt'}")
        logs.append(f"[OK] {paths.json_dir / 'consistency_checklist.txt'}")
        logs.append(f"[OK] {paths.json_dir / 'master_system_prompt.md'}")

    if mode in {"prompt", "hybrid"}:
        write_prompt(inputs)
        logs.append(f"[OK] {paths.json_dir / 'story_engine_conversion_prompt.md'}")

    issues = _validate_converted_json_outputs(paths.json_dir)
    if issues:
        logs.append("[ERROR] Converted outputs failed validation:")
        for item in issues:
            logs.append(f"- {item}")
        logs.append("Fix source inputs and run conversion again.")

    return "\n".join(logs)


def clear_project_data(project_name: str, force_stop: bool, clear_root_pipeline_files: bool) -> str:
    if not project_name:
        return "No active project selected."

    state = _runner_state()
    pid = state.get("pid")
    running = _running_pid(pid)

    stop_note = "not running"
    if running and not force_stop:
        return f"Pipeline is running (pid={pid}). Enable force stop for start-fresh cleanup."
    if running and force_stop:
        try:
            os.kill(pid, 15)
            stop_note = f"stopped pid={pid}"
        except OSError as exc:
            stop_note = f"failed to stop pid={pid}: {exc}"

    paths = initialize_project(project_name)
    removed = 0

    for p in paths.inputs_dir.glob("*.txt"):
        if _safe_unlink(p):
            removed += 1
    for p in paths.json_dir.glob("*.json"):
        if _safe_unlink(p):
            removed += 1
    for p in paths.json_dir.glob("*.txt"):
        if _safe_unlink(p):
            removed += 1
    for p in paths.json_dir.glob("*.md"):
        if _safe_unlink(p):
            removed += 1

    for d in [paths.chapters_dir, paths.audio_dir, paths.reviews_dir, paths.exports_dir]:
        removed += _safe_rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    if clear_root_pipeline_files:
        for p in [
            ROOT / "story_bible.json",
            ROOT / "characters.json",
            ROOT / "chapter_briefs.json",
            ROOT / "style_guide.txt",
            ROOT / "consistency_checklist.txt",
            ROOT / "master_system_prompt.md",
        ]:
            if _safe_unlink(p):
                removed += 1
        removed += _clear_root_runtime_outputs()

    _reset_runner_state()
    update_session(project_name, active_stage="idle", pause_reason="", last_run_finished_at=str(int(time.time())))
    return (
        f"Start-fresh cleanup complete for project '{project_name}': removed {removed} files; "
        f"process {stop_note}."
    )



def load_json_preview(project_name: str, json_name: str) -> str:
    if not project_name:
        return "No active project selected."
    paths = initialize_project(project_name)
    p = paths.json_dir / json_name
    if not p.exists():
        return f"Not found: {p.name}"

    text = p.read_text(encoding="utf-8", errors="replace")
    if json_name.endswith(".json"):
        try:
            data = json.loads(text)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            return text
    return text



def sync_project_json_to_root(project_name: str) -> str:
    if not project_name:
        return "No active project selected."

    paths = initialize_project(project_name)
    json_issues = _validate_converted_json_outputs(paths.json_dir)
    if json_issues:
        return "Sync blocked: converted JSON outputs are invalid:\n- " + "\n- ".join(json_issues)

    copied: list[str] = []
    missing_guides = _validate_required_guides(project_name)

    for name in ["story_bible.json", "characters.json", "chapter_briefs.json"]:
        src = paths.json_dir / name
        if src.exists():
            shutil.copyfile(src, ROOT / name)
            copied.append(name)

    for key in ["style_guide", "consistency"]:
        # Prefer explicit project input guides over converter-generated guides.
        # This allows Phase 4 (Writing Prompts) mapped into style_guide slot to
        # directly steer runtime output quality when present.
        src = input_path(project_name, key)
        if not src.exists() or not src.read_text(encoding="utf-8", errors="replace").strip():
            src = paths.json_dir / INPUT_FILES[key]
        if src.exists() and src.read_text(encoding="utf-8", errors="replace").strip():
            out_name = "style_guide.txt" if key == "style_guide" else "consistency_checklist.txt"
            shutil.copyfile(src, ROOT / out_name)
            copied.append(out_name)

    master_prompt_src = paths.json_dir / "master_system_prompt.md"
    if master_prompt_src.exists() and master_prompt_src.read_text(encoding="utf-8", errors="replace").strip():
        shutil.copyfile(master_prompt_src, ROOT / "master_system_prompt.md")
        copied.append("master_system_prompt.md")

    note = ""
    if missing_guides:
        note = "\n\nNote: Missing guide files were not synced:\n- " + "\n- ".join(missing_guides)

    if not copied:
        return "Nothing to sync yet. Convert files first."
    return "Synced to project root:\n- " + "\n- ".join(copied) + note



def _voice_path(project_name: str, voice_name: str) -> Path:
    paths = initialize_project(project_name)
    return paths.voices_dir / voice_name



def list_project_voices(project_name: str) -> list[str]:
    if not project_name:
        return []
    paths = initialize_project(project_name)
    out: list[str] = []
    for p in sorted(paths.voices_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_VOICE_EXTENSIONS:
            out.append(p.name)
    return out



def _validate_voice_upload(src: Path) -> str | None:
    if not src.exists() or not src.is_file():
        return "Uploaded voice file was not found."
    ext = src.suffix.lower()
    if ext not in SUPPORTED_VOICE_EXTENSIONS:
        return (
            "Unsupported voice format. Accepted by Story Studio for Chatterbox workflow: "
            + ", ".join(SUPPORTED_VOICE_EXTENSIONS)
            + " (WAV recommended)."
        )
    if src.stat().st_size == 0:
        return "Uploaded voice file is empty."
    return None



def import_project_voice(project_name: str, uploaded_path: str | None) -> tuple[str, list[str], str | None]:
    if not project_name:
        return "No active project selected.", [], None
    if not uploaded_path:
        return "Choose a voice file to upload.", list_project_voices(project_name), None

    src = Path(uploaded_path)
    err = _validate_voice_upload(src)
    if err:
        voices = list_project_voices(project_name)
        return err, voices, (voices[0] if voices else None)

    paths = initialize_project(project_name)
    target = paths.voices_dir / src.name
    shutil.copyfile(src, target)
    voices = list_project_voices(project_name)
    return f"Imported voice: {src.name}", voices, src.name



def sync_selected_voice_to_root(project_name: str, voice_name: str) -> str:
    if not project_name:
        return "No active project selected."
    if not voice_name:
        return "Choose a voice first."

    src = _voice_path(project_name, voice_name)
    if not src.exists():
        return "Selected voice file no longer exists."

    root_voices = ROOT / "voices"
    root_voices.mkdir(parents=True, exist_ok=True)
    dst = root_voices / voice_name
    shutil.copyfile(src, dst)

    env_path = ROOT / ".env"
    line = f"VOICE_SAMPLE=voices/{voice_name}"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines()
        replaced = False
        for idx, raw in enumerate(lines):
            if raw.startswith("VOICE_SAMPLE="):
                lines[idx] = line
                replaced = True
                break
        if not replaced:
            lines.append(line)
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        env_path.write_text(line + "\n", encoding="utf-8")

    update_session(project_name, active_voice=voice_name)
    return f"Synced voice to root and updated .env: voices/{voice_name}"



def get_project_voice_download_path(project_name: str, voice_name: str) -> str | None:
    if not project_name or not voice_name:
        return None
    p = _voice_path(project_name, voice_name)
    if not p.exists() or not p.is_file():
        return None
    return str(p)


RUNNER_STATE_DIR = ROOT / ".state"
RUNNER_STATE_FILE = RUNNER_STATE_DIR / "pipeline_runner.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _runner_state() -> dict[str, Any]:
    return _read_json(
        RUNNER_STATE_FILE,
        {
            "pid": None,
            "log_path": "",
            "started_at": 0.0,
            "mode": "",
            "chapter_limit": 0,
            "start_chapter": 1,
            "last_chapter": 1,
            "target_chapter": 0,
            "chapter_complete_alert": "double_beep",
            "kv_cache_mode": "",
            "kv_cache_evidence": "",
            "project": "",
        },
    )


def _save_runner_state(state: dict[str, Any]) -> None:
    _write_json(RUNNER_STATE_FILE, state)


def _chapter_artifacts(chapter_num: int) -> dict[str, Path]:
    ch = f"ch{chapter_num:02d}"
    return {
        "draft": ROOT / "chapters" / f"{ch}_draft.txt",
        "edited": ROOT / "chapters" / f"{ch}_edited.txt",
        "final": ROOT / "chapters" / f"{ch}_final.txt",
        "tts": ROOT / "chapters" / f"{ch}_tts.txt",
        "summary": ROOT / "summaries" / f"{ch}_summary.txt",
        "audio": ROOT / "audio" / f"{ch}_narration.wav",
        "pre_marker": ROOT / SETTINGS.reviews_dir / f"{ch}_pre_narration.approved",
        "post_marker": ROOT / SETTINGS.reviews_dir / f"{ch}_post_chapter.approved",
    }


def _safe_unlink(path: Path) -> bool:
    try:
        if path.exists() and (path.is_file() or path.is_symlink()):
            path.unlink()
            return True
    except OSError:
        return False
    return False


def _safe_rmtree(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    removed = sum(1 for p in path.rglob("*") if p.is_file())
    shutil.rmtree(path, ignore_errors=True)
    return removed


def _chapter_review_files(chapter_num: int) -> list[Path]:
    ch = f"ch{chapter_num:02d}"
    reviews = ROOT / SETTINGS.reviews_dir
    return [
        reviews / f"{ch}_pre_narration_review.md",
        reviews / f"{ch}_post_chapter_review.md",
        reviews / f"{ch}_lint.md",
        reviews / f"{ch}_lint.json",
        reviews / f"{ch}_scene_plan.md",
        reviews / f"{ch}_local_critic.md",
        reviews / f"{ch}_external_critic.md",
        reviews / f"{ch}_external_critic_prompt.md",
        reviews / f"{ch}_edited_for_external.txt",
    ]


def _reset_chapter_outputs(chapter_num: int) -> int:
    removed = 0
    for path in _chapter_artifacts(chapter_num).values():
        if _safe_unlink(path):
            removed += 1

    for path in _chapter_review_files(chapter_num):
        if _safe_unlink(path):
            removed += 1

    segments_dir = ROOT / "audio" / "segments" / f"ch{chapter_num:02d}"
    removed += _safe_rmtree(segments_dir)
    scene_dir = ROOT / "chapters" / "scenes" / f"ch{chapter_num:02d}"
    removed += _safe_rmtree(scene_dir)
    return removed


def _reset_runner_state() -> None:
    _save_runner_state(
        {
            "pid": None,
            "log_path": "",
            "started_at": 0.0,
            "mode": "",
            "chapter_limit": 0,
            "start_chapter": 1,
            "last_chapter": 1,
            "target_chapter": 0,
            "chapter_complete_alert": "double_beep",
            "kv_cache_mode": "",
            "kv_cache_evidence": "",
            "project": "",
        }
    )


def _max_known_chapters() -> int:
    max_chapters = SETTINGS.chapter_count
    briefs_path = ROOT / "chapter_briefs.json"
    if not briefs_path.exists():
        return max(1, int(max_chapters))
    try:
        briefs = json.loads(briefs_path.read_text(encoding="utf-8"))
        if isinstance(briefs, list) and briefs:
            max_chapters = len(briefs)
    except json.JSONDecodeError:
        pass
    return max(1, int(max_chapters))


def get_default_chapter_range() -> tuple[int, int]:
    last = _max_known_chapters()
    return 1, last


def _clear_root_runtime_outputs() -> int:
    removed = 0

    for path in (ROOT / "chapters").glob("ch*.txt"):
        if _safe_unlink(path):
            removed += 1

    for path in (ROOT / "summaries").glob("ch*_summary.txt"):
        if _safe_unlink(path):
            removed += 1

    for path in (ROOT / "audio").glob("ch*_narration.wav"):
        if _safe_unlink(path):
            removed += 1

    removed += _safe_rmtree(ROOT / "audio" / "segments")
    removed += _safe_rmtree(ROOT / "chapters" / "scenes")

    reviews = ROOT / SETTINGS.reviews_dir
    for path in reviews.glob("ch*.*"):
        if _safe_unlink(path):
            removed += 1

    for path in RUNNER_STATE_DIR.glob("pipeline_run_*.log"):
        if _safe_unlink(path):
            removed += 1

    return removed


def clear_run_logs() -> str:
    removed = 0
    RUNNER_STATE_DIR.mkdir(parents=True, exist_ok=True)
    for path in RUNNER_STATE_DIR.glob("pipeline_run_*.log"):
        if _safe_unlink(path):
            removed += 1
    return f"Cleared {removed} run log(s)."


def _chapter_complete(chapter_num: int) -> bool:
    files = _chapter_artifacts(chapter_num)
    required = [files["final"], files["tts"], files["summary"], files["audio"]]
    if not all(path.exists() for path in required):
        return False
    if SETTINGS.pause_before_narration_review and not files["pre_marker"].exists():
        return False
    if SETTINGS.pause_after_chapter_review and not files["post_marker"].exists():
        return False
    return True


def _next_pending_chapter(start_chapter: int, last_chapter: int) -> int:
    for chapter_num in range(max(1, start_chapter), max(1, last_chapter) + 1):
        if not _chapter_complete(chapter_num):
            return chapter_num
    return max(1, last_chapter)


def _chapter_completion_status(chapter_num: int) -> str:
    files = _chapter_artifacts(chapter_num)
    outputs_ready = all(path.exists() for path in [files["final"], files["tts"], files["summary"], files["audio"]])
    text_ready = all(path.exists() for path in [files["final"], files["tts"], files["summary"]])
    if _chapter_complete(chapter_num):
        return "already complete"
    if outputs_ready:
        return "outputs complete (review markers pending)"
    if text_ready and not files["audio"].exists():
        return "missing narration"
    if any(path.exists() for path in [files["draft"], files["edited"], files["final"], files["tts"], files["summary"], files["audio"]]):
        return "partial outputs"
    return "not started"


def _chapter_outputs_ready(chapter_num: int) -> bool:
    files = _chapter_artifacts(chapter_num)
    return all(path.exists() for path in [files["final"], files["tts"], files["summary"], files["audio"]])


def _format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _wav_seconds(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        with closing(wave.open(str(path), "rb")) as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return 0.0


def _resolved_backend() -> str:
    backend = SETTINGS.llm_backend.strip().lower()
    if backend not in {"openclaw", "local_disk_kv"}:
        backend = "local_disk_kv"
    if SETTINGS.use_local_disk_kv:
        backend = "local_disk_kv"
    return backend


def _kv_cache_status() -> tuple[str, str]:
    runs_dir = ROOT / SETTINGS.diagnostics_dir / "runs"
    if runs_dir.exists():
        startup_files = sorted(runs_dir.glob("*_startup.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if startup_files:
            try:
                payload = json.loads(startup_files[0].read_text(encoding="utf-8"))
                inventory = payload.get("router_health", {}).get("prose_inventory", {})
                if inventory.get("configured_present"):
                    return "runtime-verified", "configured Hypura model observed in /api/tags"
                if inventory.get("ok"):
                    return "partially-verified", "Hypura /api/tags reachable but configured model not observed"
            except Exception:
                pass

    configured_mode = (
        os.getenv("OLLAMA_KV_CACHE_TYPE")
        or os.getenv("LLAMA_CACHE_TYPE_V")
        or os.getenv("KV_CACHE_TYPE")
        or ""
    ).strip()
    if not configured_mode:
        return "unknown", "no KV compression env flag detected in app process"
    if configured_mode.lower().startswith("turbo"):
        return configured_mode, "compression mode configured by env"
    return configured_mode, "configured mode is not a turbo compression type"


def get_service_status() -> str:
    backend = _resolved_backend()
    ffmpeg_ok, ffmpeg_detail = check_ffmpeg()
    ollama_ok, ollama_detail = check_ollama()
    local_kv_ok, local_kv_detail = check_local_disk_kv()
    chatterbox_ok, chatterbox_detail = check_chatterbox()
    endpoints: list[str] = []
    if chatterbox_ok:
        try:
            endpoints = discover_api_names()
        except Exception:
            endpoints = []

    lines = [
        "Service status:",
        f"- active backend: {backend}",
        f"- ffmpeg: {'OK' if ffmpeg_ok else 'MISSING'} ({ffmpeg_detail})",
        f"- chatterbox: {'OK' if chatterbox_ok else 'DOWN'} ({chatterbox_detail})",
    ]
    kv_mode, kv_detail = _kv_cache_status()
    lines.append(f"- kv cache compression: {kv_mode} ({kv_detail})")
    if backend == "local_disk_kv":
        lines.append(f"- local_disk_kv (required): {'OK' if local_kv_ok else 'DOWN'} ({local_kv_detail})")
        lines.append(f"- ollama (optional): {'OK' if ollama_ok else 'DOWN'} ({ollama_detail})")
    else:
        lines.append(f"- ollama (required): {'OK' if ollama_ok else 'DOWN'} ({ollama_detail})")
        lines.append(f"- local_disk_kv (optional): {'OK' if local_kv_ok else 'DOWN'} ({local_kv_detail})")

    if endpoints:
        lines.append("- chatterbox endpoints: " + ", ".join(endpoints))
    else:
        lines.append("- chatterbox endpoints: none discovered")
    return "\n".join(lines)


def _running_pid(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _chapter_phase(chapter_num: int) -> str:
    files = _chapter_artifacts(chapter_num)
    if not files["draft"].exists():
        return "drafting"
    if not files["edited"].exists():
        return "editing"
    if not files["final"].exists():
        return "critic/revision"
    if not files["summary"].exists() or not files["tts"].exists():
        return "summary + tts-prep"
    if SETTINGS.pause_before_narration_review and not files["pre_marker"].exists():
        return "paused: pre-narration review"
    if not files["audio"].exists():
        return "narration"
    if SETTINGS.pause_after_chapter_review and not files["post_marker"].exists():
        return "paused: post-chapter review"
    return "complete"


def _env_limit(default_limit: int) -> int:
    raw = os.getenv("CHAPTER_COUNT")
    if not raw:
        return default_limit
    try:
        value = int(raw)
        return value if value > 0 else default_limit
    except ValueError:
        return default_limit


def _normalize_chapter_range(
    requested_start: int,
    requested_last: int,
    available: int,
    legacy_limit: int = 0,
) -> tuple[int, int]:
    if available <= 0:
        return 1, 1

    start = requested_start if requested_start > 0 else 1
    if requested_last > 0:
        last = requested_last
    elif legacy_limit > 0:
        last = legacy_limit
    else:
        last = available

    start = min(max(1, start), available)
    last = min(max(1, last), available)
    if last < start:
        last = start
    return start, last


def _latest_review_packet(chapter_num: int) -> Path | None:
    reviews = ROOT / SETTINGS.reviews_dir
    pre = reviews / f"ch{chapter_num:02d}_pre_narration_review.md"
    post = reviews / f"ch{chapter_num:02d}_post_chapter_review.md"
    if post.exists():
        return post
    if pre.exists():
        return pre
    return None


def _split_sentences_for_hash(text: str) -> list[str]:
    normalized = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [p.strip() for p in parts if p.strip()]


def _tts_source_hash(text: str) -> str:
    sentences = _split_sentences_for_hash(text)
    canonical = "\n".join(sentences)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _segment_manifest_status(chapter_num: int, tts_path: Path) -> str:
    manifest_path = ROOT / "audio" / "segments" / f"ch{chapter_num:02d}" / "manifest.json"
    if not manifest_path.exists():
        return "Segment cache: none"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "Segment cache: manifest invalid JSON"

    completed = len(manifest.get("completed", []))
    sentence_count = int(manifest.get("sentence_count", 0) or 0)
    source_hash = str(manifest.get("source_hash", "") or "")

    if not tts_path.exists():
        return f"Segment cache: {completed}/{sentence_count} segments (tts text missing)"

    text = tts_path.read_text(encoding="utf-8", errors="replace")
    current_hash = _tts_source_hash(text)
    match = source_hash == current_hash if source_hash else False
    return (
        f"Segment cache: {completed}/{sentence_count} segments | "
        f"source match: {'yes' if match else 'no'}"
    )


def get_pipeline_runtime_snapshot(
    start_chapter: int | str | None,
    last_chapter: int | str | None,
    chapter_limit: int | str | None = None,
) -> tuple[str, str, str, str]:
    state = _runner_state()
    pid = state.get("pid")
    running = _running_pid(pid)
    available = _max_known_chapters()
    requested_start = _coerce_int(start_chapter, default=0)
    requested_last = _coerce_int(last_chapter, default=0)
    legacy_limit = _coerce_int(chapter_limit, default=0)

    state_start = _coerce_int(state.get("start_chapter"), default=1)
    state_last = _coerce_int(state.get("last_chapter"), default=0)
    if state_last <= 0:
        state_limit = _coerce_int(state.get("chapter_limit"), default=_env_limit(SETTINGS.chapter_count))
        state_start, state_last = _normalize_chapter_range(1, 0, available, state_limit)

    if requested_start > 0 or requested_last > 0 or legacy_limit > 0:
        run_start, run_last = _normalize_chapter_range(requested_start, requested_last, available, legacy_limit)
    else:
        run_start, run_last = _normalize_chapter_range(state_start, state_last, available)
    total_in_range = max(1, run_last - run_start + 1)

    completed = 0
    for chapter_num in range(run_start, run_last + 1):
        if _chapter_complete(chapter_num):
            completed += 1

    current = _next_pending_chapter(run_start, run_last)
    mode_name = str(state.get("mode") or "")
    selected_chapter = int(state.get("target_chapter") or 0)
    if mode_name.strip().lower().startswith("one") and selected_chapter > 0:
        current = selected_chapter
    if completed >= total_in_range:
        phase = "complete"
    else:
        phase = _chapter_phase(current)
    packet = _latest_review_packet(current)
    started_at = float(state.get("started_at") or 0.0)
    elapsed = _format_hms(time.time() - started_at) if running and started_at > 0 else "00:00"

    timing_text = "ETA unavailable"
    done_durations = [_wav_seconds(_chapter_artifacts(i)["audio"]) for i in range(run_start, run_last + 1)]
    done_durations = [v for v in done_durations if v > 0]
    if done_durations and completed < total_in_range:
        avg = sum(done_durations) / len(done_durations)
        remaining = (total_in_range - completed) * avg
        timing_text = f"Estimated narration remaining: {_format_hms(remaining)}"

    status_lines = [
        f"Runner: {'running' if running else 'idle'}",
        f"Mode: {mode_name or 'n/a'}",
        f"Elapsed: {elapsed}",
        f"Chapter range: {run_start}-{run_last}",
        f"Chapter progress: {completed}/{total_in_range}",
        f"Current chapter: {current}",
        f"Phase: {phase}",
        timing_text,
    ]
    model_profile = str(state.get("model_profile") or "")
    model_name = str(state.get("model_name") or "")
    if model_profile:
        status_lines.append(f"Model profile: {model_profile}")
    if model_name:
        status_lines.append(f"Model: {model_name}")
    kv_mode = str(state.get("kv_cache_mode") or "")
    kv_evidence = str(state.get("kv_cache_evidence") or "")
    if kv_mode or kv_evidence:
        status_lines.append(f"KV cache: {kv_mode or 'unknown'} ({kv_evidence or 'no evidence'})")
    if selected_chapter > 0:
        status_lines.append(
            f"Selected chapter status: ch{selected_chapter:02d} is {_chapter_completion_status(selected_chapter)}"
        )
    if packet:
        status_lines.append(f"Review packet: {packet}")

    files = _chapter_artifacts(current)
    file_lines = [
        f"- draft: {'yes' if files['draft'].exists() else 'no'}",
        f"- edited: {'yes' if files['edited'].exists() else 'no'}",
        f"- final: {'yes' if files['final'].exists() else 'no'}",
        f"- summary: {'yes' if files['summary'].exists() else 'no'}",
        f"- tts text: {'yes' if files['tts'].exists() else 'no'}",
        f"- narration: {'yes' if files['audio'].exists() else 'no'}",
        f"- pre approved: {'yes' if files['pre_marker'].exists() else 'no'}",
        f"- post approved: {'yes' if files['post_marker'].exists() else 'no'}",
        f"- {_segment_manifest_status(current, files['tts'])}",
    ]

    log_tail = ""
    log_path = Path(state.get("log_path") or "")
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            log_tail = "\n".join(lines[-40:])
        except Exception:
            log_tail = "Unable to read run log"
    else:
        log_tail = "No run log yet"

    packet_text = str(packet) if packet else "No review packet currently"
    return "\n".join(status_lines), "\n".join(file_lines), packet_text, log_tail


def _validate_root_pipeline_payload() -> list[str]:
    issues: list[str] = []
    required = {
        "story_bible.json": dict,
        "characters.json": list,
        "chapter_briefs.json": list,
    }
    for name, expected_type in required.items():
        p = ROOT / name
        if not p.exists():
            issues.append(f"{name} missing")
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            issues.append(f"{name} invalid JSON")
            continue
        if not isinstance(data, expected_type):
            issues.append(f"{name} has wrong JSON type")
            continue
        if isinstance(data, (dict, list)) and len(data) == 0:
            issues.append(f"{name} is empty")
    return issues


def start_pipeline_run(
    project_name: str,
    run_mode: str,
    start_chapter: int | str | None,
    last_chapter: int | str | None,
    word_target_min: int | None = None,
    word_target_max: int | None = None,
    narration_speed: float | None = None,
    target_chapter: int | None = None,
    existing_chapter_action: str = "Prompt each time",
    model_profile: str = MODEL_PROFILE_QWEN25_Q5,
    chapter_complete_alert: str = "double_beep",
    chapter_limit: int | str | None = None,
) -> str:
    state = _runner_state()
    if _running_pid(state.get("pid")):
        return f"Pipeline already running (pid={state.get('pid')})."

    chapter_briefs = ROOT / "chapter_briefs.json"
    if not chapter_briefs.exists():
        return "Missing chapter_briefs.json in project root. Sync/convert first."

    payload_issues = _validate_root_pipeline_payload()
    if payload_issues:
        return "Root pipeline files are invalid. Fix/sync before run:\n- " + "\n- ".join(payload_issues)

    try:
        briefs = json.loads(chapter_briefs.read_text(encoding="utf-8"))
        available = len(briefs)
    except json.JSONDecodeError:
        return "chapter_briefs.json is invalid JSON."

    requested_start = _coerce_int(start_chapter, default=0)
    requested_last = _coerce_int(last_chapter, default=0)
    legacy_limit = _coerce_int(chapter_limit, default=0)
    start_num, last_num = _normalize_chapter_range(requested_start, requested_last, max(1, available), legacy_limit)

    mode = (run_mode or "Ask Every Run").strip().lower()
    selected_target = 0
    if mode.startswith("one"):
        requested_target = _coerce_int(target_chapter, default=0)
        if requested_target > 0:
            selected_target = min(max(1, requested_target), max(1, available))
        else:
            selected_target = _next_pending_chapter(start_num, last_num)
        start_num = selected_target
        last_num = selected_target

        action = (existing_chapter_action or "Prompt each time").strip().lower()
        chapter_has_outputs = _chapter_outputs_ready(selected_target)
        if chapter_has_outputs:
            if action.startswith("prompt"):
                return (
                    f"Chapter {selected_target} already has complete outputs. "
                    "Set Existing Chapter Action to Rebuild, Skip, or Cancel, then start again."
                )
            if action.startswith("cancel"):
                return f"Run cancelled: chapter {selected_target} is already complete."
            if action.startswith("skip"):
                return f"Skipped start: chapter {selected_target} is already complete."
            if action.startswith("rebuild"):
                removed = _reset_chapter_outputs(selected_target)
                clear_note = f" Rebuild requested; removed {removed} existing files."
            else:
                clear_note = ""
        else:
            clear_note = ""
    else:
        clear_note = ""

    ffmpeg_ok, ffmpeg_detail = check_ffmpeg()
    if not ffmpeg_ok:
        return f"Run blocked: ffmpeg is required ({ffmpeg_detail})."

    chatterbox_ok, chatterbox_detail = check_chatterbox()
    if not chatterbox_ok:
        return f"Run blocked: chatterbox is unavailable ({chatterbox_detail})."

    backend = _resolved_backend()
    local_kv_ok, local_kv_detail = check_local_disk_kv()
    ollama_ok, ollama_detail = check_ollama()
    if backend == "local_disk_kv" and not local_kv_ok:
        return f"Run blocked: local_disk_kv backend is unavailable ({local_kv_detail})."
    if backend == "openclaw" and not ollama_ok:
        return f"Run blocked: ollama is unavailable ({ollama_detail})."

    env = os.environ.copy()
    env["LLM_BACKEND"] = "local_disk_kv"
    env["USE_LOCAL_DISK_KV"] = "true"
    env["PROJECT_NAME"] = project_name
    endpoint, model_name = _model_profile_runtime(model_profile)
    env["LOCAL_DISK_KV_URL"] = endpoint
    env["LOCAL_DISK_KV_MODEL"] = model_name
    env["LLM_MODEL"] = model_name
    env["EXPANSION_PASSES"] = "0"
    env["CHAPTER_START"] = str(start_num)
    env["CHAPTER_LAST"] = str(last_num)
    env["CHAPTER_COUNT"] = str(last_num)
    manual_voice_tuning = os.getenv("STORY_STUDIO_MANUAL_VOICE_TUNING", "true").strip().lower() in {"1", "true", "yes", "on"}
    env["PAUSE_BEFORE_NARRATION_REVIEW"] = "true" if manual_voice_tuning else "false"
    env["PAUSE_AFTER_CHAPTER_REVIEW"] = "false"
    alert_mode = (chapter_complete_alert or "double_beep").strip().lower().replace(" ", "_")
    if alert_mode not in {"double_beep", "gong", "off"}:
        alert_mode = "double_beep"
    env["CHAPTER_COMPLETE_ALERT"] = alert_mode

    kv_mode, kv_evidence = _kv_cache_status()

    explicit_word_targets = False
    if word_target_min is not None and int(word_target_min) > 0:
        env["WORD_TARGET_MIN"] = str(int(word_target_min))
        explicit_word_targets = True
    if word_target_max is not None and int(word_target_max) > 0:
        env["WORD_TARGET_MAX"] = str(int(word_target_max))
        explicit_word_targets = True
    if explicit_word_targets:
        env["TARGET_MINUTES_MIN"] = "0"
        env["TARGET_MINUTES_MAX"] = "0"

    if narration_speed is not None and float(narration_speed) > 0:
        env["NARRATION_SPEED"] = f"{float(narration_speed):.2f}"
    env["PYTHONUNBUFFERED"] = "1"

    RUNNER_STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNNER_STATE_DIR / f"pipeline_run_{int(time.time())}.log"
    log_file = log_path.open("w", encoding="utf-8")

    process = subprocess.Popen(
        [sys.executable, "-u", "pipeline_novel.py"],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    log_file.close()

    _save_runner_state(
        {
            "pid": process.pid,
            "log_path": str(log_path),
            "started_at": time.time(),
            "mode": run_mode,
            "chapter_limit": last_num,
            "start_chapter": start_num,
            "last_chapter": last_num,
            "target_chapter": selected_target,
            "chapter_complete_alert": alert_mode,
            "kv_cache_mode": kv_mode,
            "kv_cache_evidence": kv_evidence,
            "project": project_name,
            "model_profile": model_profile,
            "model_name": model_name,
            "model_url": endpoint,
        }
    )
    update_session(project_name, active_stage="running", pause_reason="", last_run_started_at=str(int(time.time())))
    return (
        f"Pipeline started (pid={process.pid}) for chapters {start_num}-{last_num}. "
        f"Alert: {alert_mode}. KV cache: {kv_mode or 'unknown'}. Log: {log_path}.{clear_note}"
    )


def stop_pipeline_run(project_name: str) -> str:
    state = _runner_state()
    pid = state.get("pid")
    if not _running_pid(pid):
        return "Pipeline is not running."
    try:
        os.kill(pid, 15)
        update_session(project_name, active_stage="stopped", pause_reason="manual stop", last_run_finished_at=str(int(time.time())))
        return f"Stop signal sent to pipeline pid={pid}."
    except OSError as exc:
        return f"Failed to stop pipeline pid={pid}: {exc}"


def switch_project_cleanup(project_name: str, force_stop: bool) -> str:
    state = _runner_state()
    pid = state.get("pid")
    running = _running_pid(pid)

    stop_note = "not running"
    if running and not force_stop:
        return f"Pipeline is running (pid={pid}). Enable force stop for switch cleanup."

    if running and force_stop:
        try:
            os.kill(pid, 15)
            stop_note = f"stopped pid={pid}"
        except OSError as exc:
            stop_note = f"failed to stop pid={pid}: {exc}"

    removed = _clear_root_runtime_outputs()
    _reset_runner_state()

    if project_name:
        update_session(project_name, active_stage="idle", pause_reason="", last_run_finished_at=str(int(time.time())))

    return f"Switch cleanup complete: removed {removed} files; process {stop_note}. Sync project JSON and voice before run."


def reset_pipeline_run(
    project_name: str,
    reset_scope: str,
    chapter_num: int,
    force_stop: bool,
    confirm_all: bool,
) -> str:
    scope = (reset_scope or "").strip()
    allowed = {"Current Chapter", "All Chapters", "Runner State Only"}
    if scope not in allowed:
        return "Invalid reset scope. Use Current Chapter, All Chapters, or Runner State Only."

    state = _runner_state()
    pid = state.get("pid")
    running = _running_pid(pid)

    if running and not force_stop:
        return f"Pipeline is running (pid={pid}). Enable force stop to reset."

    stop_note = "not running"
    if running and force_stop:
        try:
            os.kill(pid, 15)
            stop_note = f"stopped pid={pid}"
        except OSError as exc:
            stop_note = f"failed to stop pid={pid}: {exc}"

    if scope == "All Chapters" and not confirm_all:
        return "Please confirm All Chapters reset before proceeding."

    removed = 0
    checkpoint_file = ROOT / SETTINGS.checkpoint_dir / f"{project_name}.json"
    if scope == "Current Chapter":
        target = max(1, int(chapter_num))
        removed += _reset_chapter_outputs(target)
        if _safe_unlink(checkpoint_file):
            removed += 1
        _reset_runner_state()
    elif scope == "All Chapters":
        for num in range(1, _max_known_chapters() + 1):
            removed += _reset_chapter_outputs(num)
        removed += _safe_rmtree(ROOT / "audio" / "segments")
        if _safe_unlink(checkpoint_file):
            removed += 1
        for path in RUNNER_STATE_DIR.glob("pipeline_run_*.log"):
            if _safe_unlink(path):
                removed += 1
        _reset_runner_state()
    else:
        if _safe_unlink(checkpoint_file):
            removed += 1
        _reset_runner_state()

    if project_name:
        update_session(project_name, active_stage="idle", pause_reason="", last_run_finished_at=str(int(time.time())))

    return f"Reset complete ({scope}): removed {removed} files; process {stop_note}."


def approve_review_marker(chapter_num: int, stage: str) -> str:
    allowed = {"pre_narration", "post_chapter"}
    if stage not in allowed:
        return "Stage must be pre_narration or post_chapter."
    marker = ROOT / SETTINGS.reviews_dir / f"ch{chapter_num:02d}_{stage}.approved"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("approved\n", encoding="utf-8")
    return f"Created marker: {marker}"


def load_narration_text(chapter_num: int) -> str:
    path = ROOT / "chapters" / f"ch{chapter_num:02d}_tts.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def save_narration_text(chapter_num: int, text: str) -> str:
    path = ROOT / "chapters" / f"ch{chapter_num:02d}_tts.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")
    return f"Saved narration text: {path.name}"



def list_downloadable_files(project_name: str) -> list[str]:
    if not project_name:
        return []

    paths = initialize_project(project_name)
    out: list[str] = []
    for folder in [paths.json_dir, paths.chapters_dir, paths.reviews_dir, paths.audio_dir, paths.voices_dir]:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                out.append(str(p.relative_to(paths.root)))
    return out



def get_download_path(project_name: str, rel_path: str) -> str | None:
    if not project_name or not rel_path:
        return None

    paths = initialize_project(project_name)
    p = (paths.root / rel_path).resolve()
    if not str(p).startswith(str(paths.root.resolve())):
        return None
    if not p.exists() or not p.is_file():
        return None
    return str(p)
