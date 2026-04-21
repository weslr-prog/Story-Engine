from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_ROOT = ROOT / "projects"
APP_STATE_FILE = PROJECTS_ROOT / ".studio_state.json"


INPUT_FILES = {
    "dna": "Story DNA Summary.txt",
    "bible": "Story Bible.txt",
    "blueprint": "Chapter Blueprint.txt",
    "style_guide": "style_guide.txt",
    "consistency": "consistency_checklist.txt",
}


@dataclass(frozen=True)
class ProjectPaths:
    name: str
    root: Path
    inputs_dir: Path
    voices_dir: Path
    json_dir: Path
    chapters_dir: Path
    audio_dir: Path
    reviews_dir: Path
    exports_dir: Path
    output_dir: Path
    db_dir: Path
    db_path: Path
    state_dir: Path
    checkpoint_dir: Path
    session_file: Path
    lock_file: Path



def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()



def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    slug = slug.strip("-._")
    if not slug:
        raise ValueError("Project name cannot be empty.")
    return slug



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



def ensure_projects_root() -> None:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)



def project_paths(project_name: str) -> ProjectPaths:
    ensure_projects_root()
    slug = _slugify(project_name)
    root = PROJECTS_ROOT / slug
    return ProjectPaths(
        name=slug,
        root=root,
        inputs_dir=root / "inputs",
        voices_dir=root / "voices",
        json_dir=root / "json",
        chapters_dir=root / "chapters",
        audio_dir=root / "audio",
        reviews_dir=root / "reviews",
        exports_dir=root / "exports",
        output_dir=root / "output",
        db_dir=root / "db",
        db_path=root / "db" / "story_bible.sqlite",
        state_dir=root / ".state",
        checkpoint_dir=root / ".state" / "checkpoints",
        session_file=root / ".state" / "session.json",
        lock_file=root / ".state" / "pipeline.lock",
    )



def initialize_project(project_name: str) -> ProjectPaths:
    paths = project_paths(project_name)
    for d in [
        paths.inputs_dir,
        paths.voices_dir,
        paths.json_dir,
        paths.chapters_dir,
        paths.audio_dir,
        paths.reviews_dir,
        paths.exports_dir,
        paths.output_dir,
        paths.db_dir,
        paths.state_dir,
        paths.checkpoint_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    meta_file = paths.state_dir / "project.json"
    if not meta_file.exists():
        _write_json(
            meta_file,
            {
                "name": paths.name,
                "display_name": project_name.strip() or paths.name,
                "created_at": now_iso(),
                "updated_at": now_iso(),
            },
        )

    if not paths.session_file.exists():
        _write_json(
            paths.session_file,
            {
                "project": paths.name,
                "active_stage": "idle",
                "active_voice": "",
                "pause_reason": "",
                "last_run_started_at": "",
                "last_run_finished_at": "",
                "updated_at": now_iso(),
            },
        )

    return paths



def list_projects() -> list[str]:
    ensure_projects_root()
    names: list[str] = []
    for p in sorted(PROJECTS_ROOT.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            names.append(p.name)
    return names



def get_active_project() -> str:
    state = _read_json(APP_STATE_FILE, {})
    active = state.get("active_project", "")
    if active and (PROJECTS_ROOT / active).exists():
        return active
    projects = list_projects()
    return projects[0] if projects else ""



def set_active_project(project_name: str) -> str:
    paths = initialize_project(project_name)
    _write_json(
        APP_STATE_FILE,
        {
            "active_project": paths.name,
            "updated_at": now_iso(),
        },
    )
    return paths.name



def update_session(project_name: str, **fields: Any) -> None:
    paths = initialize_project(project_name)
    data = _read_json(paths.session_file, {})
    data.update(fields)
    data["updated_at"] = now_iso()
    _write_json(paths.session_file, data)



def is_locked(project_name: str) -> bool:
    paths = project_paths(project_name)
    return paths.lock_file.exists()



def acquire_lock(project_name: str, reason: str) -> None:
    paths = initialize_project(project_name)
    if paths.lock_file.exists():
        raise RuntimeError(f"Project '{project_name}' is already running.")
    paths.lock_file.write_text(json.dumps({"reason": reason, "created_at": now_iso()}, indent=2), encoding="utf-8")



def release_lock(project_name: str) -> None:
    paths = project_paths(project_name)
    if paths.lock_file.exists():
        paths.lock_file.unlink()



def input_path(project_name: str, input_key: str) -> Path:
    if input_key not in INPUT_FILES:
        raise KeyError(f"Unknown input key: {input_key}")
    paths = initialize_project(project_name)
    return paths.inputs_dir / INPUT_FILES[input_key]
