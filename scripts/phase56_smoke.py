#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SETTINGS
from engine.rag_memory import StoryMemory
from engine.story_bible_db import Project, WorldFact, make_session_factory


def main() -> int:
    db_path = ROOT / SETTINGS.story_db_filename
    sf = make_session_factory(str(db_path))

    with sf() as session:
        project = session.query(Project).filter(Project.name == "phase56_smoke").one_or_none()
        if project is None:
            project = Project(name="phase56_smoke", genre_pack=SETTINGS.default_genre_pack, status="smoke")
            session.add(project)
            session.flush()
        wf = session.query(WorldFact).filter(WorldFact.project_id == project.id, WorldFact.key == "smoke_fact").one_or_none()
        if wf is None:
            session.add(WorldFact(project_id=project.id, key="smoke_fact", value="Smoke fact persisted", scope="global"))
        session.commit()

    mem = StoryMemory()
    mem.add_world_fact("phase56_smoke", "smoke_fact", "Smoke fact persisted")
    hits = mem.query_relevant("Smoke fact persisted", collection="world_facts", limit=1)

    payload = {
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "rag_hit_count": len(hits),
        "rag_hit_preview": hits[0].document if hits else "",
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["db_exists"] and payload["rag_hit_count"] >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
