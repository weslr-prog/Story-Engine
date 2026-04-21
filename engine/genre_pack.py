from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from .config import SETTINGS


class GenrePack(BaseModel):
    genre_name: str
    writer_prefix: str
    editor_prefix: str
    structural_fields: list[str]
    content_rules: list[str]


def load_genre_pack(name: str) -> GenrePack:
    pack_path = Path(SETTINGS.genre_pack_dir) / f"{name}.yaml"
    data = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    return GenrePack.model_validate(data)
