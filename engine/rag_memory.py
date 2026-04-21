from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import chromadb

from .config import SETTINGS


@dataclass
class MemoryChunk:
    document: str
    metadata: dict[str, Any]


class _LocalHashEmbedding:
    """Simple deterministic embedding to keep local/offline Chroma operations stable."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            vec = [0.0] * self.dimensions
            for token in (text or "").lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:2], "big") % self.dimensions
                sign = 1.0 if (digest[2] % 2 == 0) else -1.0
                vec[idx] += sign
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


class StoryMemory:
    def __init__(self, path: str | None = None) -> None:
        db_path = path or str(SETTINGS.chroma_db_path)
        self.client = chromadb.PersistentClient(path=db_path)
        self._embedder = _LocalHashEmbedding()
        self.scenes = self.client.get_or_create_collection("scenes")
        self.characters = self.client.get_or_create_collection("characters")
        self.world_facts = self.client.get_or_create_collection("world_facts")

    def add_scene(self, project_id: str, chapter: int, scene_num: int, text: str) -> None:
        started = time.time()
        chunk_id = f"{project_id}:ch{chapter}:sc{scene_num}"
        self.scenes.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=self._embedder([text]),
            metadatas=[{"project_id": project_id, "chapter": chapter, "scene_num": scene_num}],
        )
        elapsed = round(time.time() - started, 3)
        try:
            total = self.scenes.count()
        except Exception:
            total = -1
        print(f"[RAG] scenes upsert project={project_id} chapter={chapter} elapsed={elapsed}s total={total}", flush=True)

    def update_character(self, project_id: str, name: str, text: str) -> None:
        started = time.time()
        chunk_id = f"{project_id}:character:{name}"
        self.characters.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=self._embedder([text]),
            metadatas=[{"project_id": project_id, "name": name}],
        )
        elapsed = round(time.time() - started, 3)
        try:
            total = self.characters.count()
        except Exception:
            total = -1
        print(f"[RAG] characters upsert project={project_id} name={name} elapsed={elapsed}s total={total}", flush=True)

    def add_world_fact(self, project_id: str, key: str, text: str) -> None:
        started = time.time()
        chunk_id = f"{project_id}:world:{key}"
        self.world_facts.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=self._embedder([text]),
            metadatas=[{"project_id": project_id, "key": key}],
        )
        elapsed = round(time.time() - started, 3)
        try:
            total = self.world_facts.count()
        except Exception:
            total = -1
        print(f"[RAG] world_facts upsert project={project_id} key={key} elapsed={elapsed}s total={total}", flush=True)

    def query_relevant(self, query: str, collection: str = "scenes", limit: int = 5) -> list[MemoryChunk]:
        target = {
            "scenes": self.scenes,
            "characters": self.characters,
            "world_facts": self.world_facts,
        }[collection]
        result = target.query(query_embeddings=self._embedder([query]), n_results=limit)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [MemoryChunk(document=doc, metadata=meta) for doc, meta in zip(documents, metadatas)]
