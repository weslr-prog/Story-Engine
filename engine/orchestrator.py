from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FSMState(str, Enum):
    INIT = "init"
    OUTLINE = "outline"
    CHAPTER_PLAN = "chapter_plan"
    SCENE_WRITE = "scene_write"
    SCENE_EDIT = "scene_edit"
    MEMORY_UPDATE = "memory_update"
    CHECKPOINT = "checkpoint"
    NEXT_SCENE = "next_scene"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PipelineContext:
    project_name: str
    current_chapter: int = 1
    current_scene: int = 1
    chapter_limit: int | None = None
    state: FSMState = FSMState.INIT
    word_count: int = 0
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    inference_log: dict[str, Any] = field(default_factory=dict)

    def record_inference(self, role: str, payload: dict[str, Any]) -> None:
        agents = self.inference_log.setdefault("agents", {})
        agents[role] = payload


class Orchestrator:
    def __init__(self) -> None:
        self._transitions = {
            FSMState.INIT: FSMState.OUTLINE,
            FSMState.OUTLINE: FSMState.CHAPTER_PLAN,
            FSMState.CHAPTER_PLAN: FSMState.SCENE_WRITE,
            FSMState.SCENE_WRITE: FSMState.SCENE_EDIT,
            FSMState.SCENE_EDIT: FSMState.MEMORY_UPDATE,
            FSMState.MEMORY_UPDATE: FSMState.CHECKPOINT,
            FSMState.CHECKPOINT: FSMState.NEXT_SCENE,
        }

    def advance(self, context: PipelineContext) -> PipelineContext:
        if context.state in {FSMState.COMPLETE, FSMState.FAILED}:
            return context

        next_state = self._transitions.get(context.state)
        if next_state is None and context.state == FSMState.NEXT_SCENE:
            context.current_scene += 1
            context.state = FSMState.SCENE_WRITE
            return context
        if next_state is None:
            context.state = FSMState.FAILED
            context.metadata["error"] = f"No transition defined for state: {context.state}"
            return context

        context.state = next_state
        return context
