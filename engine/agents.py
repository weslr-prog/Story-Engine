from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .orchestrator import PipelineContext


@dataclass
class AgentOutput:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceClient(Protocol):
    def chat(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        ...


class BaseAgent:
    role = "base"

    def __init__(self, client: InferenceClient | None = None) -> None:
        self.client = client

    def run(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> AgentOutput:
        prompt = self.build_prompt(context, prompt_vars)
        if self.client is None:
            return AgentOutput(role=self.role, content=prompt, metadata={"mode": "dry-run"})
        started = time.time()
        content = self.client.chat(prompt)
        latency_s = round(time.time() - started, 3)
        diagnostics = {}
        if hasattr(self.client, "pop_last_diagnostics"):
            try:
                diagnostics = getattr(self.client, "pop_last_diagnostics")() or {}
            except Exception:
                diagnostics = {}
        payload = {
            "mode": "live",
            "latency_s": latency_s,
            "diagnostics": diagnostics,
        }
        context.record_inference(self.role, payload)
        return AgentOutput(role=self.role, content=content, metadata=payload)

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        raise NotImplementedError


class ArchitectAgent(BaseAgent):
    role = "architect"

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        return f"Architect story foundation for project {context.project_name}: {prompt_vars}"


class PlannerAgent(BaseAgent):
    role = "planner"

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        return f"Plan chapter {context.current_chapter} scene beats: {prompt_vars}"


class WriterAgent(BaseAgent):
    role = "writer"

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        return f"Write scene {context.current_scene} of chapter {context.current_chapter}: {prompt_vars}"


class EditorAgent(BaseAgent):
    role = "editor"

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        return f"Edit scene {context.current_scene} of chapter {context.current_chapter}: {prompt_vars}"


class MemoryManagerAgent(BaseAgent):
    role = "memory_manager"

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        return f"Update canonical memory after chapter {context.current_chapter} scene {context.current_scene}: {prompt_vars}"
