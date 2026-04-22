from __future__ import annotations

import json
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

    @staticmethod
    def _dump(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, ensure_ascii=True)


class ArchitectAgent(BaseAgent):
    role = "architect"

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        return (
            "You are the story architect for a novel pipeline.\n"
            "Produce a concise architecture note that identifies the central dramatic engine,\n"
            "the protagonist pressure line, and the chapter-to-chapter escalation path.\n"
            "Use plain text only. No markdown tables.\n\n"
            f"Project: {context.project_name}\n"
            f"Inputs:\n{self._dump(prompt_vars)}"
        )


class PlannerAgent(BaseAgent):
    role = "planner"

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        return (
            "You are planning the next chapter of a novel.\n"
            "Return a clean scene plan only.\n"
            "Requirements:\n"
            "- 3 to 6 bullet points\n"
            "- each bullet must describe a concrete beat with action, obstacle, and consequence\n"
            "- no prose draft, no analysis, no explanation of your process\n"
            "- preserve the chapter brief facts exactly\n\n"
            f"Chapter: {context.current_chapter}\n"
            f"Planning inputs:\n{self._dump(prompt_vars)}"
        )


class WriterAgent(BaseAgent):
    role = "writer"

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        brief = prompt_vars.get("brief", {})
        style = prompt_vars.get("style", "")
        scene_plan = prompt_vars.get("scene_plan", "")
        previous_summary = prompt_vars.get("previous_summary", "")
        characters = prompt_vars.get("characters", [])
        instruction = prompt_vars.get("instruction", "")
        word_target = brief.get("word_target", 2200) if isinstance(brief, dict) else 2200
        return (
            "Write immersive chapter prose for a novel.\n"
            "Hard requirements:\n"
            "- Output only the chapter prose\n"
            "- No headings, labels, scene numbers, notes, bullets, or commentary\n"
            "- No prompt echo, no JSON, no explanation of choices\n"
            "- Use past tense and stay in the specified POV\n"
            "- End on a concrete consequence-driven hook\n"
            f"- Target length: about {word_target} words\n\n"
            f"Chapter: {context.current_chapter}\n"
            f"Scene slot: {context.current_scene}\n"
            f"Writer instruction: {instruction}\n"
            f"Style direction:\n{self._dump(style)}\n\n"
            f"Previous chapter summary:\n{self._dump(previous_summary)}\n\n"
            f"Character roster:\n{self._dump(characters)}\n\n"
            f"Chapter brief:\n{self._dump(brief)}\n\n"
            f"Scene plan:\n{self._dump(scene_plan)}\n"
        )


class EditorAgent(BaseAgent):
    role = "editor"

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        return (
            "You are editing a novel chapter draft.\n"
            "Return only the revised chapter prose.\n"
            "Requirements:\n"
            "- preserve plot facts and continuity\n"
            "- remove repetition and prompt artifacts\n"
            "- keep the prose immersive and natural\n"
            "- do not add headings, labels, commentary, or analysis\n"
            "- if the draft contains prompt echo or metadata, strip it completely\n\n"
            f"Chapter: {context.current_chapter}\n"
            f"Scene slot: {context.current_scene}\n"
            f"Editing inputs:\n{self._dump(prompt_vars)}"
        )


class MemoryManagerAgent(BaseAgent):
    role = "memory_manager"

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        return (
            "Summarize the chapter for continuity memory.\n"
            "Return plain text only. Focus on plot movement, character changes, and unresolved threads.\n\n"
            f"Chapter: {context.current_chapter}\n"
            f"Inputs:\n{self._dump(prompt_vars)}"
        )
