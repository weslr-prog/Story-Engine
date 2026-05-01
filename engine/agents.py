from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .orchestrator import PipelineContext


def _extract_prose_only(text: str) -> str:
    """
    Extract only the narrative prose from LLM output, removing:
    - JSON structures and metadata dumps
    - Model artifacts and special tokens
    - System information and metadata lines
    
    Strategy:
    1. First, look for explicit markers (BEGIN PROSE/END PROSE or BEGIN REVISED PROSE/END REVISED PROSE)
    2. If markers found, extract content between them
    3. Otherwise, use heuristic filtering to identify prose lines
    
    Returns the cleaned prose content.
    """
    if not text:
        return ""
    
    # Try explicit marker extraction first
    prose_markers = [
        (r"=== BEGIN PROSE ===\n(.*?)\n=== END PROSE ===", re.DOTALL),
        (r"=== BEGIN REVISED PROSE ===\n(.*?)\n=== END REVISED PROSE ===", re.DOTALL),
        (r"===\s*BEGIN\s+(?:PROSE|REVISED PROSE)\s*===\n(.*?)\n===\s*END\s+(?:PROSE|REVISED PROSE)\s*===", re.DOTALL | re.IGNORECASE),
    ]
    
    for pattern, flags in prose_markers:
        match = re.search(pattern, text, flags)
        if match:
            extracted = match.group(1).strip()
            # Must be substantial AND must not be a prompt-echo (LLM echoing its own system prompt)
            if (extracted
                    and len(extracted) > 200
                    and "editing a novel chapter" not in extracted.lower()
                    and "editor's prefix" not in extracted.lower()
                    and "you are editing" not in extracted.lower()):
                return extracted
    
    # Fallback: heuristic filtering
    lines = text.split("\n")
    result = []
    in_json = False
    json_depth = 0
    
    # List of metadata/instruction keywords that indicate non-prose lines
    metadata_keywords = {
        "instruction:", "editor_prefix:", "editing inputs:", "planner_prefix:",
        "planning inputs:", "inputs:", "drafter:", "writer instruction:",
        "chapter brief:", "scene plan:", "character roster:", "style direction:",
        "previous chapter summary:", "writing inputs:", "scene slot:",
        "requirements:", "hard requirements:", "return only", "output only",
        "goals?:", "goal", "launching", "approval", "begin prose", "end prose",
        "begin revised", "end revised",
        # Prompt-echo patterns: LLM echoing its own system prompt
        "editing a novel chapter", "you are editing", "editor's prefix",
        "i'm editing", "chapter:",
    }
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines at the beginning until we find content
        if not result and not stripped:
            continue
        
        # Track JSON nesting
        json_depth += stripped.count("{") + stripped.count("[")
        json_depth -= stripped.count("}") + stripped.count("]")
        in_json = json_depth > 0
        
        # Skip lines while inside JSON structure
        if in_json:
            continue
        
        # Skip lines that start JSON structures
        if stripped.startswith("{") or stripped.startswith("["):
            continue
        
        # Skip model artifacts and special tokens
        if any(x in stripped for x in [
            "<|user|>", "<|assistant|>", "<|system|>",
            "[INST]", "[/INST]", "<<<", ">>>",
            "Write a", "Copy/Paste", "Example for",
            "Then rerun", "touch reviews", "bash scripts",
            "```bash", "```", "## ",
        ]):
            continue
        
        # Detect metadata lines with specific patterns
        lower_stripped = stripped.lower()
        is_metadata = False
        
        # Check for metadata keywords
        for keyword in metadata_keywords:
            if keyword in lower_stripped:
                is_metadata = True
                break
        
        # Check for JSON key-value pattern (but allow dialogue)
        if re.match(r'^\s*"[^"]{2,30}"\s*:\s*["{[]', stripped):
            is_metadata = True
        
        # Skip bullet points (requirements/instructions)
        if stripped.startswith("-"):
            continue
        
        # Skip chapter headings (any chapter number)
        if re.match(r'^chapter\s+\d+', stripped, re.IGNORECASE):
            continue
        
        # Skip scene markers
        if re.match(r'^scene\s*:', stripped, re.IGNORECASE):
            continue
        
        # Keep narrative content
        if stripped and not is_metadata and not in_json:
            result.append(line)
    
    # Join and clean excessive whitespace
    prose = "\n".join(result).strip()
    
    # Remove any remaining JSON at the start
    if "{" in prose[:100]:
        # Find first closing brace and remove everything up to and including it
        match = re.search(r'[}]\s*\n', prose)
        if match:
            prose = prose[match.end():].strip()
    
    # Final cleanup: ensure we have actual content
    prose = prose.strip()
    
    # If result is empty or too short, return empty string
    # This signals to the caller that NO PROSE WAS FOUND
    if not prose or len(prose) < 50:
        return ""
    
    return prose.strip()


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
        max_tokens = int(prompt_vars.get("max_tokens", 1024) or 1024)
        temperature = float(prompt_vars.get("temperature", 0.7) or 0.7)
        content = self.client.chat(prompt, max_tokens=max_tokens, temperature=temperature)
        
        # Clean prose-generating agents of metadata/JSON artifacts
        if self.role in ("writer", "editor", "planner"):
            content = _extract_prose_only(content)
        
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
        scene_beat = prompt_vars.get("scene_beat", "")
        prior_scenes = prompt_vars.get("prior_scenes", "")
        draft = prompt_vars.get("draft", "")
        previous_summary = prompt_vars.get("previous_summary", "")
        characters = prompt_vars.get("characters", [])
        instruction = prompt_vars.get("instruction", "")
        word_target = prompt_vars.get("word_target")
        if not word_target and isinstance(brief, dict):
            word_target = brief.get("word_target", 2200)
        word_target = int(word_target or 2200)
        return (
            "Write immersive chapter prose for a novel.\n"
            "Hard requirements:\n"
            "- Output ONLY the chapter prose between the markers below\n"
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
            f"Scene plan:\n{self._dump(scene_plan)}\n\n"
            f"Current scene beat:\n{self._dump(scene_beat)}\n\n"
            f"Prior scenes:\n{self._dump(prior_scenes)}\n\n"
            f"Current draft to continue/expand:\n{self._dump(draft)}\n\n"
            "=== BEGIN PROSE ===\n"
            "(Write only the chapter prose here)\n"
            "=== END PROSE ==="
        )


class EditorAgent(BaseAgent):
    role = "editor"

    def build_prompt(self, context: PipelineContext, prompt_vars: dict[str, Any]) -> str:
        return (
            "You are editing a novel chapter draft.\n"
            "Return ONLY the revised chapter prose between the markers below.\n"
            "Requirements:\n"
            "- preserve plot facts and continuity\n"
            "- remove repetition and prompt artifacts\n"
            "- keep the prose immersive and natural\n"
            "- do not add headings, labels, commentary, or analysis\n"
            "- if the draft contains prompt echo or metadata, strip it completely\n\n"
            f"Chapter: {context.current_chapter}\n"
            f"Scene slot: {context.current_scene}\n"
            f"Editing inputs:\n{self._dump(prompt_vars)}\n\n"
            "=== BEGIN REVISED PROSE ===\n"
            "(Write only the edited chapter prose here)\n"
            "=== END REVISED PROSE ==="
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
