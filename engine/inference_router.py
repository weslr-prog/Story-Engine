from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .local_llm import HypuraClient, OllamaClient, _log


@dataclass
class InferenceRouter:
    structural: OllamaClient
    prose: HypuraClient
    fallback: OllamaClient | None = None

    def health_check(self) -> dict[str, Any]:
        structural_ok = self.structural.ping()
        prose_ok = self.prose.ping()
        inventory = self.prose.model_inventory() if prose_ok else {"ok": False, "models": []}
        report = {
            "structural_ok": structural_ok,
            "prose_ok": prose_ok,
            "fallback_configured": self.fallback is not None,
            "prose_inventory": inventory,
        }
        _log("ROUTER", f"health report structural_ok={structural_ok} prose_ok={prose_ok} fallback={self.fallback is not None}")
        return report

    def route_to_structural(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.4) -> str:
        return self.structural.chat(prompt, max_tokens=max_tokens, temperature=temperature)

    def route_to_prose(self, prompt: str, max_tokens: int = 1400, temperature: float = 0.7) -> str:
        try:
            return self.prose.chat(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            if self.fallback is None:
                _log("ROUTER", f"prose lane failure with no fallback error={exc}")
                raise
            _log("ROUTER", f"prose fallback activated error={exc}")
            return self.fallback.chat(prompt, max_tokens=max_tokens, temperature=temperature)
