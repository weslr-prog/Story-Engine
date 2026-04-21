from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests

try:
    from config import SETTINGS
except Exception:
    SETTINGS = None


def _setting(name: str, default: Any) -> Any:
    if SETTINGS is None:
        return default
    return getattr(SETTINGS, name, default)


DEFAULT_TIMEOUT = int(_setting("llm_call_timeout_seconds", 300))
DEFAULT_RETRIES = int(_setting("llm_max_retries", 2))
DEFAULT_RETRY_DELAY = float(_setting("llm_retry_delay_seconds", 2.0))
THINKING_OVERHEAD_TOKENS = int(_setting("llm_thinking_overhead", 0))
ALLOW_MODEL_FALLBACK = bool(_setting("allow_model_fallback", True))


def _log(channel: str, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [{channel}] {message}", flush=True)


@dataclass
class ChatResult:
    content: str
    raw: dict[str, Any]


class BaseInferenceClient:
    channel = "LLM"

    def __init__(self, url: str, model: str) -> None:
        self._url = url
        self._model = model
        self._temperature = 0.7
        self._max_output_tokens = 1024
        self._headers = {"Content-Type": "application/json"}
        self._last_diagnostics: dict[str, Any] = {}

    def set_role(self, role: str) -> None:
        del role

    def set_temperature(self, temp: float) -> None:
        self._temperature = temp

    def set_max_output_tokens(self, max_tokens: int) -> None:
        self._max_output_tokens = max_tokens

    def apply_preset(self, preset_name: str) -> None:
        del preset_name

    def _endpoint(self) -> str:
        return self._url

    def _model_name(self) -> str:
        return self._model

    def _host_root(self) -> str:
        if "/v1/" in self._url:
            return self._url.split("/v1/", 1)[0] + "/"
        if self._url.endswith("/api/chat"):
            return self._url.rsplit("/api/chat", 1)[0] + "/"
        if "/api/" in self._url:
            return self._url.split("/api/", 1)[0] + "/"
        return self._url.rsplit("/", 1)[0] + "/"

    def ping(self) -> bool:
        try:
            response = requests.get(self._host_root(), timeout=4)
            return response.ok
        except requests.RequestException:
            return False

    def _payload(self, prompt: str, model_name: str, max_tokens: int, temperature: float) -> dict[str, Any]:
        return {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "request_id": str(uuid.uuid4()),
        }

    def _extract_content(self, data: dict[str, Any]) -> str:
        message = data["choices"][0]["message"]
        return (message.get("content") or "").strip()

    def pop_last_diagnostics(self) -> dict[str, Any]:
        payload = dict(self._last_diagnostics)
        self._last_diagnostics = {}
        return payload

    @staticmethod
    def _error_category(exc: Exception) -> str:
        if isinstance(exc, requests.Timeout):
            return "timeout"
        if isinstance(exc, requests.ConnectionError):
            return "connection"
        if isinstance(exc, requests.HTTPError):
            return "http"
        if isinstance(exc, KeyError):
            return "schema"
        if isinstance(exc, RuntimeError):
            return "runtime"
        return "unknown"

    def invoke(self, prompt: str) -> ChatResult:
        endpoint = self._endpoint()
        model_candidates = [self._model_name()]
        fallback_model = os.getenv("LLM_MODEL", "").strip()
        if ALLOW_MODEL_FALLBACK and fallback_model and fallback_model not in model_candidates:
            model_candidates.append(fallback_model)

        started = time.time()
        last_error: Exception | None = None
        diagnostics: dict[str, Any] = {
            "endpoint": endpoint,
            "model_candidates": model_candidates,
            "attempts": [],
            "configured_model": self._model_name(),
        }
        for model_name in model_candidates:
            payload = self._payload(
                prompt,
                model_name,
                self._max_output_tokens + THINKING_OVERHEAD_TOKENS,
                self._temperature,
            )
            for attempt in range(1, DEFAULT_RETRIES + 2):
                request_started = time.time()
                try:
                    _log(
                        self.channel,
                        f"request start model={model_name} attempt={attempt}/{DEFAULT_RETRIES + 1} endpoint={endpoint}",
                    )
                    response = requests.post(endpoint, headers=self._headers, json=payload, timeout=DEFAULT_TIMEOUT)
                    response.raise_for_status()
                    data = response.json()
                    content = self._extract_content(data)
                    if not content:
                        raise RuntimeError("Model returned empty content.")
                    latency_s = round(time.time() - request_started, 3)
                    diagnostics["attempts"].append(
                        {
                            "model": model_name,
                            "attempt": attempt,
                            "status": "success",
                            "http_status": response.status_code,
                            "latency_s": latency_s,
                        }
                    )
                    diagnostics["selected_model"] = model_name
                    diagnostics["fallback_used"] = model_name != self._model_name()
                    diagnostics["total_latency_s"] = round(time.time() - started, 3)
                    self._last_diagnostics = diagnostics
                    _log(
                        self.channel,
                        f"request success model={model_name} status={response.status_code} latency={latency_s}s fallback={diagnostics['fallback_used']}",
                    )
                    return ChatResult(content=content, raw=data)
                except (requests.RequestException, KeyError, RuntimeError) as exc:
                    last_error = exc
                    category = self._error_category(exc)
                    latency_s = round(time.time() - request_started, 3)
                    diagnostics["attempts"].append(
                        {
                            "model": model_name,
                            "attempt": attempt,
                            "status": "error",
                            "error": str(exc),
                            "error_category": category,
                            "latency_s": latency_s,
                        }
                    )
                    _log(
                        self.channel,
                        f"request error model={model_name} attempt={attempt} category={category} latency={latency_s}s error={exc}",
                    )
                    if attempt <= DEFAULT_RETRIES:
                        time.sleep(DEFAULT_RETRY_DELAY * attempt)

        diagnostics["total_latency_s"] = round(time.time() - started, 3)
        diagnostics["final_error"] = str(last_error) if last_error is not None else "unknown"
        self._last_diagnostics = diagnostics
        raise RuntimeError(f"{self.channel} request failed: {last_error}") from last_error

    def chat(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        self.set_max_output_tokens(max_tokens)
        self.set_temperature(temperature)
        return self.invoke(prompt).content


class OllamaClient(BaseInferenceClient):
    channel = "OLLAMA"

    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        super().__init__(
            url or str(_setting("ollama_url", os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions"))),
            model or str(_setting("ollama_model", os.getenv("OLLAMA_MODEL", "phi3.5:3.8b-mini-instruct-q5_K_M"))),
        )


class HypuraClient(BaseInferenceClient):
    channel = "HYPURA"

    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        resolved_url = url or str(_setting("hypura_url", os.getenv("HYPURA_URL", "http://127.0.0.1:11435/api/chat")))
        # Hypura is Ollama-compatible; normalize OpenAI-style configs to native endpoint.
        if resolved_url.endswith("/v1/chat/completions"):
            resolved_url = resolved_url.rsplit("/v1/chat/completions", 1)[0] + "/api/chat"
        super().__init__(
            resolved_url,
            model or str(_setting("hypura_model", os.getenv("HYPURA_MODEL", "mixtral-8x7b-instruct-turboquant"))),
        )
        self._last_tags: dict[str, Any] = {}

    def _payload(self, prompt: str, model_name: str, max_tokens: int, temperature: float) -> dict[str, Any]:
        return {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": int(_setting("llm_num_ctx", 8192)),
            },
        }

    def _extract_content(self, data: dict[str, Any]) -> str:
        message = data.get("message", {})
        return (message.get("content") or "").strip()

    def model_inventory(self, timeout: int = 5) -> dict[str, Any]:
        endpoint = self._host_root().rstrip("/") + "/api/tags"
        try:
            response = requests.get(endpoint, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            models = [m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)]
            data = {
                "ok": True,
                "endpoint": endpoint,
                "configured_model": self._model_name(),
                "models": models,
                "configured_present": self._model_name() in models,
            }
            self._last_tags = data
            return data
        except Exception as exc:
            data = {
                "ok": False,
                "endpoint": endpoint,
                "configured_model": self._model_name(),
                "error": str(exc),
                "models": [],
                "configured_present": False,
            }
            self._last_tags = data
            return data


LocalLLMClient = OllamaClient


def get_llm_client() -> LocalLLMClient:
    return LocalLLMClient()
