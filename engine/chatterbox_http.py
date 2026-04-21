import json
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import requests


def _root_url(target: str) -> str:
    parts = urlsplit(target)
    if not parts.scheme or not parts.netloc:
        return target.rstrip("/")
    return f"{parts.scheme}://{parts.netloc}"


def _normalize_endpoint_name(name: str) -> str:
    return name if name.startswith("/") else f"/{name}"


def _call_path(name: str) -> str:
    normalized = _normalize_endpoint_name(name)
    if normalized.startswith("//"):
        return normalized[1:]
    return normalized.lstrip("/")


def discover_chatterbox(base_url: str, timeout: int = 5) -> dict[str, Any]:
    root = _root_url(base_url)
    config_resp = requests.get(f"{root}/config", timeout=timeout)
    config_resp.raise_for_status()
    config = config_resp.json()

    api_prefix = str(config.get("api_prefix") or "/gradio_api").rstrip("/")
    info_resp = requests.get(f"{root}{api_prefix}/info", timeout=timeout)
    info_resp.raise_for_status()
    info = info_resp.json()

    named_endpoints = info.get("named_endpoints", {}) or {}
    endpoints = sorted(named_endpoints.keys())

    return {
        "root_url": root,
        "api_prefix": api_prefix,
        "config": config,
        "info": info,
        "named_endpoints": named_endpoints,
        "endpoints": endpoints,
    }


def resolve_generate_endpoint(discovery: dict[str, Any], preferred: str = "") -> str:
    endpoints = discovery.get("endpoints", []) or []
    if preferred:
        preferred_name = _normalize_endpoint_name(preferred)
        if preferred_name in endpoints:
            return preferred_name

    for candidate in ("/generate", "/predict", "/infer"):
        if candidate in endpoints:
            return candidate

    if endpoints:
        return endpoints[0]

    raise RuntimeError("No public Gradio endpoints were exposed by the Chatterbox server.")


def upload_file(
    root_url: str,
    api_prefix: str,
    file_path: str,
    timeout: int = 30,
) -> str:
    with open(file_path, "rb") as handle:
        response = requests.post(
            f"{root_url}{api_prefix}/upload",
            files=[("files", (file_path.split("/")[-1], handle))],
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Unexpected upload response: {payload}")
    return str(payload[0])


def parse_sse_lines(lines: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current_event = "message"
    current_data: list[str] = []

    def flush_event() -> None:
        nonlocal current_event, current_data
        if not current_data:
            return
        raw = "\n".join(current_data)
        parsed: Any = raw
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = raw
        events.append({"event": current_event, "data": parsed, "raw": raw})
        current_event = "message"
        current_data = []

    for line in lines:
        if not line:
            flush_event()
            continue
        if line.startswith("event:"):
            flush_event()
            current_event = line.split(":", 1)[1].strip() or "message"
            continue
        if line.startswith("data:"):
            current_data.append(line.split(":", 1)[1].strip())

    flush_event()
    return events


def call_endpoint(
    root_url: str,
    api_prefix: str,
    endpoint_name: str,
    data: list[Any],
    request_timeout: int = 30,
    stream_timeout: int = 180,
    log: Optional[Callable[[str, str], None]] = None,
) -> dict[str, Any]:
    endpoint_path = _call_path(endpoint_name)
    call_url = f"{root_url}{api_prefix}/call/{endpoint_path}"
    init_resp = requests.post(call_url, json={"data": data}, timeout=request_timeout)
    init_resp.raise_for_status()
    payload = init_resp.json()
    event_id = payload.get("event_id")
    if not event_id:
        raise RuntimeError(f"Missing event_id from {call_url}: {payload}")

    stream_url = f"{call_url}/{event_id}"
    if log:
        log("DEBUG", f"SSE open {stream_url}")

    raw_lines: list[str] = []
    with requests.get(stream_url, stream=True, timeout=stream_timeout) as stream_resp:
        stream_resp.raise_for_status()
        for raw_line in stream_resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = str(raw_line)
            raw_lines.append(line)
            if log and line:
                log("DEBUG", f"SSE {endpoint_name} {line}")

    events = parse_sse_lines(raw_lines)
    return {
        "event_id": event_id,
        "events": events,
        "raw_lines": raw_lines,
        "call_url": call_url,
        "stream_url": stream_url,
    }
