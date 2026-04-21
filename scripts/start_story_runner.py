#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".state"
STATE_FILE = STATE_DIR / "story_runner_services.json"
PIPELINE_RUNNER_FILE = STATE_DIR / "pipeline_runner.json"

OLLAMA_URL = "http://127.0.0.1:11434"
CHATTERBOX_URL = "http://127.0.0.1:7865"
STUDIO_HOST = os.getenv("STUDIO_HOST", "127.0.0.1")
STUDIO_PORT = int(os.getenv("STUDIO_PORT", "7861"))
STUDIO_PORT_MAX = int(os.getenv("STUDIO_PORT_MAX", "7871"))


@dataclass
class ManagedProc:
    name: str
    pid: int
    cmd: list[str]


def _http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _spawn(name: str, cmd: list[str], cwd: Path) -> ManagedProc:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return ManagedProc(name=name, pid=proc.pid, cmd=cmd)


def _studio_urls() -> list[str]:
    stop_port = max(STUDIO_PORT, STUDIO_PORT_MAX)
    return [f"http://{STUDIO_HOST}:{port}" for port in range(STUDIO_PORT, stop_port + 1)]


def _discover_studio_url(timeout: float = 0.75) -> str | None:
    for url in _studio_urls():
        if _http_ok(url, timeout=timeout):
            return url
    return None


def _read_state() -> dict:
    if not STATE_FILE.exists():
        return {"services": []}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"services": []}


def _write_state(data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _start_services() -> list[ManagedProc]:
    started: list[ManagedProc] = []

    if not _http_ok(OLLAMA_URL):
        started.append(_spawn("ollama", ["ollama", "serve"], ROOT))

    if not _http_ok(CHATTERBOX_URL):
        started.append(_spawn("chatterbox", ["bash", "scripts/start_chatterbox_tts_ui.sh"], ROOT))

    if (ROOT / "app.py").exists() and not _discover_studio_url():
        started.append(_spawn("story_studio", [sys.executable, "app.py"], ROOT))

    if started:
        existing = _read_state()
        services = existing.get("services", [])
        for proc in started:
            services.append({"name": proc.name, "pid": proc.pid, "cmd": proc.cmd})
        _write_state({"services": services, "updated_at": int(time.time())})

    return started


def _pipeline_line() -> str:
    if not PIPELINE_RUNNER_FILE.exists():
        return "Pipeline: idle"
    try:
        state = json.loads(PIPELINE_RUNNER_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "Pipeline: state unreadable"

    pid = state.get("pid")
    chapter_limit = state.get("chapter_limit")
    mode = state.get("mode") or "n/a"
    if pid and _pid_alive(int(pid)):
        return f"Pipeline: running (pid={pid}, mode={mode}, chapters<= {chapter_limit})"
    return "Pipeline: idle"


def _clear_screen() -> None:
    print("\033[2J\033[H", end="")


def _print_dashboard(studio_url: str | None) -> None:
    ollama = "UP" if _http_ok(OLLAMA_URL) else "DOWN"
    chatterbox = "UP" if _http_ok(CHATTERBOX_URL) else "DOWN"
    active_studio_url = studio_url or f"http://{STUDIO_HOST}:{STUDIO_PORT}"
    studio = "UP" if _http_ok(active_studio_url) else "DOWN"

    _clear_screen()
    print("Story Runner Dashboard")
    print("======================")
    print(f"Ollama     : {ollama} ({OLLAMA_URL})")
    print(f"Chatterbox : {chatterbox} ({CHATTERBOX_URL})")
    print(f"Studio UI  : {studio} ({active_studio_url})")
    print(_pipeline_line())
    print()
    print("Use Story Studio > Run Dashboard for phase details, review approvals, and narration edits.")
    print("Press Ctrl+C to stop refreshing this dashboard.")


def _wait_until_up(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _http_ok(url):
            return True
        time.sleep(0.5)
    return False


def _wait_until_studio_up(timeout_s: float) -> str | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        studio_url = _discover_studio_url(timeout=0.5)
        if studio_url:
            return studio_url
        time.sleep(0.5)
    return None


def _stop_managed_services() -> None:
    state = _read_state()
    services = state.get("services", [])
    if not services:
        print("No tracked services found in .state/story_runner_services.json")
        return

    for service in services:
        pid = int(service.get("pid", 0))
        if pid <= 0:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue

    time.sleep(0.5)
    survivors = []
    for service in services:
        pid = int(service.get("pid", 0))
        if pid > 0 and _pid_alive(pid):
            survivors.append(service)

    _write_state({"services": survivors, "updated_at": int(time.time())})
    print(f"Stop signal sent. Still running: {len(survivors)} service(s).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Story Studio + dependencies with a simple live dashboard.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser tabs automatically.")
    parser.add_argument("--watch-seconds", type=int, default=0, help="Dashboard refresh duration (0 = until Ctrl+C).")
    parser.add_argument("--stop", action="store_true", help="Stop services started by this runner.")
    args = parser.parse_args()

    if args.stop:
        _stop_managed_services()
        return 0

    started = _start_services()
    if started:
        started_names = ", ".join(s.name for s in started)
        print(f"Started: {started_names}")
    else:
        print("All services already appear to be running.")

    studio_url = _wait_until_studio_up(timeout_s=25)
    _wait_until_up(CHATTERBOX_URL, timeout_s=20)

    if not studio_url:
        studio_url = _discover_studio_url() or f"http://{STUDIO_HOST}:{STUDIO_PORT}"

    if not args.no_browser:
        webbrowser.open(studio_url)

    end_time = None if args.watch_seconds <= 0 else (time.time() + args.watch_seconds)
    try:
        while True:
            _print_dashboard(_discover_studio_url() or studio_url)
            if end_time is not None and time.time() >= end_time:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopped dashboard refresh.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
