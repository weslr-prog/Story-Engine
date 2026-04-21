from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import SETTINGS


@dataclass
class MemorySnapshot:
    timestamp: float
    label: str
    chapter: int
    rss_mb: float
    swap_used_mb: float
    pageouts: int
    free_disk_gb: float


class MemoryAction:
    OK = "ok"
    WARN = "warn"
    THROTTLE = "throttle"
    PAUSE = "pause"
    EMERGENCY = "emergency"


def _rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports ru_maxrss in bytes.
        return usage.ru_maxrss / (1024.0 * 1024.0)


def _swap_and_pageouts() -> tuple[float, int]:
    swap_used_mb = 0.0
    pageouts = 0

    try:
        proc = subprocess.run(["sysctl", "-n", "vm.swapusage"], check=True, capture_output=True, text=True)
        # Example: total = 1024.00M  used = 0.00M  free = 1024.00M
        text = (proc.stdout or "").strip()
        marker = "used ="
        if marker in text:
            tail = text.split(marker, 1)[1].strip()
            raw_value = tail.split()[0]
            if raw_value.endswith("M"):
                swap_used_mb = float(raw_value[:-1])
            elif raw_value.endswith("G"):
                swap_used_mb = float(raw_value[:-1]) * 1024.0
    except Exception:
        swap_used_mb = 0.0

    try:
        proc = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True)
        for line in (proc.stdout or "").splitlines():
            if line.startswith("Pageouts:"):
                value = line.split(":", 1)[1].strip().rstrip(".")
                pageouts = int(value)
                break
    except Exception:
        pageouts = 0

    return swap_used_mb, pageouts


def _free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024.0**3)


class MemoryMonitor:
    def __init__(self, root: Path | None = None) -> None:
        base = root or SETTINGS.diagnostics_dir
        self._root = Path(base)
        self._memory_dir = self._root / "memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self, chapter: int, label: str) -> MemorySnapshot:
        swap_used_mb, pageouts = _swap_and_pageouts()
        return MemorySnapshot(
            timestamp=time.time(),
            label=label,
            chapter=chapter,
            rss_mb=round(_rss_mb(), 2),
            swap_used_mb=round(swap_used_mb, 2),
            pageouts=pageouts,
            free_disk_gb=round(_free_disk_gb(self._root), 2),
        )

    def classify(self, snapshot: MemorySnapshot) -> str:
        if not SETTINGS.memory_monitor_enabled:
            return MemoryAction.OK
        if snapshot.rss_mb >= SETTINGS.memory_emergency_rss_mb:
            return MemoryAction.EMERGENCY
        if snapshot.rss_mb >= SETTINGS.memory_pause_rss_mb:
            return MemoryAction.PAUSE
        if snapshot.rss_mb >= SETTINGS.memory_throttle_rss_mb:
            return MemoryAction.THROTTLE
        if snapshot.rss_mb >= SETTINGS.memory_warn_rss_mb:
            return MemoryAction.WARN
        return MemoryAction.OK

    def write(self, run_id: str, snapshot: MemorySnapshot, action: str) -> None:
        payload = asdict(snapshot)
        payload["action"] = action
        payload["run_id"] = run_id

        out = self._memory_dir / f"ch{snapshot.chapter:02d}.jsonl"
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
