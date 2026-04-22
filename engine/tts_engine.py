import json
import os
import re
import signal
import shutil
import subprocess
import time
import hashlib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from config import SETTINGS
from engine.chatterbox_http import call_endpoint, discover_chatterbox, resolve_generate_endpoint, upload_file

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_ABBREV_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx)\.", re.IGNORECASE)
_PARALINGUISTIC_TAG_RE = re.compile(r"<[^>]+>")


def _format_fields(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            rendered = f"{value:.3f}"
        else:
            rendered = str(value).replace("\n", "\\n")
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def _make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(level: str, message: str, **fields: Any) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        suffix = _format_fields(fields)
        line = f"[{timestamp}] [{level}] {message}"
        if suffix:
            line = f"{line} {suffix}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    return emit


def _split_sentences_with_paragraph_breaks(text: str) -> list[tuple[str, bool]]:
    entries: list[tuple[str, bool]] = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    for p_idx, paragraph in enumerate(paragraphs):
        cleaned = " ".join(paragraph.split())
        protected = _ABBREV_RE.sub(lambda m: f"{m.group(1)}<DOT>", cleaned)
        parts = _SENTENCE_END_RE.split(protected)
        sentences = [p.replace("<DOT>", ".").strip() for p in parts]
        sentences = [p for p in sentences if len(p) > 2]

        for s_idx, sentence in enumerate(sentences):
            para_break_after = s_idx == len(sentences) - 1 and p_idx < len(paragraphs) - 1
            entries.append((sentence, para_break_after))

    if entries:
        return entries

    cleaned = " ".join(text.split())
    if cleaned:
        return [(cleaned, False)]
    return []


def split_sentences(text: str) -> list[str]:
    return [sentence for sentence, _ in _split_sentences_with_paragraph_breaks(text)]


def _extract_paralinguistic_tags(sentence: str) -> tuple[str, dict[str, str]]:
    """
    Extract paralinguistic tags from a sentence.
    
    Matches patterns like [emotion: sad], [pace: slow], or [whisper]
    Returns: (cleaned_sentence, tags_dict)
    """
    if not SETTINGS.paralingustic_enabled:
        return sentence, {}
    
    tags: dict[str, str] = {}
    # Pattern: [word] or [word: value] where word can contain lowercase/uppercase/numbers
    tag_pattern = re.compile(r'\[([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*:\s*([^\]]+))?\]')
    
    def extract_tag(match):
        tag_name = match.group(1).lower()
        tag_value = match.group(2).strip() if match.group(2) else "true"
        tags[tag_name] = tag_value
        return ""
    
    # Remove tags from sentence
    cleaned = tag_pattern.sub(extract_tag, sentence).strip()
    
    return cleaned, tags


def _generate_sentence(
    backend: dict[str, Any],
    api_name: str,
    sentence: str,
    voice_payload: dict[str, Any],
    log,
) -> str:
    # Extract paralinguistic tags from sentence
    cleaned_sentence, para_tags = _extract_paralinguistic_tags(sentence)
    
    if para_tags and log:
        log("DEBUG", "Extracted paralinguistic tags", tags=",".join(f"{k}={v}" for k, v in para_tags.items()))
    
    # Use cleaned sentence (without tags) for TTS
    tts_sentence = cleaned_sentence if SETTINGS.paralingustic_strip_from_narration else sentence
    
    payload = [
        None,
        tts_sentence,
        voice_payload,
        SETTINGS.exaggeration,
        SETTINGS.temperature,
        0,
        SETTINGS.cfg_weight,
        0.05,
        1.0,
        1.2,
    ]

    result = call_endpoint(
        backend["root_url"],
        backend["api_prefix"],
        api_name,
        payload,
        request_timeout=max(30, SETTINGS.tts_sentence_timeout_seconds),
        stream_timeout=max(60, SETTINGS.tts_sentence_timeout_seconds),
        log=log,
    )

    events = result.get("events", [])
    for event in events:
        if event.get("event") == "complete":
            data = event.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and isinstance(first.get("path"), str):
                    return str(first["path"])
                if isinstance(first, str):
                    return first
            raise RuntimeError(f"Unexpected complete payload from {api_name}: {data}")
        if event.get("event") == "error":
            raise RuntimeError(
                f"Chatterbox generation failed via {api_name}; raw_events={json.dumps(events, ensure_ascii=True)}"
            )

    raise RuntimeError(
        f"Chatterbox generation produced no completion event via {api_name}; raw_events={json.dumps(events, ensure_ascii=True)}"
    )


def _segment_manifest_path(segments_dir: Path) -> Path:
    return segments_dir / "manifest.json"


def _segment_cache_stats(segments_dir: Path) -> tuple[int, float]:
    total_files = 0
    total_bytes = 0
    for seg in segments_dir.glob("seg_*.wav"):
        total_files += 1
        try:
            total_bytes += seg.stat().st_size
        except OSError:
            continue
    return total_files, total_bytes / (1024.0 * 1024.0)


def _load_manifest(segments_dir: Path) -> dict[str, Any]:
    path = _segment_manifest_path(segments_dir)
    if not path.exists():
        return {"completed": []}
    return json.loads(path.read_text())


def _save_manifest(segments_dir: Path, manifest: dict[str, Any]) -> None:
    _segment_manifest_path(segments_dir).write_text(json.dumps(manifest, indent=2))


def _source_fingerprint(sentences: list[tuple[str, bool]]) -> str:
    canonical = "\n".join(f"{s.strip()}|pb={1 if para_break else 0}" for s, para_break in sentences)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reset_segment_cache(segments_dir: Path) -> None:
    for seg in segments_dir.glob("seg_*.wav"):
        seg.unlink(missing_ok=True)


def _segment_pad_seconds(sentence: str, paragraph_break_after: bool = False) -> float:
    s = sentence.strip()
    if s.endswith(("?", "!", ".")):
        base = max(SETTINGS.min_pause_end, SETTINGS.silence_pad * SETTINGS.pause_multiplier_end)
    elif s.endswith((":", ";", ",")):
        base = max(SETTINGS.min_pause_mid, SETTINGS.silence_pad * SETTINGS.pause_multiplier_mid)
    else:
        base = max(SETTINGS.min_pause_mid * 0.85, SETTINGS.silence_pad * (SETTINGS.pause_multiplier_mid * 0.9))

    if paragraph_break_after:
        base += max(0.0, SETTINGS.pause_paragraph_bonus)

    return base


def _with_timeout(timeout_seconds: int, label: str, fn):
    if timeout_seconds <= 0 or not hasattr(signal, "setitimer"):
        return fn()

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"{label} exceeded {timeout_seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def narrate_chapter(
    text: str,
    voice_sample: str,
    output_path: str,
    chapter_num: int,
    resume: bool = True,
) -> str:
    started = time.time()
    segments_dir = ROOT / "audio" / "segments" / f"ch{chapter_num:02d}"
    segments_dir.mkdir(parents=True, exist_ok=True)
    log = _make_logger(segments_dir / "tts_debug.log")

    backend = discover_chatterbox(SETTINGS.chatterbox_url)
    api_name = resolve_generate_endpoint(backend, SETTINGS.chatterbox_api)
    log(
        "INFO",
        "Resolved Chatterbox backend",
        root_url=backend["root_url"],
        api_prefix=backend["api_prefix"],
        api_name=api_name,
        endpoints=",".join(backend.get("endpoints", [])),
    )

    try:
        load_model_result = call_endpoint(
            backend["root_url"],
            backend["api_prefix"],
            "/load_model",
            [],
            request_timeout=30,
            stream_timeout=max(60, SETTINGS.tts_sentence_timeout_seconds),
            log=log,
        )
        log("INFO", "Model load completed", event_id=load_model_result.get("event_id"))
    except Exception as exc:
        log("WARN", "Model preload failed; continuing with lazy generation", error=exc)

    uploaded_voice_path = upload_file(
        backend["root_url"],
        backend["api_prefix"],
        voice_sample,
        timeout=30,
    )
    voice_payload = {
        "path": uploaded_voice_path,
        "orig_name": Path(voice_sample).name,
        "meta": {"_type": "gradio.FileData"},
    }
    log("INFO", "Uploaded voice sample", local_voice=voice_sample, remote_voice=uploaded_voice_path)

    if _PARALINGUISTIC_TAG_RE.search(text):
        log("WARN", "Paralinguistic or SSML-style tags detected; stripping unsupported tags")
        text = _PARALINGUISTIC_TAG_RE.sub(" ", text)

    sentence_entries = _split_sentences_with_paragraph_breaks(text)
    if not sentence_entries:
        raise ValueError("No narratable sentences found.")

    before_files, before_mb = _segment_cache_stats(segments_dir)
    log("INFO", "Segment cache before generation", files=before_files, size_mb=before_mb)

    log(
        "INFO",
        f"TTS chapter {chapter_num} start",
        sentences=len(sentence_entries),
        voice=voice_sample,
        api=api_name,
        resume=resume,
    )

    source_hash = _source_fingerprint(sentence_entries)

    manifest = _load_manifest(segments_dir) if resume else {"completed": []}
    manifest_hash = str(manifest.get("source_hash", ""))
    manifest_count = int(manifest.get("sentence_count", 0) or 0)
    manifest_stale = manifest_hash != source_hash or manifest_count != len(sentence_entries)
    if resume and not manifest_hash:
        manifest_stale = True
        log("WARN", f"Missing source hash in chapter {chapter_num} manifest; forcing regeneration")

    if resume and manifest_stale:
        log(
            "INFO",
            f"Narration source changed for chapter {chapter_num}; invalidating cached segments",
        )
        _reset_segment_cache(segments_dir)
        manifest = {"completed": []}

    manifest["manifest_version"] = 2
    manifest["source_hash"] = source_hash
    manifest["sentence_count"] = len(sentence_entries)
    manifest["voice_sample"] = str(voice_sample)
    manifest["uploaded_voice_sample"] = uploaded_voice_path
    manifest["api_name"] = api_name
    manifest["backend"] = {
        "root_url": backend["root_url"],
        "api_prefix": backend["api_prefix"],
        "endpoints": backend.get("endpoints", []),
    }
    manifest["pacing_profile"] = {
        "intro_lead_in_seconds": SETTINGS.intro_lead_in_seconds,
        "pause_multiplier_end": SETTINGS.pause_multiplier_end,
        "pause_multiplier_mid": SETTINGS.pause_multiplier_mid,
        "pause_paragraph_bonus": SETTINGS.pause_paragraph_bonus,
    }
    _save_manifest(segments_dir, manifest)

    completed = set(manifest.get("completed", []))
    failed = set(manifest.get("failed", []))

    segment_files: list[Path] = []
    segment_pads: list[float] = []
    for i, (sentence, paragraph_break_after) in enumerate(sentence_entries):
        seg_name = f"seg_{i:04d}.wav"
        seg_path = segments_dir / seg_name
        pad_seconds = _segment_pad_seconds(sentence, paragraph_break_after)

        if resume and seg_name in completed and seg_path.exists():
            segment_files.append(seg_path)
            segment_pads.append(pad_seconds)
            continue

        last_exc: Exception | None = None
        for attempt in range(1, SETTINGS.max_retries + 1):
            try:
                log(
                    "DEBUG",
                    f"TTS sentence attempt",
                    chapter=chapter_num,
                    sentence_index=i + 1,
                    sentence_total=len(sentence_entries),
                    attempt=attempt,
                    attempt_total=SETTINGS.max_retries,
                    chars=len(sentence),
                    preview=sentence[:96],
                )
                generated = _with_timeout(
                    SETTINGS.tts_sentence_timeout_seconds,
                    f"chapter {chapter_num} sentence {i}",
                    lambda: _generate_sentence(backend, api_name, sentence, voice_payload, log),
                )
                shutil.copy(generated, seg_path)
                completed.add(seg_name)
                failed.discard(seg_name)
                manifest["completed"] = sorted(completed)
                manifest["failed"] = sorted(failed)
                manifest["last_error"] = ""
                manifest["last_generated"] = str(generated)
                _save_manifest(segments_dir, manifest)
                segment_files.append(seg_path)
                segment_pads.append(pad_seconds)
                log(
                    "INFO",
                    "Sentence generated",
                    chapter=chapter_num,
                    sentence_index=i + 1,
                    output_segment=seg_path,
                    source_audio=generated,
                    pad_seconds=pad_seconds,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                sleep_s = SETTINGS.retry_backoff * attempt
                log(
                    "WARN",
                    "Sentence generation failed",
                    chapter=chapter_num,
                    sentence_index=i + 1,
                    attempt=attempt,
                    sleep_seconds=sleep_s,
                    error=exc,
                )
                time.sleep(sleep_s)

        if last_exc is not None:
            log(
                "WARN",
                "Sentence failed after retries",
                chapter=chapter_num,
                sentence_index=i + 1,
                error=last_exc,
            )
            failed.add(seg_name)
            manifest["failed"] = sorted(failed)
            manifest["last_error"] = f"sentence {i}: {last_exc}"
            _save_manifest(segments_dir, manifest)
            continue

        time.sleep(SETTINGS.request_delay)

    if not segment_files:
        raise RuntimeError("No audio segments produced.")

    stitch_audio(
        segment_files,
        segment_pads,
        Path(output_path),
        lead_in_seconds=max(0.0, SETTINGS.intro_lead_in_seconds),
    )
    log(
        "INFO",
        f"TTS chapter {chapter_num} done",
        segments=len(segment_files),
        output=output_path,
        elapsed=max(0.0, time.time() - started),
    )
    after_files, after_mb = _segment_cache_stats(segments_dir)
    log("INFO", "Segment cache after generation", files=after_files, size_mb=after_mb)
    return output_path


def stitch_audio(
    segment_files: list[Path],
    segment_pads: list[float],
    output_path: Path,
    lead_in_seconds: float = 0.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_path.parent / ".tmp_stitch"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    concat_sources: list[Path] = []
    if lead_in_seconds > 0.0:
        pre_roll = tmp_dir / "pre_roll.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={SETTINGS.sample_rate}:cl=mono",
                "-t",
                f"{lead_in_seconds:.2f}",
                "-ar",
                str(SETTINGS.sample_rate),
                "-ac",
                "1",
                "-codec:a",
                "pcm_s16le",
                str(pre_roll),
            ],
            check=True,
            capture_output=True,
        )
        concat_sources.append(pre_roll)

    padded_files: list[Path] = []
    for idx, seg in enumerate(segment_files):
        padded = tmp_dir / f"pad_{idx:04d}.wav"
        pad = segment_pads[idx] if idx < len(segment_pads) else SETTINGS.silence_pad
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(seg),
                "-af",
                f"aresample={SETTINGS.sample_rate},apad=pad_dur={pad}",
                "-ar",
                str(SETTINGS.sample_rate),
                "-ac",
                "1",
                "-codec:a",
                "pcm_s16le",
                str(padded),
            ],
            check=True,
            capture_output=True,
        )
        padded_files.append(padded)

    concat_sources.extend(padded_files)

    concat_list = tmp_dir / "concat.txt"
    with concat_list.open("w", encoding="utf-8") as f:
        for seg in concat_sources:
            f.write(f"file '{seg.resolve()}'\n")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-ar",
            str(SETTINGS.sample_rate),
            "-ac",
            "1",
            "-codec:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    if abs(SETTINGS.narration_speed - 1.0) > 0.001:
        sped_path = tmp_dir / "sped_output.wav"
        speed = max(0.5, min(2.0, SETTINGS.narration_speed))
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(output_path),
                "-af",
                f"atempo={speed}",
                "-ar",
                str(SETTINGS.sample_rate),
                "-ac",
                "1",
                "-codec:a",
                "pcm_s16le",
                str(sped_path),
            ],
            check=True,
            capture_output=True,
        )
        shutil.move(str(sped_path), str(output_path))

    for p in concat_sources:
        p.unlink(missing_ok=True)
    concat_list.unlink(missing_ok=True)
    tmp_dir.rmdir()


def smoke_test(text: str, chapter_num: int = 0) -> str:
    out = ROOT / "audio" / f"smoke_{chapter_num:02d}.wav"
    return narrate_chapter(
        text=text,
        voice_sample=SETTINGS.voice_sample,
        output_path=str(out),
        chapter_num=chapter_num,
    )


if __name__ == "__main__":
    sample = (
        "This is a smoke test. The quick brown fox jumps over the lazy dog. "
        "If you hear clear sentence endings, the setup is healthy."
    )
    final = smoke_test(sample, chapter_num=0)
    print(f"Wrote: {final}")
