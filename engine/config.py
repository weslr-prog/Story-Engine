from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Load .env file if it exists
load_dotenv(ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    projects_root: Path
    checkpoint_dir: Path
    chroma_db_path: Path
    story_db_filename: str
    genre_pack_dir: Path
    reviews_dir: str
    chapter_count: int
    ollama_url: str
    ollama_model: str
    local_disk_kv_url: str
    local_disk_kv_model: str
    embedding_model: str
    hypura_url: str
    hypura_model: str
    hypura_models_dir: Path
    hypura_port: int
    rag_budget_tokens: int
    scene_chunk_tokens: int
    scene_chunk_overlap: int
    default_genre_pack: str
    target_minutes_min: int
    target_minutes_max: int
    default_words_per_minute: int
    llm_call_timeout_seconds: int
    llm_max_retries: int
    llm_retry_delay_seconds: float
    llm_thinking_overhead: int
    allow_model_fallback: bool
    llm_backend: str
    llm_num_ctx: int
    use_local_disk_kv: bool
    pause_before_narration_review: bool
    pause_after_chapter_review: bool
    memory_monitor_enabled: bool
    memory_warn_rss_mb: int
    memory_throttle_rss_mb: int
    memory_pause_rss_mb: int
    memory_emergency_rss_mb: int
    memory_throttle_token_scale: float
    memory_throttle_request_delay_scale: float
    diagnostics_dir: Path
    chatterbox_url: str
    voice_sample: str
    exaggeration: float
    cfg_weight: float
    temperature: float
    silence_pad_ms: int
    chatterbox_api: str
    max_retries: int
    retry_backoff: float
    request_delay: float
    tts_sentence_timeout_seconds: int
    sample_rate: int
    narration_speed: float
    silence_pad: float
    min_pause_end: float
    min_pause_mid: float
    pause_multiplier_end: float
    pause_multiplier_mid: float
    pause_paragraph_bonus: float
    intro_lead_in_seconds: float
    paralingustic_enabled: bool
    paralingustic_strip_from_narration: bool


def load_settings() -> Settings:
    return Settings(
        projects_root=ROOT / _env_str("PROJECTS_ROOT", "projects"),
        checkpoint_dir=ROOT / _env_str("CHECKPOINT_DIR", ".state/checkpoints"),
        chroma_db_path=ROOT / _env_str("CHROMA_DB_PATH", "chroma_db"),
        story_db_filename=_env_str("STORY_DB_FILENAME", "story_bible.sqlite"),
        genre_pack_dir=ROOT / _env_str("GENRE_PACK_DIR", "genre_packs"),
        reviews_dir=_env_str("REVIEWS_DIR", "reviews"),
        chapter_count=_env_int("CHAPTER_COUNT", 12),
        ollama_url=_env_str("OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions"),
        ollama_model=_env_str("OLLAMA_MODEL", "phi3.5:3.8b-mini-instruct-q5_K_M"),
        local_disk_kv_url=_env_str("LOCAL_DISK_KV_URL", _env_str("OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions")),
        local_disk_kv_model=_env_str("LOCAL_DISK_KV_MODEL", _env_str("OLLAMA_MODEL", "phi3.5:3.8b-mini-instruct-q5_K_M")),
        embedding_model=_env_str("EMBEDDING_MODEL", "nomic-embed-text"),
        hypura_url=_env_str("HYPURA_URL", "http://127.0.0.1:11435/api/chat"),
        hypura_model=_env_str("HYPURA_MODEL", "mixtral-8x7b-instruct-turboquant"),
        hypura_models_dir=Path(_env_str("HYPURA_MODELS_DIR", "/Volumes/YOUR_EXTERNAL_DRIVE/story-engine-models")),
        hypura_port=_env_int("HYPURA_PORT", 11435),
        rag_budget_tokens=_env_int("RAG_BUDGET_TOKENS", 1700),
        scene_chunk_tokens=_env_int("SCENE_CHUNK_TOKENS", 500),
        scene_chunk_overlap=_env_int("SCENE_CHUNK_OVERLAP", 75),
        default_genre_pack=_env_str("DEFAULT_GENRE_PACK", "thriller_scifi"),
        target_minutes_min=_env_int("TARGET_MINUTES_MIN", 14),
        target_minutes_max=_env_int("TARGET_MINUTES_MAX", 17),
        default_words_per_minute=_env_int("DEFAULT_WORDS_PER_MINUTE", 150),
        llm_call_timeout_seconds=_env_int("LLM_CALL_TIMEOUT_SECONDS", 300),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
        llm_retry_delay_seconds=_env_float("LLM_RETRY_DELAY_SECONDS", 2.0),
        llm_thinking_overhead=_env_int("LLM_THINKING_OVERHEAD", 0),
        allow_model_fallback=_env_bool("ALLOW_MODEL_FALLBACK", True),
        llm_backend=_env_str("LLM_BACKEND", "ollama"),
        llm_num_ctx=_env_int("LLM_NUM_CTX", 8192),
        use_local_disk_kv=_env_bool("USE_LOCAL_DISK_KV", False),
        pause_before_narration_review=_env_bool("PAUSE_BEFORE_NARRATION_REVIEW", True),
        pause_after_chapter_review=_env_bool("PAUSE_AFTER_CHAPTER_REVIEW", False),
        memory_monitor_enabled=_env_bool("MEMORY_MONITOR_ENABLED", True),
        memory_warn_rss_mb=_env_int("MEMORY_WARN_RSS_MB", 11000),
        memory_throttle_rss_mb=_env_int("MEMORY_THROTTLE_RSS_MB", 12500),
        memory_pause_rss_mb=_env_int("MEMORY_PAUSE_RSS_MB", 13800),
        memory_emergency_rss_mb=_env_int("MEMORY_EMERGENCY_RSS_MB", 14800),
        memory_throttle_token_scale=_env_float("MEMORY_THROTTLE_TOKEN_SCALE", 0.72),
        memory_throttle_request_delay_scale=_env_float("MEMORY_THROTTLE_REQUEST_DELAY_SCALE", 1.5),
        diagnostics_dir=ROOT / _env_str("DIAGNOSTICS_DIR", ".state/diagnostics"),
        chatterbox_url=_env_str("CHATTERBOX_URL", "http://127.0.0.1:7865"),
        voice_sample=_env_str("VOICE_SAMPLE", ""),
        exaggeration=_env_float("EXAGGERATION", 0.4),
        cfg_weight=_env_float("CFG_WEIGHT", 0.65),
        temperature=_env_float("TEMPERATURE", 0.72),
        silence_pad_ms=_env_int("SILENCE_PAD_MS", 250),
        chatterbox_api=_env_str("CHATTERBOX_API", ""),
        max_retries=_env_int("MAX_RETRIES", 2),
        retry_backoff=_env_float("RETRY_BACKOFF", 0.75),
        request_delay=_env_float("REQUEST_DELAY", 0.5),
        tts_sentence_timeout_seconds=_env_int("TTS_SENTENCE_TIMEOUT_SECONDS", 180),
        sample_rate=_env_int("SAMPLE_RATE", 22050),
        narration_speed=_env_float("NARRATION_SPEED", 1.0),
        silence_pad=_env_float("SILENCE_PAD", _env_int("SILENCE_PAD_MS", 250) / 1000.0),
        min_pause_end=_env_float("MIN_PAUSE_END", 0.24),
        min_pause_mid=_env_float("MIN_PAUSE_MID", 0.12),
        pause_multiplier_end=_env_float("PAUSE_MULTIPLIER_END", 1.0),
        pause_multiplier_mid=_env_float("PAUSE_MULTIPLIER_MID", 0.7),
        pause_paragraph_bonus=_env_float("PAUSE_PARAGRAPH_BONUS", 0.15),
        intro_lead_in_seconds=_env_float("INTRO_LEAD_IN_SECONDS", 0.0),
        paralingustic_enabled=_env_bool("PARALINGUSTIC_ENABLED", True),
        paralingustic_strip_from_narration=_env_bool("PARALINGUSTIC_STRIP_FROM_NARRATION", True),
    )


SETTINGS = load_settings()
