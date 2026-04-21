import argparse
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv

from config import SETTINGS
from tts_engine import (
    ROOT,
    _generate_sentence,
    _load_client,
    _segment_manifest_path,
    resolve_api_name,
    split_sentences,
    stitch_audio,
)

load_dotenv(ROOT / ".env")


def _tts_path(chapter: int) -> Path:
    return ROOT / "chapters" / f"ch{chapter:02d}_tts.txt"


def _segment_dir(chapter: int) -> Path:
    return ROOT / "audio" / "segments" / f"ch{chapter:02d}"


def _segment_path(chapter: int, idx: int) -> Path:
    return _segment_dir(chapter) / f"seg_{idx:04d}.wav"


def _output_path(chapter: int) -> Path:
    return ROOT / "audio" / f"ch{chapter:02d}_narration.wav"


def _load_sentences(chapter: int) -> list[str]:
    tts_file = _tts_path(chapter)
    if not tts_file.exists():
        raise FileNotFoundError(f"Missing TTS text file: {tts_file}")
    text = tts_file.read_text(encoding="utf-8")
    sentences = split_sentences(text)
    if not sentences:
        raise RuntimeError("No narratable sentences found in chapter TTS text.")
    return sentences


def _save_sentences(chapter: int, sentences: list[str]) -> None:
    out_text = " ".join(s.strip() for s in sentences if s.strip())
    _tts_path(chapter).write_text(out_text + "\n", encoding="utf-8")


def _restitch(chapter: int, sentence_count: int) -> None:
    seg_files = []
    for i in range(sentence_count):
        seg = _segment_path(chapter, i)
        if not seg.exists():
            raise FileNotFoundError(
                f"Missing segment {seg}. Run full chapter narration first or regenerate missing indices."
            )
        seg_files.append(seg)

    stitch_audio(seg_files, _output_path(chapter))



def _update_manifest(chapter: int, sentence_count: int) -> None:
    seg_dir = _segment_dir(chapter)
    seg_dir.mkdir(parents=True, exist_ok=True)
    completed = [f"seg_{i:04d}.wav" for i in range(sentence_count) if _segment_path(chapter, i).exists()]
    manifest = {"completed": completed}
    _segment_manifest_path(seg_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _regen_indices(chapter: int, indices: list[int], sentences: list[str]) -> None:
    client = _load_client()
    api_name = resolve_api_name(client)

    seg_dir = _segment_dir(chapter)
    seg_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        if idx < 0 or idx >= len(sentences):
            raise IndexError(f"Sentence index out of range: {idx}")
        generated = _generate_sentence(client, api_name, sentences[idx], SETTINGS.voice_sample)
        shutil.copy(generated, _segment_path(chapter, idx))
        print(f"[OK] Regenerated sentence {idx}")



def _parse_indices(args: argparse.Namespace) -> list[int]:
    if args.sentence is not None:
        return [args.sentence]

    if args.range:
        if ":" not in args.range:
            raise ValueError("--range must be START:END (inclusive)")
        start_s, end_s = args.range.split(":", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise ValueError("range end must be >= start")
        return list(range(start, end + 1))

    raise ValueError("Provide --sentence N or --range START:END")



def main() -> None:
    parser = argparse.ArgumentParser(description="Patch chapter narration by regenerating selected sentence segments.")
    parser.add_argument("--chapter", type=int, required=True, help="Chapter number (e.g., 1)")
    parser.add_argument("--sentence", type=int, help="Single sentence index to regenerate")
    parser.add_argument("--range", type=str, help="Sentence range START:END (inclusive)")
    parser.add_argument("--text", type=str, help="Replacement text for --sentence index")
    parser.add_argument("--write-tts", action="store_true", help="Persist replacement text into chapter tts file")
    parser.add_argument("--list", action="store_true", help="Print indexed sentence list and exit")
    args = parser.parse_args()

    sentences = _load_sentences(args.chapter)

    if args.list:
        for i, sentence in enumerate(sentences):
            print(f"{i:04d}: {sentence}")
        return

    indices = _parse_indices(args)

    if args.text is not None:
        if len(indices) != 1:
            raise ValueError("--text can only be used with --sentence")
        idx = indices[0]
        if idx < 0 or idx >= len(sentences):
            raise IndexError(f"Sentence index out of range: {idx}")
        sentences[idx] = args.text.strip()
        if args.write_tts:
            _save_sentences(args.chapter, sentences)
            print(f"[OK] Updated chapter TTS text at sentence {idx}")

    _regen_indices(args.chapter, indices, sentences)
    _update_manifest(args.chapter, len(sentences))
    _restitch(args.chapter, len(sentences))
    print(f"[OK] Restitched chapter audio: {_output_path(args.chapter)}")


if __name__ == "__main__":
    main()
