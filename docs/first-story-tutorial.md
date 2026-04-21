# Story-Engine First Story Tutorial (3 Short Chapters + Narration)

This guide walks a new user through a complete first run.

You will learn:
- what the pipeline needs before starting
- where to edit prompts and story guidance
- how to run 3 short chapters
- how narration approval works
- what to do when something fails

## 1. What You Need Before Starting

1. macOS terminal access
2. Python virtual environment in this repo
3. Ollama running on port 11434
4. Hypura running on port 11435 with a GGUF model
5. Chatterbox TTS UI running on port 7865
6. ffmpeg installed
7. a voice sample WAV file in voices/

## 2. Activate Environment

From the repo root:

```bash
cd /Users/wes/Desktop/Story-Engine
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2.1 Create an Offline Backup (Recommended Before Cleanup)

```bash
bash scripts/backup_story_engine_to_m2.sh "/Volumes/256 M.2/Story-Engine-backups"
```

This creates timestamped archives, git bundles, status snapshots, and checksums.

## 3. Preflight Check (Do This First)

Run:

```bash
python scripts/preflight.py
```

You want these to be true:
- ffmpeg ok: true
- ollama ok: true
- hypura ok: true
- local_disk_kv_model_probe ok: true
- chatterbox_webui ok: true

If one is false, fix that service first (see Troubleshooting section).

## 4. Start Services (If Not Already Running)

Use separate terminals.

### 4.1 Ollama

```bash
ollama serve
```

### 4.2 Hypura

```bash
bash scripts/start_hypura.sh
```

If needed, pass a specific GGUF path:

```bash
bash scripts/start_hypura.sh "/Volumes/256 M.2/story-engine-models/your-model.gguf"
```

### 4.3 Chatterbox TTS UI

```bash
bash scripts/start_chatterbox_tts_ui.sh
```

UI should open at http://127.0.0.1:7865

## 5. Set Up a First 3-Chapter Story

You can use existing files or edit them for your own story.

### 5.1 Core Story Files to Edit

Use the dedicated prompt/input guide here:
- docs/PROMPT_REFERENCE.md

1. story_bible.json
- high-level premise and world facts
- update title, logline, themes, world_facts

2. characters.json
- character behavior and voice
- update name, role, core_wound, flawed_belief, voice_style

3. chapter_briefs.json
- chapter-by-chapter objectives
- for this first run, focus on chapters 1-3

Recommended: make chapter goals explicitly short and focused.

Example chapter goal style:
- "Write a compact chapter (about 500-900 words) with one central conflict and one cliffhanger."

### 5.1.1 Validate JSON Before Running

```bash
python -c "import json; json.load(open('story_bible.json')); print('story_bible.json OK')"
python -c "import json; json.load(open('characters.json')); print('characters.json OK')"
python -c "import json; json.load(open('chapter_briefs.json')); print('chapter_briefs.json OK')"
```

### 5.2 Prompt Tuning Location (Very Important)

Main pipeline prompt style controls are in:
- genre_packs/thriller_scifi.yaml

You can tune:
- writer_prefix: writing style instructions
- editor_prefix: revision style instructions
- content_rules: hard scene rules

If prose is too long, too vague, or too repetitive, tighten these first.

### 5.3 Dashboard for Sections 5.2 to 9

Launch the Story Studio tabbed dashboard:

```bash
python ui/gradio_dashboard.py
```

Open:
- http://127.0.0.1:7800

What it shows:
- active runner state (running or idle)
- current chapter and phase
- chapter progress and ETA text
- current artifact status
- review packet path
- run log tail

Tabs include:
- Projects
- Inputs
- Convert
- Voice
- Downloads
- Run Dashboard

Use this to monitor the active/current work during generation, review, and narration gates.

### 5.4 Validate Phase 1-4 Prompt Outputs (Recommended)

If you generated source docs with `docs/Update Story Pipeline.txt`, validate the output docs before conversion:

```bash
python scripts/validate_phase_docs.py --source-dir "The Last Signal" --require-phase4
```

Then import those docs from the Inputs tab (or via `Import The Last Signal Sources`) and run conversion.

CLI alternative (no UI):

```bash
python scripts/ingest_prompt_outputs.py --source-dir "The Last Signal" --out-dir . --require-phase4
```

## 6. Voice Sample Setup

Place your narrator sample in voices/ (WAV recommended), for example:
- voices/p233_023.wav

In .env set:

```bash
VOICE_SAMPLE=voices/p233_023.wav
```

If VOICE_SAMPLE is empty, pipeline falls back to voices/p233_023.wav.

## 7. Run the First 3 Chapters

Run from repo root:

```bash
PROJECT_NAME=first_story CHAPTER_START=1 CHAPTER_LAST=3 python main.py run --project first_story --chapter-limit 3
```

What to expect:
1. Chapter text is generated
2. Pipeline pauses before narration review for each chapter
3. You tune voice in Chatterbox UI
4. You approve and resume
5. Dashboard updates phase/progress while the run is active

## 8. Narration Approval Loop (Per Chapter)

By default, the pipeline pauses at a review gate before narration.

For chapter N:
1. Open the generated review file in reviews/
   - reviews/chNN_pre_narration_review.md
2. In Chatterbox UI, test voice settings with that chapter text
3. Approve chapter narration:

```bash
touch reviews/chNN_pre_narration.approved
```

Example for chapter 1:

```bash
touch reviews/ch01_pre_narration.approved
```

4. Resume pipeline:

```bash
PROJECT_NAME=first_story python main.py resume --project first_story
```

Repeat for chapter 2 and 3.

If status appears stale in the dashboard, click `Refresh Status` and `Refresh Services`.

## 9. Output Files You Should See

For each chapter NN:
- chapters/chNN_draft.txt
- chapters/chNN_edited.txt
- chapters/chNN_final.txt
- chapters/chNN_tts.txt
- summaries/chNN_summary.txt
- audio/chNN_narration.wav
- reviews/chNN_lint.json
- reviews/chNN_lint.md
- reviews/chNN_inference_report.json

Global outputs:
- chapters/manuscript.md
- .state/checkpoints/first_story.json
- .state/diagnostics/runs/*_startup.json
- .state/diagnostics/memory/*.jsonl

## 10. Helpful Features You Can Use

1. Dry run (no model calls)

```bash
PROJECT_NAME=first_story DRY_RUN=true CHAPTER_START=1 CHAPTER_LAST=3 python main.py run --project first_story --chapter-limit 3 --dry-run
```

2. Resume interrupted run

```bash
PROJECT_NAME=first_story python main.py resume --project first_story
```

3. Narrate one chapter manually

```bash
python main.py narrate --project first_story --chapter 2
```

4. Export manuscript

```bash
python main.py export --project first_story --format md
```

5. Regenerate only specific narration sentences

```bash
python scripts/patch_narration.py --chapter 2 --sentence 14
```

6. Reset one chapter outputs and rerun

```bash
python scripts/reset_chapter.py --chapter 2
```

## 11. Troubleshooting

### Error: hypura connection refused (127.0.0.1:11435)

Cause:
- Hypura not running, wrong port, or wrong model path.

Fix:
1. start it with scripts/start_hypura.sh
2. verify .env HYPURA_PORT matches runtime
3. rerun python scripts/preflight.py

### Error: chatterbox connection refused (127.0.0.1:7865)

Cause:
- Chatterbox UI not running.

Fix:
1. run bash scripts/start_chatterbox_tts_ui.sh
2. open http://127.0.0.1:7865
3. rerun preflight

### Error: ffmpeg missing

Cause:
- ffmpeg not installed or not in PATH.

Fix:
1. install ffmpeg
2. verify: ffmpeg -version
3. rerun preflight

### Error: pipeline pauses with memory threshold message

Cause:
- memory guardrails triggered (warn/throttle/pause/emergency).

Fix:
1. close heavy apps and free RAM
2. rerun resume command
3. if needed, tune memory limits in .env:
   - MEMORY_WARN_RSS_MB
   - MEMORY_THROTTLE_RSS_MB
   - MEMORY_PAUSE_RSS_MB
   - MEMORY_EMERGENCY_RSS_MB

### Error: no narration generated

Cause:
- missing voice sample, missing pre-narration approval, or TTS service down.

Fix:
1. ensure VOICE_SAMPLE path exists (typo check)
2. ensure reviews/chNN_pre_narration.approved exists
3. ensure chatterbox is up
4. rerun resume

### Error: chapter feels off-style or too long

Fix order:
1. tighten chapter_briefs.json chapter goal and key events
2. tighten genre_packs/thriller_scifi.yaml writer_prefix and content_rules
3. rerun only that chapter after reset_chapter.py

## 12. Recommended First-Run Workflow Summary

1. Activate venv
2. Run preflight until core checks are true
3. Edit story_bible.json, characters.json, chapter_briefs.json
4. Tune genre_packs/thriller_scifi.yaml prompts
5. Start 3-chapter run
6. Approve narration chapter by chapter
7. Resume after each approval
8. Verify manuscript + 3 narration WAV files

Once this succeeds, scale to more chapters by increasing CHAPTER_LAST.
