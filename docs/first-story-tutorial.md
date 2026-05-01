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
2. Python virtual environment in this repo (Python 3.9–3.11)
3. RotorQuant/llama.cpp structural server on port 11436
4. RotorQuant/llama.cpp prose server on port 11435
5. Chatterbox TTS UI running on port 7865
6. ffmpeg installed
7. a voice sample WAV file in voices/

> **Note:** Ollama can be used as a fallback for either lane. Set `LLM_BACKEND=ollama` in `.env` and start `ollama serve` on the relevant port. The active backend is shown in the Dashboard → Run Dashboard → Service Status panel.

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
- structural lane listener ok: true (port 11436)
- prose lane listener ok: true (port 11435)
- chatterbox_webui ok: true

If one is false, fix that service first (see Section 4 and Troubleshooting).

## 4. Start Services (If Not Already Running)

The pipeline uses a **dual-lane llama.cpp backend**: one server for structural tasks (chapter planning, scene outlines) and one for prose generation. Both are started via the RotorQuant scripts bundled under `scripts/`.

Use separate terminals (or click **Start Services** in the Dashboard).

### 4.1 Structural Lane (port 11436)

```bash
bash scripts/start_rotorquant_structural.sh
```

The model path is read from `STRUCTURAL_MODEL_PATH` in `.env`. You can override it per-run:

```bash
STRUCTURAL_MODEL_PATH="/path/to/your-structural.gguf" bash scripts/start_rotorquant_structural.sh
```

### 4.2 Prose Lane (port 11435)

```bash
bash scripts/start_rotorquant_prose.sh
```

The model path is read from `PROSE_MODEL_PATH` in `.env`. You can override it:

```bash
PROSE_MODEL_PATH="/path/to/your-prose.gguf" bash scripts/start_rotorquant_prose.sh
```

**Fallback — Ollama:**
If you do not have RotorQuant built, Ollama can serve either lane:
```bash
ollama serve  # serves both lanes via the URL in .env
```
Set `LLM_BACKEND=ollama` in `.env` and update `STRUCTURAL_URL` and `HYPURA_URL` to point at your Ollama instance (default: port 11434).

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

### 5.3 Dashboard for Monitoring and Controls

Launch the Story Studio tabbed dashboard:

```bash
python ui/gradio_dashboard.py
```

Open your browser to:
- http://127.0.0.1:7800

**Note:** If you get a port error, the default fallback port is 7801. Check the terminal output.

What the dashboard shows:
- Active project and pipeline state (running or idle)
- Current chapter and phase during generation
- Chapter progress and ETA
- Current artifact status and review packet path
- Real-time run log tail

Dashboard tabs include:
- **Projects:** Create and manage projects
- **Inputs:** Upload and manage input documents
- **Convert:** Run phase-output conversion
- **Voice:** Upload narrator voice sample
- **Downloads:** Export generated files
- **Run Dashboard:** Start/stop pipeline, set parameters, reset chapters, approve narration, monitor progress

#### Run Dashboard Controls

| Control | Description |
|---|---|
| LLM Model Profile | Active prose model (DavidAU 13.7B, GPT-OSS 20B, Qwen3.5 MLX, Qwen2.5 Q5, Mixtral TurboQuant) |
| Operating Profile | Preset stack config: **Work** (DavidAU + Qdrant), **Play** (RotorQuant Dual Lane), **Recovery** (Qwen2.5 stable) |
| Start/Last Chapter | Chapter range for Sequential and Resume modes |
| Word Target Min/Max | Prose length target per chapter |
| Narration Pace | TTS speed multiplier (0.7–1.3) |
| **Advanced LLM Parameters** *(accordion, collapsed by default)* | |
| Context Window (num_ctx) | Tokens available to the LLM per call (1024–16384, default 4096) |
| Temperature | Sampling temperature (0–2, default 0.8) |
| LLM Timeout (s) | Max seconds per LLM call before retry (default 900) |
| Max Retries | Retry attempts per failing call (1–5, default 2) |
| Block on Lint Fail | If unchecked, pipeline continues even when prose fails quality checks |

#### Operating Profiles vs. Model Profiles

- **Operating Profile** sets the full stack preset (backend type, memory backend, which model profile is active by default).
- **Model Profile** overrides just the LLM model for prose generation.

The two can be mixed — e.g., use the *Recovery* operating profile (stable RotorQuant) but swap in the *DavidAU* model profile for prose quality.

You can run the pipeline from the Dashboard UI *or* from the CLI (`python main.py run ...`). Both use the same backend.

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

## 6.1 Paralinguistic Prompts (Optional Feature)

You can add emotional and pacing markup to your chapter text using paralinguistic tags. These are passed to the TTS system but NOT spoken aloud.

**Syntax:**
- `[emotion: sad]` — specify emotional tone
- `[pace: slow]` or `[pace: fast]` — narration speed hint
- `[whisper]` — delivery style
- `[angry]`, `[confused]`, `[hopeful]` — any emotion or style

**Example:**
```
"I'm sorry," she said. [emotion: sad] [pace: slow] "I didn't mean to."
```

Result: TTS narrates only "I'm sorry," she said. "I didn't mean to." but applies sadness and slow pace.

**Configuration:**
- Enable/disable: `PARALINGUSTIC_ENABLED=true` in .env (default: true)
- Strip from audio: `PARALINGUSTIC_STRIP_FROM_NARRATION=true` (default: true) — if false, tags are included in spoken text

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

## 11. Reset a Chapter or Entire Project

You can reset chapter outputs at any time to regenerate with new prompts or parameters.

### Reset One Chapter (CLI)

To delete outputs for chapter 2 and rerun generation:

```bash
python scripts/reset_chapter.py --chapter 2
```

This removes:
- ch02_draft.txt, ch02_edited.txt, ch02_final.txt, ch02_tts.txt
- ch02_narration.wav and all sentence segments
- ch02_summary.txt and all review files

**Preview what will be deleted (dry-run):**
```bash
python scripts/reset_chapter.py --chapter 2 --dry-run
```

### Reset Current Chapter (Dashboard UI)

In the Dashboard "Run Dashboard" tab:
1. Click "Reset Run"
2. Select scope: "Current Chapter"
3. Enter chapter number (e.g., 2)
4. Optional: enable "Force Stop" if pipeline is running
5. Click "Reset"

### Reset All Chapters (Full Project Reset)

In the Dashboard "Run Dashboard" tab:
1. Click "Reset Run"
2. Select scope: "All Chapters"
3. Optional: enable "Force Stop" and "Root Pipeline Files" (to also clear story_bible.json, characters.json, etc.)
4. Confirm "All Chapters reset" checkbox
5. Click "Reset"

Or from CLI:
```bash
for ch in {1..10}; do python scripts/reset_chapter.py --chapter $ch; done
```

After reset, you can immediately start a new run from the beginning.

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
