# Story-Engine Build Tracker

Status legend: `⬜ Not started` | `🔵 In progress` | `✅ Complete` | `❌ Blocked`

## Quick Config Checklist
- [x] External M.2 mount path confirmed: `/Volumes/256 M.2`
- [x] `HYPURA_MODELS_DIR` set in `.env` to `/Volumes/256 M.2/story-engine-models`
- [x] Hypura built and serving on port `11435`
- [x] Ollama running with Qwen2.5:7b-instruct available
- [x] Large prose model artifact available on external SSD (TurboQuant tooling validated)
- [x] ChromaDB path writable and initialized
- [ ] Python 3.11 virtual environment active (currently 3.9.6, fallback acceptable)

## Phase Status
| Phase | Title | Status | Notes |
|---|---|---|---|
| 0 | Project scaffold | ✅ Complete | New repo tree, copied utilities, workspace, docs |
| 1 | System prerequisites | ✅ Complete | FFmpeg ✅, Ollama ✅, venv ✅, M.2 ready ✅, ChromaDB ✅ |
| 2 | Hypura build | ✅ Complete | Built from source in `third_party/hypura`, API validated on `11435` |
| 3 | TurboQuant pipeline | 🔵 In progress | `scripts/phase3_turboquant.py --smoke` confirms tools and GGUF discovery; quant plan wiring added |
| 4 | Ollama structural model | ✅ Complete | Qwen2.5:7b-instruct available and tested |
| 5 | Story bible database | ✅ Complete | ORM initialized and validated via `scripts/phase56_smoke.py` |
| 6 | RAG memory system | ✅ Complete | Chroma round-trip validated; switched to deterministic local embeddings for offline stability |
| 7 | Inference abstraction | ✅ Complete | Structural (Ollama) + prose (Hypura) lanes pass ping/chat smoke |
| 8 | Orchestrator FSM | ✅ Complete | `pipeline_novel.py` runtime loop and checkpoints implemented |
| 9 | Agent system | ✅ Complete | Architect/Planner/Writer/Editor/Memory agents wired into pipeline execution |
| 10 | Genre packs | ✅ Complete | Genre pack loading integrated into runtime prompts |
| 11 | Output pipeline | ✅ Complete | Chapter stitching and manuscript export validated (md) |
| 12 | Integration and preflight | 🔵 In progress | Added Hypura model inventory + KV confidence diagnostics in `scripts/preflight.py`; rerun smoke to revalidate |
| 13 | TTS integration | 🔵 In progress | Direct Gradio HTTP integration, manual Gradio review gating, and repo-local narration validation now work; root causes fixed were malformed `generate` API naming, missing Gradio `State` input in the direct payload, and an overly short 60s sentence timeout. Remaining work is to run the repo-local UI on the canonical port and validate a full chapter narration resume after manual approval. |

## Immediate Notes
- `pipeline_novel.py` is now the active runtime entrypoint for chapter generation.
- Reused modules were copied from `Story_Time` where they are still structurally useful.
- Port `7865` was serving `/Users/wes/Desktop/Story_Time/chatterbox`, not this repo-local `chatterbox/` app.
- Step 3 should use the repo-local Gradio TTS UI for voice tuning before narration is approved.
- Root-level `config.py` exists only as a compatibility shim for copied modules.
- Runtime memory guardrails are now staged (warn -> throttle -> auto-pause -> emergency stop) via environment thresholds.
- Pipeline startup now emits diagnostics artifacts under `.state/diagnostics/runs/` with router health and Hypura model inventory evidence.
- Per-chapter inference report artifacts now include fallback evidence and memory events.

## Implementation Strategy
- Use Ollama for structural work only: Architect, Planner, and Memory Manager stay on the small fast model.
- Use Hypura for prose work only: Writer and Editor run on the large TurboQuant GGUF model.
- Store large GGUF files on the external M.2 SSD and point `HYPURA_MODELS_DIR` there.
- Do not force a dual-large-model design on day one. Keep one large prose model in Hypura and one small structural model in Ollama.
- Treat TurboQuant as a prose-model optimization, not as a prerequisite for every model in the stack.
- Keep TTS out of the critical path until the text pipeline reaches checkpoint-resume stability.

## Recommended Initial Model Split
- Structural lane: `phi3.5:3.8b-mini-instruct-q5_K_M` via Ollama
- Prose lane on 16 GB hardware: Mixtral 8x7B TurboQuant GGUF via Hypura
- Prose fallback lane: keep Ollama available as a fallback if Hypura is offline during early testing

## External SSD Notes
- Minimum target free space: 100 GB after model placement, even though the current drive has more available.
- Place only large GGUF artifacts on the external drive; keep project code, SQLite, and ChromaDB local for lower-latency metadata access.
- Validate sustained access through Hypura before starting quantization jobs.

## Current File Reuse
- `engine/story_lint.py`: copied directly
- `engine/local_llm.py`: copied and now being adapted to dual-client routing
- `engine/tts_engine.py`: copied for later activation
- `ui/`: copied for later adaptation
- `scripts/preflight.py`: copied for later Hypura and Chroma checks

## Acceptance Gates
1. Preflight passes for Ollama, Hypura, ChromaDB, and external SSD path.
2. Structural and prose clients both answer through one common interface.
3. FSM can persist and load checkpoints.
4. DB and RAG layers can round-trip canonical story state.
5. Export pipeline can emit manuscript output before TTS is added.
6. TTS is enabled only after the prose pipeline is stable.
