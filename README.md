# Story-Engine

Story-Engine is a new offline-first long-form fiction pipeline designed around a Python state machine, structured story memory, and dual local inference backends:

- Ollama + Phi-3.5 Mini for structural planning and memory updates
- Hypura + TurboQuant large GGUF models for prose generation and revision
- ChromaDB for long-term retrieval
- SQLite/SQLAlchemy for canonical story state
- Chatterbox TTS as a later-phase output step

## Status

This repository is scaffolded for the new engine architecture. The current implementation focus is:

1. Core project scaffold
2. Story bible database layer
3. RAG memory layer
4. Inference routing
5. FSM orchestrator
6. TTS integration after prose pipeline validation

## Planned build sequence

See BUILD.md for the tracked implementation phases.

## Model storage

Large GGUF models should live on the external M.2 SSD. See models/README.md.

## Current structure

- engine/: core runtime modules
- genre_packs/: prompt and rule packs per genre
- ui/: Story-Engine local studio backend and dashboard
- scripts/: operational scripts and preflight checks
- projects/: per-story working directories and template inputs
- chatterbox/: repo-local TTS integration source
