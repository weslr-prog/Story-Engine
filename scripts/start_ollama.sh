#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-phi3.5:3.8b-mini-instruct-q5_K_M}"
ollama run "$MODEL"
