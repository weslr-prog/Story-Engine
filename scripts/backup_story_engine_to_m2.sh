#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_BASE="${1:-/Volumes/256 M.2/Story-Engine-backups}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${DEST_BASE}/story_engine_backup_${STAMP}"
ARCHIVE_DIR="${RUN_DIR}/archives"
BUNDLE_DIR="${RUN_DIR}/bundles"
STATUS_DIR="${RUN_DIR}/status"

mkdir -p "${ARCHIVE_DIR}" "${BUNDLE_DIR}" "${STATUS_DIR}"

repo_head() {
  local repo="$1"
  if [ -d "${repo}/.git" ]; then
    (cd "${repo}" && git rev-parse --short HEAD)
  else
    echo "n/a"
  fi
}

capture_status() {
  local name="$1"
  local repo="$2"
  if [ -d "${repo}/.git" ]; then
    (
      cd "${repo}"
      {
        echo "repo: ${repo}"
        echo "head: $(git rev-parse HEAD)"
        echo "branch: $(git rev-parse --abbrev-ref HEAD)"
        echo "remote:"
        git remote -v || true
        echo
        echo "status --short:"
        git status --short || true
      } > "${STATUS_DIR}/${name}_git_status.txt"
    )
  else
    printf 'repo: %s\nhead: n/a\nstatus: not a git repo in this checkout\n' "${repo}" > "${STATUS_DIR}/${name}_git_status.txt"
  fi
}

make_archive() {
  local name="$1"
  local dir="$2"
  if [ ! -d "${dir}" ]; then
    return
  fi

  tar -C "${dir}" \
    --exclude '.git' \
    --exclude '.DS_Store' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude '.mypy_cache' \
    --exclude '.venv' \
    -czf "${ARCHIVE_DIR}/${name}.tar.gz" .
}

make_bundle() {
  local name="$1"
  local repo="$2"
  if [ -d "${repo}/.git" ]; then
    (cd "${repo}" && git bundle create "${BUNDLE_DIR}/${name}.bundle" --all)
  fi
}

ROOT_REPO="${ROOT_DIR}"
CHATTERBOX_REPO="${ROOT_DIR}/chatterbox"
HYPURA_REPO="${ROOT_DIR}/third_party/hypura"

capture_status "root" "${ROOT_REPO}"
capture_status "chatterbox" "${CHATTERBOX_REPO}"
capture_status "hypura" "${HYPURA_REPO}"

make_archive "story_engine_root_snapshot" "${ROOT_REPO}"
make_archive "chatterbox_snapshot" "${CHATTERBOX_REPO}"
make_archive "hypura_snapshot" "${HYPURA_REPO}"

make_bundle "story_engine" "${ROOT_REPO}"
make_bundle "chatterbox" "${CHATTERBOX_REPO}"
make_bundle "hypura" "${HYPURA_REPO}"

{
  echo "backup_timestamp: ${STAMP}"
  echo "destination: ${RUN_DIR}"
  echo "root_head: $(repo_head "${ROOT_REPO}")"
  echo "chatterbox_head: $(repo_head "${CHATTERBOX_REPO}")"
  echo "hypura_head: $(repo_head "${HYPURA_REPO}")"
} > "${RUN_DIR}/manifest.txt"

(
  cd "${RUN_DIR}"
  find . -type f \
    ! -name 'SHA256SUMS.txt' \
    -print0 | xargs -0 shasum -a 256 > SHA256SUMS.txt
)

cat > "${RUN_DIR}/RESTORE_CHECK.md" <<'EOF'
# Restore Check

1. Validate checksums:

```bash
cd <backup_run_dir>
shasum -a 256 -c SHA256SUMS.txt
```

2. Inspect snapshot archive:

```bash
tar -tzf archives/story_engine_root_snapshot.tar.gz | head
```

3. Rehydrate git history from bundle (example):

```bash
git clone bundles/story_engine.bundle restore-story-engine
```
EOF

echo "Backup complete: ${RUN_DIR}"
