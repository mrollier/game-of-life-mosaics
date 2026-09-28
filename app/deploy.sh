#!/usr/bin/env bash
#
# Deploy the Gradio web app to the Hugging Face Space.
#
# Stages a self-contained copy of the app (app.py, its requirements, the
# Space README and the gol_mosaics package) in a temporary folder and uploads
# it over HTTP (large files go up as LFS automatically). Refuses to deploy a
# working tree with uncommitted changes, and runs the tests first, so what
# goes live is always a commit that passed them.
#
# Usage (from anywhere):
#   app/deploy.sh [--dry-run] [--skip-tests] ["commit message"]
#
#   --dry-run     stage and list the files, but do not upload
#   --skip-tests  skip the test run (the clean-tree check still applies)
#
# Prerequisites (one-time):
#   hf auth login        # token with WRITE permission
#
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$APP_DIR")"
cd "$REPO_ROOT"

SPACE_ID="mrollier/game-of-life-mosaics"
DRY_RUN=0
RUN_TESTS=1
COMMIT_MSG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --skip-tests) RUN_TESTS=0 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) COMMIT_MSG="$arg" ;;
    esac
done
REVISION="$(git rev-parse --short HEAD)"
COMMIT_MSG="${COMMIT_MSG:-Deploy Game of Life Mosaics app ($REVISION)}"

echo "==> Checking for uncommitted changes"
if [ -n "$(git status --porcelain -- src app)" ]; then
    git status --short -- src app
    echo "Refusing to deploy: commit or stash the changes above first." >&2
    exit 1
fi

if [ "$RUN_TESTS" -eq 1 ]; then
    echo "==> Running the library and app tests"
    python -m pytest -q -p no:cacheprovider tests \
        --ignore=tests/freeform --ignore=tests/experiments
fi

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
echo "==> Staging $REVISION in $STAGING"

# App entrypoint + runtime deps; the Space reads its config from README.md's
# YAML front matter.
cp app/app.py app/requirements.txt "$STAGING/"
cp app/README_space.md "$STAGING/README.md"

# Ship the package at the Space root so `import gol_mosaics` works with no pip
# install (data files in gol_mosaics/data/ come along). The free-form solver
# needs OR-Tools, which the Space does not install, and the app does not use
# it, so it stays behind.
cp -R src/gol_mosaics "$STAGING/gol_mosaics"
rm -rf "$STAGING/gol_mosaics/freeform"
find "$STAGING" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGING" -name '*.pyc' -delete

# Track the pattern files as LFS.
printf '*.npy filter=lfs diff=lfs merge=lfs -text\n' > "$STAGING/.gitattributes"

(cd "$STAGING" && find . -type f | sort)

if [ "$DRY_RUN" -eq 1 ]; then
    echo "==> Dry run: nothing uploaded"
    exit 0
fi

echo "==> Confirming HF login"
hf auth whoami

# --delete '*' makes the Space mirror the staged folder: files the package no
# longer has (renamed modules, old data files) are removed in the same commit.
echo "==> Uploading to https://huggingface.co/spaces/$SPACE_ID"
hf upload "$SPACE_ID" "$STAGING" . --repo-type space --delete '*' \
    --commit-message "$COMMIT_MSG"

echo "==> Done. The Space will rebuild automatically:"
echo "    https://huggingface.co/spaces/$SPACE_ID"
