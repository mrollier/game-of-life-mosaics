#!/usr/bin/env bash
#
# Deploy the Gradio web app to the Hugging Face Space.
#
# Rebuilds the self-contained staging folder (hf_space/) from the current
# working tree, then uploads it over HTTP (large files go up as LFS
# automatically). Run from anywhere; paths are resolved relative to this script.
#
# Usage:
#   ./deploy.sh ["commit message"]
#
# Prerequisites (one-time):
#   hf auth login        # token with WRITE permission
#
set -euo pipefail

# Resolve the repo root (directory containing this script).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SPACE_ID="mrollier/game-of-life-mosaics"
STAGING="hf_space"
COMMIT_MSG="${1:-Deploy Game of Life Mosaics app}"

echo "==> Rebuilding staging folder: $STAGING/"
rm -rf "$STAGING"
mkdir "$STAGING"

# App entrypoint + runtime deps.
cp app.py requirements.txt "$STAGING/"

# The Space reads its config from README.md's YAML front matter.
cp README_HF.md "$STAGING/README.md"

# Ship the package at the Space root so `import gol_mosaics` works with no
# pip install (data files in gol_mosaics/data/ come along).
cp -R src/gol_mosaics "$STAGING/gol_mosaics"

# Strip caches that may have been copied.
find "$STAGING" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGING" -name '*.pyc' -delete

# Track the large pattern file(s) as LFS.
printf '*.npy filter=lfs diff=lfs merge=lfs -text\n' > "$STAGING/.gitattributes"

echo "==> Confirming HF login"
hf auth whoami

echo "==> Uploading to https://huggingface.co/spaces/$SPACE_ID"
hf upload "$SPACE_ID" "$STAGING" . --repo-type space --commit-message "$COMMIT_MSG"

echo "==> Done. The Space will rebuild automatically:"
echo "    https://huggingface.co/spaces/$SPACE_ID"
