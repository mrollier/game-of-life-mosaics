---
title: Game of Life Mosaics
emoji: 🔬
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
license: mit
short_description: Turn a photo into a Conway's Game of Life mosaic
---

# Game of Life Mosaics

Turn a portrait into a mosaic built entirely from **Conway's Game of Life still
lifes**, on an **Elementary Cellular Automaton** background. Upload an image,
pick a few settings, and download the result.

> **Tip:** for the best result, upload an image whose **background has already
> been removed** (e.g. with [remove.bg](https://www.remove.bg)) — the subject
> should sit on a transparent background. Automatic background removal may be
> added in a future version.

## Settings
- **Detail level** (3–5) — higher is finer; level 5 is slower.
- **Colour scheme** — UGent, monochrome, or random Warhol pop colours.
- **Grid size** — number of tiles across (even).
- **Advanced** — empty-tiles cutoff, alpha cutoff, ECA background rule, and an
  optional seed for reproducible output (the Warhol scheme stays random).

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Then open the printed local URL (default http://127.0.0.1:7860).

## Notes for deploying this Space
- This app uses only the runtime dependencies in `requirements.txt`
  (numpy, scipy, cellpylib, Pillow, gradio). It does **not** install
  `gurobipy` (pattern generation only) or `rembg` (background removal).
- The pattern libraries ship inside the package at
  `src/gol_mosaics/data/*.npy`. The level-5 file is ~19 MB; Hugging Face Spaces
  require files over 10 MB to be tracked with **Git LFS**:
  ```bash
  brew install git-lfs      # or your platform's installer
  git lfs install
  git lfs track "*.npy"     # writes the rule into .gitattributes
  git add .gitattributes src/gol_mosaics/data/*.npy
  git commit -m "Track pattern libraries with Git LFS"
  ```
- To use this file as the Space landing page, it must be named `README.md` at
  the Space repo root (Hugging Face reads the YAML front matter above). Either
  rename it on the Space, or copy its contents into the root `README.md`.
