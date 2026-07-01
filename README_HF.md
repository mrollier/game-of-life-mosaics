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

> **Background removal:** if your upload still has its background, it is removed
> automatically (via `rembg`/u2net) so the subject stands out. The **Remove
> background** toggle switches between the subject-only and original versions,
> and the *Input preview* shows which one feeds the mosaic. The first removal
> after the Space starts is slow (the ~176 MB model downloads once), then cached.

## Settings
- **Remove background** — auto-detected on upload; toggle to keep the original.
- **Detail level** (3–5) — higher is finer; level 5 is slower.
- **Colour scheme** — UGent, monochrome, random Warhol pop colours, or manual.
- **Grid size** — number of tiles across (10–200, even). Level 5 at 200 is slow.
- **Advanced** — empty-tiles cutoff, alpha cutoff, ECA background rule (curated,
  random, or a custom Wolfram rule 0–255), and the background pattern size (ECA
  cell size). Use **🎲 New variation** to reroll the look.
- **Download .cells** — export the still-life pattern for the
  [Golly](https://golly.sourceforge.io) simulator (the ECA backdrop is not
  exported, since it isn't made of stable Life patterns).

## Credits
Source: [github.com/mrollier/game-of-life-mosaics](https://github.com/mrollier/game-of-life-mosaics).
The idea of Game of Life still-life mosaics is from Robert Bosch, *Opt Art: From
Mathematical Optimization to Visual Design* (Princeton University Press, 2019).

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Then open the printed local URL (default http://127.0.0.1:7860).

## Notes for deploying this Space
- This app uses the runtime dependencies in `requirements.txt`
  (numpy, scipy, cellpylib, Pillow, gradio, and `rembg[cpu]` for background
  removal). It does **not** install `gurobipy` (pattern generation only).
- `rembg` pulls in `onnxruntime`, which enlarges the build, and the u2net model
  (~176 MB) downloads on the first background removal after a (re)start.
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
