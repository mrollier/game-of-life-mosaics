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
- **Tile shape** — *Diamonds* (the classic 45° layout of diamond tiles glued by
  shared ponds) or *Squares* (axis-aligned square tiles bordered by a ring of
  ponds, sharing their border ponds). Both stay a provable still life.
- **Detail level** — higher is finer; diamonds offer 3–6 (level 6, 332k tiles,
  loads on first use), squares 3–5. Levels 5–6 are slower.
- **Pixel-perfect** — the downloaded PNG has one Game of Life cell per pixel.
  Viewers that smooth when zooming past 100 % make it look blurry; the file is
  sharp. Export from Canva as PNG, without enlarging it past its native size.
- **Colour scheme** — UGent, monochrome, random Warhol pop colours, or manual.
- **Grid size** — number of tiles across (10–200; the diamond layout rounds it
  to even). Levels 5–6 at 200 are slow.
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

## Deploying
This page is `app/README_space.md` in the source repository; `app/deploy.sh`
publishes the app here. See `docs/deploy.md` there.
