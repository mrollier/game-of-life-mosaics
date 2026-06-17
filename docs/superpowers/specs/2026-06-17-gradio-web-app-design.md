# Gradio web app for `gol_mosaics` — design

Date: 2026-06-17

## Goal
A single-file Gradio app (`app.py`, repo root) that wraps the existing
`gol_mosaics` pipeline: upload an image, choose settings, get a Game of Life
mosaic back. Deployable as a free Hugging Face Gradio Space and embeddable via
iframe in an external Hugo site. Faithful to the existing pipeline; no breaking
changes to the public API.

## Key decisions (agreed with user)
- **No background removal in the app.** rembg / onnxruntime / the 176 MB u2net
  model are kept out entirely to keep the Space lightweight and reliable on the
  free CPU tier. The app calls the pipeline with `remove_background=False` so it
  never touches rembg at runtime. Instead, a prominent UI note tells users to
  upload a background-free image (e.g. via remove.bg). A line in the README and
  the About accordion notes automatic background removal may arrive in a future
  version. The core library keeps its existing *optional* rembg path untouched.
- **Seed = global.** Optional `seed`; when set, `np.random.seed(seed)` before
  generation. Documented caveat: `ColorScheme.warhol()` uses its own
  `np.random.default_rng()` and stays non-reproducible.
- **US-English code style** to match the existing code (`color`, `rim_color`).
- I produce and verify files locally; I do **not** push to GitHub/HF. The user
  performs the LFS commit (if any) and the Space deploy.

## Verified facts
- Public entry point: `MosaicGenerator.generate_from_image(image_path, ...)` with
  the real signature `empty_tiles_cutoff, alpha_cutoff, supersample, no_eca,
  remove_background, contrast, rim_color`. `preprocess_for_mosaic` returns a
  7-tuple (adds footprint). 52 tests currently pass.
- Data files load from `../../data/*.npy` relative to `patterns.py` (fragile);
  `pyproject.toml` package-data also points at `../../data/*.npy`.
- Color schemes: `ugent`, `monochrome`, `warhol`. ECA rules:
  `COMPLEX_RULES = [54,147,110,124,137,193]`, `CHAOTIC_RULES = [30,45,106,150]`.

## Step 1 — Non-breaking library refactors
1. **In-memory entry point.** Add `MosaicGenerator.generate_from_pil(img, ...)`
   carrying the full real signature. Refactor `generate_from_image(path, ...)` to
   open the file and delegate. Add an `image` kwarg to
   `ImageProcessor.preprocess_for_mosaic`/`load_image` accepting a path *or* a
   `PIL.Image` (path stays the positional default). Additive only — no existing
   signatures change.
2. **Data files into the package.** Move `data/solutions_pattern_level_{1..5}.npy`
   to `src/gol_mosaics/data/`, load via
   `importlib.resources.files("gol_mosaics.data")`. Update `pyproject.toml`
   package-data to `gol_mosaics = ["data/*.npy"]` and `MANIFEST.in`. Grep-verify
   no other references to the old path.
3. **Seed.** Optional `seed` on the generation methods; `np.random.seed(seed)`
   when provided. Document the Warhol caveat.

After each: `pytest tests/ -v` must stay green (52 passing).

## Step 2 — Gradio app (`app.py`, repo root)
Controls / defaults:
- Image upload (PNG/JPG) with a prominent "upload a background-free image" note.
- `level`: dropdown 3/4/5, default 4, with a "level 5 is slower" warning.
- `color_scheme`: dropdown UGent / monochrome / Warhol, default UGent.
- `grid_size`: slider 40–120, **even-enforced** (step 2).
- `empty_tiles_cutoff`: slider 0–1, default 0.65.
- `alpha_cutoff`: slider 0–1, default 0.5.
- `eca_rule`: dropdown of curated COMPLEX + CHAOTIC rules (labelled) plus a
  "random" option. Not a free 0–255 field.
- `seed`: optional number (blank = random).
- Generate button, output image, download, "About" accordion.

Behaviour:
- Preload pattern libraries 1–5 once at startup.
- Gradio queue with capped concurrency (e.g. `default_concurrency_limit=2`).
- Validation: reject non-images / oversized files; set `Image.MAX_IMAGE_PIXELS`;
  cap `grid_size`/`level`. Friendly error messages, no crashes.
- Always call the pipeline with `remove_background=False`. No GIF support.

## Step 3 — HF Space deployment files
- `requirements.txt`: numpy, scipy, cellpylib, Pillow, gradio (pinned). No
  gurobipy, no rembg.
- Space `README.md` with verified Gradio YAML front matter
  (`title, emoji, colorFrom, colorTo, sdk: gradio, sdk_version, app_file: app.py,
  pinned, license, short_description`), a note that the level-5 `.npy` (~19 MB)
  is best tracked via Git LFS on HF, a local-run section (`python app.py`), and a
  "background removal may arrive in a future version" note.
- `.gitattributes` LFS rule for `*.npy` (and document the git-lfs requirement).
- `.gitignore` for Python/Gradio artifacts (`flagged/`, `__pycache__`, etc.).

## Step 4 — Verify
- `pytest tests/ -v` green.
- Install gradio; run the app locally and generate from `input/images/john.png`
  at level 4 and level 5; confirm output renders and downloads.
- Add lightweight smoke tests: the in-memory generation path returns a valid
  RGBA image on the example image.

## Out of scope
- Background removal in the app (future version).
- GIF processing.
- Pushing to GitHub / deploying the Space (user does this).

## Known environment gaps (flagged to user)
- Gradio not installed in the `gol-mosaics` env → will `pip install gradio` for
  local verification (user approved).
- Git LFS not installed → only matters for the 19 MB level-5 `.npy` on HF; the
  `.gitattributes` rule is added and the install step documented.
