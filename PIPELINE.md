# The mosaic generation pipeline, step by step

This document traces **exactly** what happens, in order, when you turn an input
image into a Game-of-Life mosaic. It is meant as a map: when something looks
wrong in the output, find the symptom in the last section and it will point you
at the stage (and the file/line) responsible, so you can tell me precisely where
to look.

Everything below describes the **current code** on this branch (the "raw
original" pipeline, with no forced rim).

Notation used throughout:
- `G` = `grid_size` (even integer, e.g. 40, 60, 80, 100, 120).
- `L` = `level` (3, 4 or 5 are pre-computed).
- `t` = tile edge in pixels = `6 * L` (level 3 → 18, level 4 → 24, level 5 → 30).
  Each Game-of-Life still-life tile is a `t × t` binary block.
- `R` = the rotated diamond's side after stage 2b ≈ `√2 · G` (e.g. G=40 → 56,
  G=60 → 84, G=100 → 140). **The diagonal grids and the whole mosaic are sized
  off `R`, not `G`** — this is the single most important thing to get right when
  predicting output shapes.

---

## 0. Entry points

| Call | File | Notes |
|------|------|-------|
| `generate_from_image(path, …)` | [mosaic.py:149](src/gol_mosaics/mosaic.py#L149) | Opens the file, forwards to `generate_from_pil`. |
| `generate_from_pil(img, …)` | [mosaic.py:190](src/gol_mosaics/mosaic.py#L190) | **The actual pipeline.** Everything below happens here. |
| `generate_from_gif(path, …)` | [mosaic.py:301](src/gol_mosaics/mosaic.py#L301) | Runs `generate_from_pil` per frame. |
| Web app `render_mosaic(…)` | [app.py:266](app.py#L266) | Calls `generate_from_pil`, then post-fits to aspect ratio (stage 9). |

The orchestration for one image is the body of `generate_from_pil`
([mosaic.py:190](src/gol_mosaics/mosaic.py#L190) onward). The stages below are
that body in order.

---

## 1. Seed the RNG (optional)

[mosaic.py:250-251](src/gol_mosaics/mosaic.py#L250-L251)

If `seed` is given, `np.random.seed(seed)` is set. This makes pattern selection,
supersample selection and ECA initial state reproducible.

> ⚠️ **Caveat:** the ECA *rule* is chosen in `__init__`
> ([mosaic.py:112](src/gol_mosaics/mosaic.py#L112)) **before** this reseed, so a
> "random" rule is not controlled by this seed. The web app works around this by
> seeding again before constructing the generator ([app.py:230](app.py#L230)).

---

## 2. Preprocess the image → diagonal grids

Called at [mosaic.py:254](src/gol_mosaics/mosaic.py#L254);
implemented in `ImageProcessor.preprocess_for_mosaic`
([image_processing.py:393](src/gol_mosaics/image_processing.py#L393)).

This stage produces **five** arrays: `lowres_first, lowres_second` (greyscale
tile grids), `mask_first, mask_second` (alpha tile grids), and `aspect_ratio`.

### 2a. Load + alpha extraction — `load_image` ([image_processing.py:40](src/gol_mosaics/image_processing.py#L40))
1. Open and convert to `RGBA`.
2. **Background removal** (`remove_background`, default `'auto'`): if the image
   still has an opaque background (`has_background()`,
   [image_processing.py:116](src/gol_mosaics/image_processing.py#L116)), run
   `rembg` to cut out the subject. The web app passes `False` (never runs it).
3. **Extract the alpha channel** as `mask` — this is the single source of truth
   for "subject vs background".
4. Composite the RGBA over a solid `alpha_color` (white) background, convert to
   **greyscale `L`**.
5. **Contrast** (`enhance_contrast`,
   [image_processing.py:141](src/gol_mosaics/image_processing.py#L141)): a
   sigmoid S-curve (`contrast=5.0` default) pushes the greyscale toward black/white.

   Returns `(grey_image, alpha_mask)`.

### 2b. Square + rotate + pixelate (done **twice**: greyscale and mask)
For the greyscale ([image_processing.py:436-438](src/gol_mosaics/image_processing.py#L436-L438)):
1. `square_image` ([image_processing.py:260](src/gol_mosaics/image_processing.py#L260)):
   pad the shorter side with `fill_color='white'` to make it square. Records
   `aspect_ratio = original_width / original_height`. **Padding, never cropping.**
2. `rotate_and_pixelate` ([image_processing.py:295](src/gol_mosaics/image_processing.py#L295)):
   - resize to `G × G` (Lanczos),
   - **rotate 45°** with `expand=True`, filling exposed corners with `255` (white),
   - convert to numpy and **trim one pixel off every edge**: `arr[1:-1, 1:-1]`.

   Result: the rotated diamond array, side `R ≈ √2·G` (e.g. G=40 → 56). PIL's
   `expand=True` rounds the bounding box **up** before the `[1:-1,1:-1]` trim, so
   `R` is a touch larger than `√2·G − 2` (measured `R = 1.4·G` for the standard
   grid sizes). Everything downstream is sized off this `R`.

The mask is run through the **same** square→rotate→pixelate path
([image_processing.py:441-443](src/gol_mosaics/image_processing.py#L441-L443))
with white padding, so the mask and greyscale stay pixel-aligned.

### 2c. Extract the two interlocking diagonal grids — `extract_diagonal_patterns` ([image_processing.py:342](src/gol_mosaics/image_processing.py#L342))
The 45°-rotated diamond is sampled along two diagonal lattices, giving two grids.
`extract_diagonal_patterns` reads its size from `grid_size = lowres.shape[0]`,
i.e. the **rotated** side `R`, not `G`:
- `first`  → shape `(R/2 + 1, R/2)`  (e.g. G=40 → `(29, 28)`)
- `second` → shape `(R/2, R/2 + 1)`  (e.g. G=40 → `(28, 29)`)

These two grids are the even/odd tiles that interlock into the diamond mosaic.
Because `R ≈ √2·G`, each grid is ≈`0.7·G` tiles across — **not** `0.5·G`.

> ⚠️ **Fragile spot #1 — the rotation/trim/diagonal-index geometry.** The index
> math in `extract_diagonal_patterns` works entirely off `R = lowres.shape[0]`
> (the rotated size), so it is the `[1:-1, 1:-1]` trim in `rotate_and_pixelate`
> that fixes the exact relationship between `R` and `G`. The hard-coded trim and
> the half-open `range(R//2 …)` bounds are exactly where "cropped half-tile" /
> "too few tiles at the edge" symptoms originate. If the edges look wrong, this
> is the first place to look.

---

## 3. Build the GoL mosaic — `_build_mosaic` ([mosaic.py:370](src/gol_mosaics/mosaic.py#L370))

For each diagonal grid:
1. **Map greyscale → still-life tiles** via
   `get_patterns_for_values` ([patterns.py:416](src/gol_mosaics/patterns.py#L416)):
   - values **above `empty_tiles_cutoff`** (default 0.65) become an **empty
     tile** (all zeros);
   - otherwise the value is matched to the still-life whose density is closest
     (`invert=True` ⇒ dark = dense), picking randomly among ties if
     `random_patterns`.

   Each grid cell becomes a `t × t` binary tile → grid shape becomes `(rows, cols, t, t)`.
2. **Assemble** the tiles into one big array with `np.block`
   (`_assemble_tiles`, [mosaic.py:409](src/gol_mosaics/mosaic.py#L409)).
3. **Pad and overlay** the two diagonals so they interlock
   (`_pad_diagonals`, [mosaic.py:416](src/gol_mosaics/mosaic.py#L416)):
   - the pad size is `PatternLibrary.tile_pad_size`
     `= ((6-3)·(2L-1) + 1 + 2) // 2 = 3L = t/2`,
   - `first` is padded left/right, `second` is padded top/bottom,
   - the two are **added** → final square mosaic of side `(R/2 + 1) · t` pixels
     (e.g. G=40, L=3 → `29 · 18 = 522`; **not** `(G/2 + 1) · t = 378`).

> 📐 **Why `tile_pad_size` works:** padding each diagonal by exactly `t/2` makes
> the two complementary grids land on the same shape and interlock. If you ever
> change tiling, this property (defined on `PatternLibrary`) must stay equal to
> `t/2` or the diagonals de-align. This is **Fragile spot #2**.

The mosaic is **binary**: `0 = GoL background`, `1 = GoL pixel`.

---

## 4. Build the transparency mask — `_build_mask` ([mosaic.py:433](src/gol_mosaics/mosaic.py#L433))

Same block/pad geometry as stage 3, but driven by the **alpha** grids via
`get_patterns_for_mask` ([patterns.py:478](src/gol_mosaics/patterns.py#L478)):

- alpha value **≥ `alpha_cutoff`** (default 0.5, i.e. opaque subject) → **empty
  tile (0)**;
- alpha value **< `alpha_cutoff`** (transparent background) → **filled tile (1)**.

The "filled tile" is `binary_fill_holes(solutions[-1])`
([patterns.py:512](src/gol_mosaics/patterns.py#L512)) — a filled *diamond*
(~44% ones for L=4), **not** a solid `t × t` square. Because the two diagonal
mask grids interlock with the same geometry as the GoL tiles, those diamonds
still tile to cover the whole background region; the small-hole fill below
closes any residue.

So in the resulting `mask`: **`1` = background region (gets ECA), `0` = subject
region (shows the GoL mosaic).**

Finally: `mask = _fill_small_holes(mask, max_hole_size=t·t/4)`
([mosaic.py:472](src/gol_mosaics/mosaic.py#L472)) — only holes smaller than a
quarter tile are filled.

> ⚠️ **Fragile spot #3 — hole filling.** A plain `binary_fill_holes` here used
> to swallow a subject that touched no image edge (the whole enclosed subject
> counted as one big "hole"). `_fill_small_holes`
> ([mosaic.py:21](src/gol_mosaics/mosaic.py#L21)) fixes this by filling only
> the tiny inter-tile gaps and leaving large enclosed zero-regions (the
> subject) open. The quarter-tile threshold is the invariant to preserve.

---

## 5. Fit to the original aspect ratio — `_adjust_aspect_ratio` ([mosaic.py:479](src/gol_mosaics/mosaic.py#L479))

The mosaic and mask are currently a **square**. This crops them back to the
original `aspect_ratio`:
- `aspect_ratio == 1.0` → no change.
- wider than tall → crop **height**; taller than wide → crop **width**.
- the new dimension is rounded **up** to a whole number of tiles
  (`ceil(new / tile) * tile`) and the crop is centred.

> ⚠️ **Fragile spot #4 — centred tile-rounded crop.** Rounding up to whole tiles
> plus integer-centring (`(total - new)//2`) means the crop rarely matches the
> original ratio exactly, and the centring can shave a different amount off
> opposite sides. Combined with stage 2's diamond geometry, this is a second
> contributor to uneven edges / half-tiles.

---

## 6. Choose the ECA supersample ([mosaic.py:287](src/gol_mosaics/mosaic.py#L287))

If `supersample` was not supplied, `generate_from_pil` uses
`max(1, min(15, mosaic_width))` — i.e. ~15-pixel ECA cells, clamped only on
tiny mosaics. (Historically this had to divide the width evenly; it no longer
does — see stage 7.)

---

## 7. Generate + composite the ECA background — `_apply_eca_background` ([mosaic.py:523](src/gol_mosaics/mosaic.py#L523))

1. **Generate the ECA pattern** (`ECABackground.generate`,
   [eca.py:58](src/gol_mosaics/eca.py#L58)) unless `no_eca`:
   - build a 1-D elementary cellular automaton at low resolution
     `ceil(width/supersample) × ceil(height/supersample)`,
   - evolve it with the chosen Wolfram `rule`,
   - **upsample** by repeating each cell into a `supersample × supersample`
     block, then **crop** to the exact mosaic `(height, width)`.

   Result is binary: `0 = ECA background`, `1 = ECA pixel`.
2. **Combine mask + ECA** ([mosaic.py:558](src/gol_mosaics/mosaic.py#L558)):
   ```
   eca_mask = transparency_mask * (eca_pattern + transparency_mask)
   ```
   With `transparency_mask ∈ {0,1}` and `eca_pattern ∈ {0,1}` this yields
   three values: `0` where there is no background (subject), `1 = ECA
   background`, `2 = ECA pixel`.
3. **Render two layers and composite** (`render_full_mosaic`,
   [renderer.py:214](src/gol_mosaics/renderer.py#L214)):
   - `render_gol_mosaic` → base RGBA, **fully opaque**: `0→gol_background`,
     `1→gol_pixel`.
   - `render_eca_overlay` → overlay RGBA: `eca_mask 0 → transparent`,
     `1 → eca_background`, `2 → eca_pixel`, `3` and up → the filler levels of
     `filled_background`, graded from `eca_pixel` to the scheme's `fill`.
   - `Image.alpha_composite(base, overlay)`: the ECA shows over the background
     region; the subject region (overlay transparent) shows the GoL mosaic
     underneath.

> 🔎 **Rim behaviour (the thing we just reverted).** The 45°-rotation corners are
> white in both greyscale and mask. White greyscale → **empty GoL tiles** (`0`),
> and high alpha → **subject** in the mask (`0`, no ECA). So the corners render
> as the **opaque `gol_background` colour**. There is no transparency and no
> special-cased rim any more — the corners are just "empty subject". If you want
> different corner behaviour, this is the interaction to change.

The final image returned by `generate_from_pil` is this composited RGBA, sized
`height × width` from stage 5 — an **opaque rectangle**, no transparent pixels.

---

## 8. (GIF only) repeat per frame — `generate_from_gif` ([mosaic.py:301](src/gol_mosaics/mosaic.py#L301))

Each frame is run through stages 1-7 independently, in memory (via
`generate_from_pil`). Defaults match the still pipeline.

> ⚠️ **Caveat — the result is currently a single still, not an animation.**
> Every frame is processed and appended to `frames`, but the function only
> returns `frames[0]` ([mosaic.py:365-368](src/gol_mosaics/mosaic.py#L365-L368));
> `frames[1:]` are computed and then discarded (never attached via
> `append_images=`), so the caller cannot reassemble the animation even with
> `save_all=True`. This path also passes no `seed`, so with
> `random_patterns=True` the tiles would flicker frame to frame if the
> animation were reassembled.

---

## 9. (Web app only) post-fit to the upload's aspect ratio — `_fit_to_aspect` ([app.py:166](app.py#L166))

The library output is already cropped to roughly the right ratio (stage 5), but
the app pads it to **exactly** the upload's ratio on a solid backdrop:
1. pad the shorter axis with the **ECA background colour** (never crop),
2. paste the opaque mosaic centred on that backdrop.

So the padding **bars** are the ECA background colour, while the mosaic's own rim
corners are the **GoL background** colour (they are part of the opaque mosaic).
The two colours differ unless the scheme makes them equal.

---

## End-to-end data-shape summary

Recall `R ≈ √2·G` (the rotated diamond side; measured `1.4·G`). Everything from
stage 2c on is sized off `R`, **not** `G`.

| Stage | Array | Approx. shape |
|-------|-------|---------------|
| input | PIL image | `W₀ × H₀` |
| 2b | resized | `G × G` |
| 2b | rotated+trimmed | `R × R`, `R ≈ √2·G` (e.g. G=40 → 56) |
| 2c | diagonal grids | `(R/2+1, R/2)` and `(R/2, R/2+1)` |
| 3/4 | tiled + padded square | `(R/2+1)·t  ×  (R/2+1)·t` |
| 5 | aspect-cropped | `(R/2+1)·t` along one axis, tile-rounded on the other |
| 7 | final RGBA | same as stage 5 |
| 9 | app-fitted | exactly the upload's aspect ratio |

---

## Where things commonly go wrong → which stage to inspect

| Symptom | Most likely stage | Where to look |
|---------|-------------------|---------------|
| Edge shows **half-tiles** or **too few tiles**; ragged border | Stage 2c geometry (and stage 5 crop) | `extract_diagonal_patterns` + the `[1:-1,1:-1]` trim ([image_processing.py:338](src/gol_mosaics/image_processing.py#L338), [:342](src/gol_mosaics/image_processing.py#L342)); `_adjust_aspect_ratio` ([mosaic.py:479](src/gol_mosaics/mosaic.py#L479)) |
| Diagonals **don't interlock** / doubled or gapped tiles | Stage 3 padding | `PatternLibrary.tile_pad_size` must equal `t/2` (`_pad_diagonals`, [mosaic.py:416](src/gol_mosaics/mosaic.py#L416)) |
| **Whole subject disappears** or background floods in | Stage 4 | `_fill_small_holes` threshold + subject touching the border ([mosaic.py:472](src/gol_mosaics/mosaic.py#L472)) |
| Subject/background **swapped** (ECA where subject should be) | Stage 4 mask polarity | `get_patterns_for_mask` cutoff direction ([patterns.py:478](src/gol_mosaics/patterns.py#L478)); `alpha_cutoff` |
| Mosaic **too dense / too sparse / inverted tones** | Stage 3 mapping | `empty_tiles_cutoff`, `invert`, `contrast` ([patterns.py:416](src/gol_mosaics/patterns.py#L416)) |
| **Corner colour** unexpected (GoL bg instead of ECA bg) | Stage 7 rim interaction | corners are "empty subject" → opaque `gol_background` (see rim note above) |
| ECA cells **wrong size** or **misaligned** at the edge | Stages 6-7 | supersample choice + the upsample-then-crop in `generate` ([eca.py:103-121](src/gol_mosaics/eca.py#L103-L121)) |
| Output **ratio slightly off** the original | Stage 5 (lib) and/or stage 9 (app) | tile-rounded centred crop; `_fit_to_aspect` |
| **Random** result for "same" settings | Stage 1 seed scope | ECA rule chosen pre-reseed ([mosaic.py:112](src/gol_mosaics/mosaic.py#L112)) |
| Odd `grid_size` raises / surprising cut-off | `__init__` even-check + stage 2c | [mosaic.py:105](src/gol_mosaics/mosaic.py#L105) |

---

### How to use this with me

Point me at a **stage number** (or a row in the table above) and describe the
visual symptom. That tells me which array to inspect and which invariant is being
violated, instead of guessing across the whole pipeline.
