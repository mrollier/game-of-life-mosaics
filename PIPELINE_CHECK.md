# PIPELINE.md verification check

This document records a stage-by-stage audit of [PIPELINE.md](PIPELINE.md)
against the **current code** on this branch. Every claim (behaviour, file/line
citation, and array shape) was checked by reading the source and, where it
mattered, by running the real code to capture actual shapes and values.

**Method:** read all cited files
([mosaic.py](src/gol_mosaics/mosaic.py),
[image_processing.py](src/gol_mosaics/image_processing.py),
[patterns.py](src/gol_mosaics/patterns.py),
[eca.py](src/gol_mosaics/eca.py),
[renderer.py](src/gol_mosaics/renderer.py),
[app.py](app.py)), then traced shapes empirically through
`preprocess_for_mosaic → _build_mosaic → _build_mask → _adjust_aspect_ratio →
_apply_eca_background` for several `(grid_size, level)` pairs, including a
genuinely transparent input so the mask had both regions.

---

## Verdict at a glance

The **prose description of each stage is accurate** — the order of operations,
the polarity of the mask, the `pad_size = t/2` formula, the ECA composite
arithmetic, the rim behaviour, and every `file:line` citation all check out.

The **shape arithmetic is systematically wrong by a factor of ≈√2.** Wherever
the document writes the diagonal-grid or final-mosaic size in terms of `G/2`, it
should be `R/2`, where `R ≈ √2·G` is the *rotated* diamond size, not the
original `grid_size`. This affects Stage 2c, Stage 3, Stage 5 and three rows of
the data-shape summary table.

One genuine code/behaviour discrepancy: **Stage 8 (GIF) does not return an
animated GIF** — it returns only the first frame.

| Stage | Prose / behaviour | Line citations | Shapes |
|-------|-------------------|----------------|--------|
| 0 Entry points | ✅ | ✅ | — |
| 1 Seed RNG | ✅ (incl. ECA-rule-pre-reseed caveat) | ✅ | — |
| 2a Load + alpha | ✅ | ✅ | — |
| 2b Square/rotate/pixelate | ✅ | ✅ | ✅ (rotated size) |
| 2c Diagonal grids | ✅ (mechanism) | ✅ | ❌ **wrong (√2 factor)** |
| 3 `_build_mosaic` | ✅ | ✅ | ❌ final side wrong (√2) |
| 4 `_build_mask` | ✅ (see "filled diamond" nuance) | ✅ | ✅ |
| 5 `_adjust_aspect_ratio` | ✅ | ✅ | ⚠️ inherits √2 error |
| 6 `_auto_select_supersample` | ✅ | ✅ | — |
| 7 `_apply_eca_background` | ✅ | ✅ | ✅ |
| 8 GIF | ⚠️ **returns one frame, not an animation** | ✅ | — |
| 9 App `_fit_to_aspect` | ✅ | ✅ | ✅ |

---

## Finding 1 (major): the `G/2` vs `√2·G/2` shape error

PIPELINE.md fixes its notation as `G = grid_size` (the even integer 40/60/…/120,
Stage 2b resizes to `G × G`). It then says in **Stage 2c**:

> - `first`  → shape `(G/2 + 1, G/2)`
> - `second` → shape `(G/2, G/2 + 1)`

This is not what the code produces. In
[`extract_diagonal_patterns`](src/gol_mosaics/image_processing.py#L363) the very
first line is:

```python
grid_size = lowres.shape[0]
```

`lowres` is the output of
[`rotate_and_pixelate`](src/gol_mosaics/image_processing.py#L295), i.e. the
image **after** the 45° `expand=True` rotation and the `[1:-1, 1:-1]` trim. Its
side is `R ≈ √2·G`, *not* `G`. So the diagonal grids are sized off `R`, giving
`(R//2 + 1, R//2)` and `(R//2, R//2 + 1)` — roughly `(0.7·G, 0.7·G)`, not
`(0.5·G, 0.5·G)`.

The function's **own docstring is internally inconsistent** about this: it shows
`diag1.shape == (16, 16)` "Approximately grid_size/2 + 1" for `grid_size=30`,
but it operates on the already-rotated `lowres` (≈41×41), so the real answer is
≈`(21, 20)`. The docstring made the same `G`-vs-`R` slip the pipeline doc did.

### Measured shapes (ground truth)

| `G` | rotated+trim `R` | `first` (actual) | doc claims `(G/2+1, G/2)` |
|----:|-----------------:|-----------------:|--------------------------:|
| 40  | `(56, 56)`  | `(29, 28)` | `(21, 20)` |
| 60  | `(84, 84)`  | `(43, 42)` | `(31, 30)` |
| 100 | `(140, 140)`| `(71, 70)` | `(51, 50)` |

Empirically `R = 1.4·G` for these sizes (PIL rounds the `√2·G` bounding box up,
then the `[1:-1,1:-1]` trim removes 2), so `first.shape[0] = R//2 + 1`.

### How it propagates to Stage 3 / Stage 5

The final square side is `(R//2 + 1)·t`, **not** `(G/2 + 1)·t`. Measured:

| `G`, `L` | `t = 6L` | `_build_mosaic` side (actual) | doc `(G/2+1)·t` |
|----------|---------:|------------------------------:|----------------:|
| 40, 3 | 18 | **522** (= 29·18) | 378 |
| 60, 4 | 24 | **1032** (= 43·24) | 744 |

So every downstream pixel dimension in the doc is under-stated by ≈40 %.

### Corrected data-shape summary table

Let `R` = rotated+trimmed side ≈ `√2·G` (measured `≈1.4·G`).

| Stage | Array | Correct shape |
|-------|-------|---------------|
| input | PIL image | `W₀ × H₀` |
| 2b | resized | `G × G` |
| 2b | rotated+trimmed | `R × R`, `R ≈ √2·G` (e.g. G=40 → 56) |
| 2c | diagonal grids | `(R/2+1, R/2)` and `(R/2, R/2+1)` |
| 3/4 | tiled + padded square | `(R/2+1)·t × (R/2+1)·t` |
| 5 | aspect-cropped | `(R/2+1)·t` on one axis, tile-rounded on the other |
| 7 | final RGBA | same as Stage 5 |
| 9 | app-fitted | exactly the upload's aspect ratio |

> Note: the Stage 2b "rotated+trimmed ≈ `(√2·G − 2)`" row is *almost* right but
> slightly low — PIL's `expand=True` rounds the bounding box **up** before the
> `−2` trim, so the actual `R` is a touch larger than `√2·G − 2` (54.6 predicted
> vs 56 actual for G=40). The *direction* (≈√2·G) is correct; it is only from
> Stage 2c onward that the doc drops the √2 and reverts to `G`.

The Stage 2c warning text ("the half-open `range(G//2 …)` bounds") has the same
slip: in code the bounds are `range(grid_size//2 …)` where `grid_size = R`.

**Impact:** the *mechanism* and the "fragile spot #1" warning are correct and
point at the right code; only the stated numbers are wrong. Anyone using the
data-shape table to predict output resolution or debug an off-by-one at the
edge will be misled by the √2 factor.

---

## Finding 2 (moderate): Stage 8 does not produce an animated GIF

PIPELINE.md Stage 8:

> Each frame is run through stages 1-7 independently and reassembled into an
> animated GIF.

[`generate_from_gif`](src/gol_mosaics/mosaic.py#L264) does process every frame,
but at the end it returns **only the first frame**
([mosaic.py:341-345](src/gol_mosaics/mosaic.py#L341-L345)):

```python
frames[0].info['duration'] = durations[0]
frames[0].info['loop'] = gif.info.get('loop', 0)
return frames[0]
```

`frames[1:]` are computed and then discarded; they are never attached via
`append_images=`. The returned object is a single still image, so the caller
cannot reconstruct the animation even with `save_all=True`. The function's own
`Returns: Animated PIL Image` docstring overstates this too. Either the doc
should say "returns the first frame only", or the function should return/attach
all frames.

(Also worth noting, though consistent with the doc: per-frame pattern selection
is independently random — `generate_from_gif` passes no `seed` and
`random_patterns` defaults to `True` — so a working animation would *flicker*
tile-to-tile between frames. And it uses different defaults from the still path:
`empty_tiles_cutoff=0.75` vs `0.65`, fixed `supersample=15`.)

---

## Finding 3 (minor nuance): a "filled" mask tile is a filled *diamond*, not a full square

PIPELINE.md Stage 4 says a transparent (background) cell becomes a "filled tile
(1)". In [`get_patterns_for_mask`](src/gol_mosaics/patterns.py#L529-L534) that
filled tile is `binary_fill_holes(solutions[-1])`, which measures at **~44 % ones**
for L=4 (a filled diamond with empty corners), not a solid `t×t` block of 1s.

This is not a bug: the two diagonal mask grids are padded and overlaid with the
same diamond geometry as the GoL tiles, so the filled diamonds **interlock to
cover the whole background region**, and the final
[`binary_fill_holes(mask)`](src/gol_mosaics/mosaic.py#L456) closes any residue.
The end result is exactly the "1 = background / 0 = subject" map the doc
describes (measured: 0.974 background for a small centred subject). The wording
"filled tile (1)" is just a simplification of "filled diamond that tiles to 1".

---

## Confirmed correct (the things that commonly go wrong — and don't here)

- **Stage 1 seed scope.** `np.random.seed(seed)` is at
  [mosaic.py:214-215](src/gol_mosaics/mosaic.py#L214-L215); the ECA *rule* is
  chosen earlier in `__init__` at [mosaic.py:86](src/gol_mosaics/mosaic.py#L86),
  so it is **not** governed by that reseed — exactly as the ⚠️ caveat warns. The
  app's work-around reseed is at [app.py:205](app.py#L205), immediately before
  `MosaicGenerator(...)`. Both verified.
- **Stage 3 `pad_size`.** `((6-3)·(2L-1)+1+2)//2` evaluates to `3L = t/2`
  (measured: 9 for L=3/t=18, 12 for L=4/t=24). `first` padded left/right,
  `second` padded top/bottom, then **added**; result is binary `{0,1}`. ✅
- **Stage 4 polarity.** `alpha ≥ alpha_cutoff → empty (0, subject)`,
  `alpha < alpha_cutoff → filled (1, background)`
  ([patterns.py:529-534](src/gol_mosaics/patterns.py#L529-L534)); then
  `binary_fill_holes` ([mosaic.py:456](src/gol_mosaics/mosaic.py#L456)). The
  "fragile spot #3" border caveat is a real property of `binary_fill_holes` and
  correctly described.
- **Stage 5 crop direction.** `aspect_ratio == 1.0` → no-op; `>1` crops height;
  `<1` crops width; new dimension `ceil(new/tile)*tile`, centred via
  `(total-new)//2`. Verified the ratio comes out *close but not exact* (0.75 →
  0.759), matching the "fragile spot #4" warning.
- **Stage 6 supersample.** `_auto_select_supersample` returns
  `max(1, min(target, mosaic_width))` with `target=15`
  ([mosaic.py:569](src/gol_mosaics/mosaic.py#L569), called with `target=15` at
  [mosaic.py:251](src/gol_mosaics/mosaic.py#L251)). No divisibility requirement —
  `generate()` crops, confirmed.
- **Stage 7 composite.** `eca_mask = transparency_mask * (eca_pattern +
  transparency_mask)` ([mosaic.py:540](src/gol_mosaics/mosaic.py#L540)) yields
  exactly `{0,1,2}` (measured). `render_gol_mosaic` is fully opaque,
  `render_eca_overlay` maps `0→transparent, 1→eca_background, 2→eca_pixel`
  ([renderer.py:86-134](src/gol_mosaics/renderer.py#L86-L134)), then
  `Image.alpha_composite`. Final image is **opaque everywhere** (alpha 255), as
  the doc states.
- **Stage 7 rim note.** Rotation corners are white in both greyscale and mask →
  empty GoL tile + high-alpha "subject" → rendered as opaque `gol_background`.
  Logic verified.
- **Stage 9 app post-fit.** `_fit_to_aspect`
  ([app.py:143](app.py#L143)) pads the shorter axis (never crops:
  `max(new, old)` at [app.py:160](app.py#L160)) with `eca_background` colour
  ([app.py:225](app.py#L225)) and pastes the opaque mosaic centred. ✅

## Line-citation audit

Every `file:line` reference in PIPELINE.md was checked and is **correct** on this
branch, including: `mosaic.py` 77, 86, 122, 163, 214-215, 264, 347, 378-386,
389-402, 406, 456, 460, 505, 540, 549; `image_processing.py` 40, 116, 141, 260,
295, 338, 342, 393, 436-438, 441-443; `patterns.py` 418, 490; `eca.py` 59,
103-120; `renderer.py` 48-168; `app.py` 143, 170, 205.

---

## Suggested edits to PIPELINE.md

1. **Stage 2c, Stage 3, and the data-shape table:** replace `G/2` with `R/2`
   (define `R ≈ √2·G` as the rotated+trimmed side) in the diagonal-grid shapes
   and the `(G/2+1)·t` final-side expressions. Use the corrected table above.
2. **Stage 2c warning:** change "`range(G//2 …)`" to "`range(R//2 …)` (the code
   reads `grid_size = lowres.shape[0]`, i.e. the *rotated* size)" — this actually
   *strengthens* the fragile-spot point.
3. **Stage 8:** either correct it to "returns the first frame only" or fix
   `generate_from_gif` to attach `append_images=frames[1:]`. As written, the doc
   and the function's docstring both promise an animation the code doesn't return.
4. *(Optional)* Stage 4: add a half-sentence that the "filled tile" is a filled
   diamond that interlocks to cover the background, not a solid square.
