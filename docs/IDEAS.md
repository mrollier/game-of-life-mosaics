# Ideas

A brainstormed list of concrete creative ideas that are within reach for this
project, grounded in machinery the codebase already has: the `tile_scheme`
abstraction, SAT enumeration for arbitrary Life-like rules
(`sat_search.build_cnf(level, birth=..., survival=...)`), packed-orbit storage,
the geometry-agnostic density selector, `life.is_still_life`, `GollyExporter`
(with optional glider), the ECA renderer, and the Gradio app.

Bugs and technical debt live in `notebooks/open_problems.ipynb`; ideas that
overlap with its "still open" list are cross-referenced rather than duplicated.

Effort tags: **(S)** an afternoon, **(M)** a few days, **(L)** a project.

## A. New tile mathematics

The `tile_scheme` machinery makes new tile families cheap to try: a new scheme
is one constructor (support, frame, lattice basis) plus the existing interlock
derivation, hypothesis check and enumeration.

1. **Oscillating mosaics (period-2 tiles)** — enumerate tiles that are period-2
   *oscillators* instead of still lifes: SAT constraints `step(A) = B`,
   `step(B) = A`, `A ≠ B` over the same cell domain. The payoff is a portrait
   that visibly blinks when run in Golly — mosaics that are alive rather than
   merely stable. (M)

2. **HighLife and other Life-like rules, end to end** — `sat_search.build_cnf`
   already takes `birth=`/`survival=` (demonstrated for HighLife); enumerate
   B36/S23 tile sets for levels 3–5, ship them as packed data like the square
   sets, and add a rule selector to the library and app. Cross-ref the
   "Non-Conway Mosaics end to end" open problem. (M)

3. **Rectangular pond-frame tiles (n × m)** — relax the currently hard-assumed
   D4 tile symmetry to D2 in `TileScheme`. Gives landscape/portrait tile
   aspect ratios and a third integer census sequence. (M)

4. **Brick-lattice scheme** — offset-row ("running bond") square tiles; needs
   the lattice symmetry-group parameter that D4 currently hides. Visually a
   distinct masonry look. (L)

5. **OEIS submissions** — submit both censuses: the diamond sequence
   7, 85, 2632, 332321, 108492376 and the pond-frame-square sequence
   1, 3, 65, 10398, 19287185 (neither is in OEIS as of July 2026). Pure
   paperwork, and a permanent citation anchor for the paper. (S)

6. **Dead-edge minimality study** — the derived interlock cells are provably
   *sufficient*; which ones can be freed for *some* tile pairings? One SAT
   query per candidate cell. Cross-ref the "Necessity of the dead edges" open
   problem. (M)

7. **Mixed-level mosaics (quadtree detail)** — large tiles in flat regions,
   small tiles where the image has detail. The interlock geometry across level
   boundaries makes this the one research-grade item in this section. (L)

## B. Rendering and artistic output

8. **QR codes made of still lifes** — QR codes are square grids, and the new
   square scheme maps one module to one tile: dark module → dense tile, light
   module → hole (holes are verified stable). A scannable QR code that is a
   global still life. Needs only an adapter from a `qrcode` module array to an
   index grid fed to `tile_scheme.assemble`. (S/M)

9. **Floyd–Steinberg dithering over tile densities** — replace per-tile
   nearest-density matching with error diffusion across neighbouring tiles.
   Markedly smoother tonal reproduction at small grid sizes, as a drop-in
   variant of `PatternLibrary.get_indices_for_values`. (S)

10. **Tinted tiles ("stained glass")** — colour each tile's live cells from
    the source image's local average colour instead of one global scheme; the
    renderer already colours per cell, so this is a per-tile colour map rather
    than a new renderer. (M)

11. **Warhol canvas mode in the app** — render the same mosaic in a 2 × 2 grid
    of different `ColorScheme.warhol` palettes (the Marilyn-diptych idea).
    Reusing one mosaic sidesteps the "Canvas pixels" open problem entirely. (S)

12. **Glider-crash GIF** — export N generations of the mosaic with a glider
    launched from a corner (`GollyExporter` already places gliders): the
    portrait is provably stable until the glider arrives, then dissolves into
    chaos. A numpy Life stepper plus the existing GIF path. (M)

13. **Vector export (SVG/PDF)** — cells as rectangles for large-format
    printing and plotting; also the gateway to laser-cut stencils and
    cross-stitch patterns. (M)

14. **"Verified still life ✓" badge in the app** — run `is_still_life` on the
    generated cell grid (the tests already do) and show the verification in
    the UI. A cheap trust-and-delight feature. (S)

15. **Mosaic-of-mosaics (photomosaic)** — tile a large image using many small
    mosaics as super-pixels. (L)

## C. Web app features

16. **Reproducible seeds** — surface the RNG seed behind "🎲 New variation"
    and accept one as input, so a particular look can be shared and
    reproduced. (S)

17. **Before/after comparison slider** — a Gradio `ImageSlider` between the
    input preview and the finished mosaic. (S)

18. **Square level-6 sampling** — the 19,287,185-tile set cannot ship
    expanded, but density-targeted *sampling* via the cube-and-conquer
    pipeline could serve random level-6 square tiles on demand. (L)

19. **Level-7 diamonds via on-demand download** — host the 1.19 GB packed
    artifact on Hugging Face datasets and lazy-fetch it behind a flag; this
    doubles as the off-site backup the artifact still needs. (M)

## D. Outreach and publication

20. **Second paper: tile-scheme generalisation and the square census** —
    theorem-level content already exists: the straight-versus-diagonal pond
    chain instability, hypothesis (H), and two integer censuses. (L)

21. **Interactive explainer page** — a self-contained page walking from a
    single pond to a full mosaic; much of the material already exists in
    `notebooks/tile_scheme_generalisation.ipynb`. (M)

22. **Open-problems refresh** — fold this ideas file and the resolved
    square-integration items back into `notebooks/open_problems.ipynb`. (S)

## Also worth fixing while nearby

Not new ideas, but open problems adjacent to the work above (see
`notebooks/open_problems.ipynb`): the outer-rim rendering glitch, surprising
`grid_size` crops, and Pond/Tile/Mosaic naming consistency.
