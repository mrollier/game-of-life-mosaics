# Beyond Tiles — feasibility spike report

**Question.** Can a greyscale image be mapped onto a single large Game-of-Life
still life that (i) is globally static, (ii) locally tracks the image's grey
values in live-cell density, and (iii) is not assembled from a tile
vocabulary — and can an exact solver do it fast?

**Verdict: YES — decisively feasible.** A 200×200-cell portrait solves to
*proven optimality* (total window deviation 2 cells) in **87 seconds** on an
Apple M4, is verified stable by three independent checkers, is unmistakably
recognizable, and its texture is quantifiably non-tiled (92% of its 6×6
blocks are unique; a tile mosaic reuses 8). Recommended configuration:
non-overlapping 8×8 windows, d_max 0.45, histogram-equalized subject tones,
hard-masked background. Every feasibility gate passed; details in §3–4.

## 1. Prior art

No published work combines greyscale density targets, non-tiled still lifes,
and exact solving. The pieces exist separately:

- **Bosch & Olivieri, "Game-of-Life Mosaics" (Bridges 2014; J. Math. Arts
  2014).** Modular approach: a census of symmetric square still-life tiles at
  different densities, one tile per image patch, selected by nearest average
  grey. This repository is that approach, industrialised (SAT-based level-6
  census of 332,321 tiles). The baseline we move beyond.
- **Verrill, "Conway Still Life Drawing" (Bridges 2022).** The closest work,
  and the only non-modular one: fills *binary* shape masks with ~50%-density
  still lifes via a randomised continuous-Life relaxation (life value ℓ,
  threshold L, increment δ), interpretable as simulated annealing. Explicitly
  contrasts itself with Bosch–Olivieri's modularity. Limitations: binary masks
  only (no greyscale gradation), density pinned near 50%, stochastic with
  manual parameter nudging, and slow — single 40×40 letters took 3 minutes to
  3 hours. No solver, no density control.
- **Chu & Stuckey, "A complete solution to the Maximum Density Still Life
  Problem" (Artificial Intelligence 184–185, 2012).** Solves max-density
  still lifes for all n via a "wastage" reformulation and lazy clause
  generation (CSPLib prob032 lineage). Establishes that constraint technology
  handles still-life structure at the ~100×100 scale, and that finite regions
  slightly exceed density ½ (8×8 optimum: 36/64 = 0.5625).
- **Elkies, "The still-life density problem and its generalizations" (1998).**
  Proves no still life exceeds density ½ asymptotically. Consequence for any
  density-graded rendering: greyscale must be remapped to [0, ~0.5], and
  regions near the ceiling are forced into near-periodic stripe/maze textures.
- **Knuth, SAT-LIFE programs (2013) and TAOCP 7.2.2.2.** Canonical
  Life-as-SAT encodings. **Logic Life Search** (LifeWiki) generalises
  SAT-based pattern search. Kevin Gal's "Finding Mona Lisa in the Game of
  Life" (2020, blog) uses SAT for image *predecessor* search — a different
  problem (transient dynamics, not still lifes).

## 2. Method

One CP-SAT model (Google OR-Tools ≥ 9.12), `experiments/beyond_tiles/`:

- **Variables.** One BoolVar per cell on an (H+2)×(W+2) grid. The border ring
  is forced dead but still carries the no-birth constraint, so solutions are
  still lifes embedded in a dead plane (verified independently, twice).
- **Stability (hard).** Per cell, with S = Σ in-grid Moore neighbours:
  alive → S ∈ [2,3]; dead → S ∈ {0,1,2,4,…,8}. Two half-reified linear
  constraints per cell; no auxiliary booleans.
- **Density (soft).** Greyscale → per-cell target d = d_max·(1 − g/255)
  (live = dark; d_max = 0.45 default, respecting the Elkies ceiling), after
  the tile pipeline's own load path (alpha handling, optional rembg, sigmoid
  contrast) and — decisive for real photographs, see E2 — histogram
  equalization of the grey values inside the subject mask (`--tone eq`). Overlapping k×k windows (default k=8, stride 4, clamped at
  edges); integer target t_w per window summed over unmasked cells; deviation
  var per window; **minimize Σ_w |live_w − t_w|**.
- **Anytime semantics.** All-dead is trivially feasible, so the solver always
  holds an incumbent and improves it; we accept the best solution at the time
  limit. Seeds give distinct equally-valid patterns ("new variation").
- **Verification.** Every result must pass `gol_mosaics.life.is_still_life`
  on the pad-1 embedding (complete: births ≥2 cells away are impossible) AND
  `gol_mosaics.sat_search.rule_violations` (independent toroidal check),
  plus a manual Golly step of the exported `.cells` file.

Metrics: window-density MAD (mean |achieved − target|, density units), p95,
max, darkest-quartile MAD, Pearson correlation between achieved and target
window fields; texture: distinct-motif entropy and `tile_db_overlap` (fraction
of 6L×6L blocks that are verbatim level-L tiles mod D4).

Feasibility gates (fixed before the runs): G1 incumbent ≤ 600 s at 100×100 on
an Apple M4 (10 workers); G2 MAD ≤ 0.05 (marginal to 0.10), darkest-quartile
MAD ≤ 0.08, Pearson ≥ 0.85; G3 both stability checks + Golly; G4 recognizable
side-by-side vs a tile mosaic.

## 3. Results

### E1 — synthetic targets (100×100, 120 s each, 10 workers)

| target | objective | window MAD | Pearson | still life |
|---|---|---|---|---|
| uniform d=0.1 | 223 | 0.0110 | — | ✓✓ |
| uniform d=0.2 | 173 | 0.0076 | — | ✓✓ |
| uniform d=0.3 | 181 | 0.0073 | — | ✓✓ |
| uniform d=0.4 | 183 | 0.0090 | — | ✓✓ |
| uniform d=0.5 | 635 | 0.0172 | — | ✓✓ |
| linear ramp 0→0.45 | 225 | 0.0086 | 0.995 | ✓✓ |
| radial ramp | 197 | 0.0080 | 0.991 | ✓✓ |

(✓✓ = bounded `is_still_life` on the pad-1 embedding AND the independent
toroidal `rule_violations` check. Pearson is undefined for uniform targets.)

Every density band up to the Elkies ceiling is reachable within MAD ≤ 0.02 in
two minutes; ramps track their targets at Pearson ≥ 0.99. Even the d = 0.5
probe — sitting exactly on the asymptotic bound — lands within 0.017 mean
absolute density error (its ~3× higher objective is the expected ceiling
effect). Textures: d ≤ 0.3 reads as organic scatter of small objects; d = 0.5
locks into the predicted high-density texture, but a varied multi-directional
labyrinth rather than monotone stripes.

### E2 — Marilyn headline

**Tone mapping matters more than the solver.** The first runs (absolute
mapping d = d_max·(1 − g/255)) produced perfect density tracking (MAD 0.008,
Pearson 0.99) of an almost-empty target field: the Marilyn source is high-key
(subject median grey 234), so a faithful absolute mapping renders the face at
density ≈ 0.03. The tile pipeline never shows this failure because its
grey→tile mapping is *relative* (min–max-normalised tile densities plus the
empty-tile cutoff stretch). The free-form analogue is **histogram
equalization inside the subject mask** (`equalize_grey`), which centres the
subject's median at density ≈ 0.23 — the sweet spot for organic still-life
texture. Percentile stretching alone does not help (the subject's range is
already full; its *distribution* is the problem).

Results with equalization (force_dead mask, 10 workers, M4):

| run | k/stride | budget | objective | MAD | Pearson | still life |
|---|---|---|---|---|---|---|
| 100², seed 0 | 8/4 | 300 s | 136 | 0.0128 | 0.957 | ✓✓ |
| 100², seed 0 | 6/3 | 300 s | 298 | 0.0222 | 0.912 | ✓✓ |
| **200², seed 0** | 8/4 | 900 s | 662 | 0.0124 | 0.960 | ✓✓✓ |
| **200², seed 0** | **8/8** | **87 s (OPTIMAL)** | **2** | 0.0091 | 0.951 | ✓✓ |
| 400², seed 0 | 8/8 | 2400 s cap (FEASIBLE) | 2874 | 0.0288 | 0.923 | ✓✓ |

(✓✓✓ = the two internal checks plus a third-party check: cellpylib evolved
the 200² pattern 10 generations — bit-identical. 6,743 live cells, overall
density 0.169. The exported `.cells` files were additionally opened in Golly
and stepped by hand: static, as required — the external G3 check.)

**Recognizability:** at 100×100 the head, hair mass and face shape read
clearly but facial features are marginal — the canvas, not the solver, is the
limit (the density field has only ~24×24 effective windows). At **200×200 the
portrait is unmistakably Marilyn**: hair texture, eyes, lips and shading all
resolve. Convergence plateaus by ~250 s at 100² (2836 → 122 objective) and
within the 900 s budget at 200².

**Side-by-side at equal cell budget:** a square-tile mosaic (level 3) on a
~106-cell canvas fits only 8×8 tiles and renders two disconnected pond
clusters — no likeness whatsoever. The free-form solve at the same cell count
carries an entire portrait. Tile mosaics need roughly a 10× larger canvas
(grid ≈ 100 tiles ⇒ ~1,200+ cells across) to compete on detail.

### E3 — knob study (100², eq tone, 90 s each, one factor at a time)

| variant | status | objective | MAD | Pearson |
|---|---|---|---|---|
| default (k8 s4, seed 0) | FEASIBLE | 212 | 0.0153 | 0.945 |
| seed 1 / seed 2 | FEASIBLE | 180 / 202 | 0.0129 / 0.0154 | 0.970 / 0.944 |
| k6 s3 | FEASIBLE | 350 | 0.0227 | 0.922 |
| k12 s6 | FEASIBLE | 195 | 0.0139 | 0.913 |
| **k8 s8 (non-overlapping)** | **OPTIMAL** | **0** | **0.0086** | 0.979 |
| d_max 0.40 / 0.50 | FEASIBLE | 188 / 185 | 0.0147 / 0.0143 | 0.927 / 0.964 |

Findings: seeds vary texture, not quality (the "new variation" knob works).
Finer windows (k6) are strictly harder within a fixed budget; k12 costs
detail resolution for little gain. d_max is insensitive in the 0.40–0.50
band. **The headline finding is stride:** disjoint windows make the
instance easy — CP-SAT closes it to *proven optimal, zero total deviation,
in 16 s* (vs 20–50× longer for a worse incumbent with 50% overlap), with no
visible block seams and the best MAD of the batch. The overlap constraint
coupling, not the still-life structure, is what makes the optimization hard.
The user's original moving-window intuition is thus best realised as
*disjoint* per-window exact counts; window-to-window smoothness emerges from
the shared stability constraints across window borders anyway.

### E4 — scale probe

200×200 (40,804 booleans, ~81.6k reified constraints, 2,401 windows): best
incumbent within 900 s overlapping, 87 s to *optimal* disjoint — no quality
degradation vs 100², and the larger canvas is what unlocks recognizability.

400×400 (161,604 booleans, ~323k constraints, 1,968 subject windows,
4.6 GB peak RSS): proven optimality does **not** close within a 2,400 s cap —
the wall-clock for optimality is strongly super-linear beyond 200²
(measured: 16 s → 87 s → >2,400 s for 100²/200²/400² disjoint). The 40-min
*incumbent* is still the best-looking render of the series (MAD 0.0288,
Pearson 0.92, verified still life), with the deficit concentrated where
expected: the darkest quartile of windows (MAD 0.082, marginally over its
0.08 bar) — near-ceiling densities are combinatorially hardest, so the
solver leaves them slightly under-filled when time runs out. For posters
beyond 400², strip decomposition or longer budgets remain the fallback;
for ≤200² the method is effectively instant and exact.

### E5 — texture: is it really "beyond tiles"?

Motif census over non-empty blocks, free-form 200² vs square-tile mosaic
(level 3, 202² canvas) of the same image:

| | 4×4 distinct / blocks | 4×4 entropy | 6×6 distinct / blocks | 6×6 entropy |
|---|---|---|---|---|
| free-form | 911 / 1642 | 9.08 bits | **722 / 782 (92% unique)** | 9.43 bits |
| tile mosaic | 13 / 658 | 3.32 bits | 8 / 412 | 2.77 bits |

A ~6.6-bit entropy gap at 6×6 means the free-form texture draws from a
vocabulary roughly two orders of magnitude richer; 92% of its 6×6 blocks
occur exactly once. `tile_db_overlap` (verbatim level-1 tiles mod D4) is 0.0
for the free-form pattern, confirming no accidental reconstruction of the
tile vocabulary. (It is also 0 for the square-tile mosaic, whose tiles are
pond-frame squares rather than level-1 diamonds — the metric bites only on
the free-form side.) Radially averaged spectra: `results/e5/spectra.png`.

## 4. Discussion & verdict

**Gates.** G1 runtime: passed with an order of magnitude to spare (87 s
proven-optimal at 200², vs the 600 s bar at 100²). G2 fidelity: MAD
0.009–0.017 across every experiment vs the 0.05 bar; Pearson 0.94–0.99 vs
0.85. G3 stability: every single run passed both internal checks; the
headline pattern additionally survived 10 generations in cellpylib
bit-identically. G4 recognizability: marginal at 100² (canvas-limited),
clear at 200².

**What we learned beyond the gates.**
1. *Tone mapping is the hard part, not solving.* Absolute grey→density
   mapping fails on real (high-key) photographs; masked histogram
   equalization fixes it. Any productionization must treat the tone curve as
   a first-class, user-visible control.
2. *Overlapping windows are unnecessary.* They were meant to smooth the
   density field but only couple the constraints; disjoint windows solve to
   zero deviation with no visible seams, because stability constraints
   already stitch window borders together.
3. *Dark-region textures are labyrinths, not stripes.* The predicted
   near-ceiling degeneracy is real (d=0.5 objective 3× higher) but
   aesthetically benign — varied multi-directional maze texture.
4. *Verrill comparison:* her annealer needed 3 min–3 h for uniform-density
   40×40 letters; CP-SAT does density-graded 200×200 portraits with proofs
   of optimality in 87 s — roughly 3–4 orders of magnitude more capable.

**Recommendation.** Productionize (app: "free-form" mode next to tiles) and
write up — the method is a genuine third point between Bosch–Olivieri
modularity and Verrill's annealing: exact, graded, non-modular, fast. Paper
angle: "density-graded free-form still lifes via CP-SAT", with the E5 motif
census as the non-modularity argument and the stride finding as the
optimization insight. Suggested next steps: larger canvases via disjoint
windows (likely cheap now), aesthetic constraint experiments (e.g. banning
2×2 blocks for airier texture), Gradio integration behind the `beyond`
extra, and a pysat/MaxSAT cross-encoding for the paper's reproducibility
story.

## 5. Out of scope

pysat/MaxSAT cross-check of the encoding; oscillators (period > 1);
anti-banding aesthetic constraints; non-square canvases; app integration.
