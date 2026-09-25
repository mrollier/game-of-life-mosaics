# Reproducing the paper's numbers

Every table and figure of the preprint *"Exhaustive enumeration of
symmetric still-life tiles in Life-like cellular automata via
orbit-reduced SAT solving"* regenerates from this repository. Commands
run from the repository root; `PY` is a Python 3.13 with the packages
pinned in `search/tiles/requirements-lock.txt`
(measurements in the paper: Apple M4, 10 cores, 16 GB, macOS 26.5).

```bash
PY="python"          # with numpy, scipy, python-sat, matplotlib
export PYTHONPATH=src
```

## Validation battery (run first)

```bash
(cd search/tiles && $PY search.py validate)     # byte-exact levels 3-5
(cd search/tiles && $PY search.py validate --solver glucose42)
$PY -m gol_mosaics.sat_search bruteforce --level 4            # SAT-free 2^22 check
$PY -m pytest tests/                                          # incl. whole-mosaic stability
```

## Tables

| Paper item | Command |
|---|---|
| Table 1 (census, CNF sizes, single-thread times) | `$PY -c "from gol_mosaics.sat_search import build_cnf, enumerate_all; ..."` — or run `notebooks/sat_tile_search.ipynb` end to end; timings: 3 runs of `enumerate_all(build_cnf(L))`, median |
| Table 2 (orbit accounting) | evidence script `paper/ars/revision_data/` (archived with the release); equivalently `build_domain(L)` and count `constants` values |
| Table 3 (hypothesis (H)) | `tests/test_tiling.py` harness; frame-only tile = `build_domain(L).expand(zeros)`; assembled check via `MosaicGenerator._build_mosaic` with a single-tile library |
| Table 4 (level-6 benchmark) | `cd search/tiles && /usr/bin/time -l $PY search.py run --level 6 --work-dir /tmp/bench --output /tmp/bench_level6.npy` (phase timings are printed by `run`/`merge`) |
| Table 5 (certificates) | `cd search/tiles && $PY certify.py --level L` for L = 3,4,5,6 (levels 3-5 auto-check with the bundled `rup_check.py`) |
| Table 6 (population statistics) | populations = `PatternLibrary.load(L).solutions.sum((1,2))`; percentiles with lower interpolation |
| Appendix (interlock representatives) | `$PY -c "from gol_mosaics.tile_domain import derive_dead_edges; print(derive_dead_edges(L))"` |

## Figures

```bash
$PY paper/figures/make_figures.py        # writes all six PDFs from live library data
```

## Completeness certificates (level 6)

The level-6 DRAT proof (102.9 MB, 501,673 lemmas) is produced by
`certify.py --level 6` in ~100 s and exceeds the bundled pure-Python
checker's practical range. Check it with drat-trim:

```bash
git clone https://github.com/marijnheule/drat-trim && (cd drat-trim && make)
cd search/tiles
PYTHONPATH=../../src $PY certify.py --level 6 --checker ../../drat-trim/drat-trim
```

## SHA-256 hashes (full)

Census files (`src/gol_mosaics/data/`):

```
1632bb7eed5f3bce408dd3f56691b795cfc66b0dcf9110c7ed825a5e00c1ce4e  tiles_diamond_level_1_orbits.npy
664fc98f5a221f12867278191c0f9fcb02ce9905ed8383d9b40b2e1dd789baca  tiles_diamond_level_2_orbits.npy
1010f67f98d28f5bcebd78cf6c56ef072131cf3e558761d26cc587fed117a8ab  tiles_diamond_level_3_orbits.npy
e3bc142940cdd79a69d341b069604207b0de4c9a73fc6ce74ae7169d660bb17f  tiles_diamond_level_4_orbits.npy
6ac415de1399bc61d5cea763b9a8748fb9cebf53ec826a3472785dcd33b0b52d  tiles_diamond_level_5_orbits.npy
9939a9b973ebc842f373d806afa49a05c848b5e245b18776812c0c4e7d28c96b  tiles_diamond_level_6_orbits.npy
0c890554a74f7819806e57b11c291b78ee5a144a75e61c770469cac9ee5ae747  tiles_diamond_nosym_level_3_cells.npy
415d80cede172a6b5e3e88490f22b7cf06a07c20757f22c361a61eefebb2db4a  tiles_square_level_3_orbits.npy
d90c110eb2799f53d87fa4d3aa8e459008278312752808c61de513c10d274a9a  tiles_square_level_4_orbits.npy
200eff56d16848d22e7b2250b7d7382496a2a2d06603f969ccf38d0b88e222b9  tiles_square_level_5_orbits.npy
```

Every file is packed orbit bits (format below) and `python tools/pack_tiles.py
check` verifies all of them. Up to version 2.3 levels 1-5 shipped as full
grids (`solutions_pattern_level_{1..5}.npy`, sha256 `2953c14b...`,
`62e904cd...`, `21383c0c...`, `1f1d8a03...`, `ba526fe2...`); the packed
files expand to exactly the same tiles in the same order, only as uint8.

```
```

CNF fingerprints (SHA-256 over level, rule, and every clause; printed by
`build_cnf(L).sha256`):

```
d4a1e80102921a3f379619582a576421692fdefb19ee12a5b6d360499709fbde  level 3
83120d119db93fbdaeb62f982ae6e6ceaf905ce18849776444a375b5abd3b665  level 4
afe3924af9054d3f73b1cccc0d74292502deb441fc5294a1cd9e6feb10d4551d  level 5
926a451cb5800b1e50316474b9849ef77830a8eb5d2806a727f82a72eb59895c  level 6
```

Certificate proofs are deterministic given solver version and are
re-derived by `certify.py`, which prints their SHA-256 on each run.

## Packed orbit-bit format

`tiles_diamond_level_6_orbits.npy` is a `(332321, 8) uint8` array:
row `k` holds the 59 free-orbit bits of tile `k` packed
most-significant-bit-first (`numpy.packbits` semantics), rows in the
canonical order (live-cell count, then raw-grid byte order). Levels 1-5 use
the same format but keep the order in which the original ILP found them. Expand with
`gol_mosaics.tile_domain.unpack_solutions(packed, level=6)`; the free
orbits are enumerated by `build_domain(6).free_reps` (lexicographic by
representative).

## No-symmetry census (`_cells` files)

The `nosym` variant drops the D4 symmetry requirement while keeping the
frame, dead-edge, and stability constraints (module
`gol_mosaics.nosym_tiles`; notebook
`notebooks/tile_nosym_enumeration.ipynb`). Censuses: 1, 2, 1061 raw
grids at levels 1-3, forming 1, 2, 181 D4 equivalence classes (level-3
class sizes {1: 7, 2: 9, 4: 71, 8: 94}; the singletons are exactly the
symmetric census, and sum|Fix| = 1448 = 8 x 181 by Burnside).

`tiles_diamond_nosym_level_3_cells.npy` is a `(1061, 9) uint8`
array: row `k` holds the 68 free-CELL bits of tile `k` (identity
domain — one bit per unforced cell, no orbit reduction) packed
MSB-first, rows in the same canonical order as above. Expand with
`gol_mosaics.nosym_tiles.unpack_nosym_solutions(packed, level=3)`;
regenerate with `$PY tools/pack_tiles.py nosym --level 3`. Levels 1-2 ship
no file (their nosym censuses equal the symmetric ones and re-derive in
milliseconds).

The level-4 nosym census (156 free cells) has not been computed: a
single-shot AllSAT probe passed 2 x 10^6 models without exhausting the
space, so the count exceeds two million and needs the cube-and-conquer
treatment of the symmetric level 7 (hours of compute, artifact too
large for git). The runner is ready:
`search/nosym/search_nosym.py run --level 4 --cube-bits 16`
(resumable; `self-test` reproduces the level-3 census through the cube
path byte-exactly). Use fine cubes — with `--cube-bits 12` a handful of
tail cubes hold nearly the whole census and the per-cube blocking-clause
loop degrades badly.
