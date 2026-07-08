# SAT tile search pipeline (levels 6, 7, ...)

Exhaustive enumeration of symmetric still-life mosaic tiles with the
open-source CaDiCaL SAT solver, replacing the licensed Gurobi ILP. The
method — one boolean per free D4 symmetry orbit, binomial still-life
clauses, AllSAT with blocking clauses, cube-and-conquer parallelism — is
documented and validated end to end in `notebooks/tile_generation_sat.ipynb`.

The core now lives in the package (`gol_mosaics.tile_domain`,
`gol_mosaics.sat_search`, rule-generalized to any Life-like CA); this
directory holds the production pipeline around it: parallel workers,
checkpointing, streamed merging, and the validation gate. `verify.py` is a
deliberately independent checker that shares no code with the encoding.
Run from a checkout of the repository (`git clone` / rsync the whole repo).

## Results so far

- **Level 6 is done**: all **332,321** tiles, enumerated in ~2 s on a
  10-core laptop, five-way validated, shipped with the package as
  `src/gol_mosaics/data/solutions_pattern_level_6_orbits.npy` (2.7 MB packed;
  `PatternLibrary.load(6)` expands it transparently).
- **Level 7 is ready to run**: dead edges are now *derived* from the mosaic
  geometry (the historical hand-made lists for levels 2–6 are reproduced
  exactly), 84 free orbits, measured ~40k tiles/s single-core on the first
  5,000 tiles. Expected ~10^8 tiles → hours on a multi-core machine.

## Usage

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Mandatory validation gate: re-derives levels 3-5 byte-exactly and
#    prints calibration timings. Nonzero exit = do not proceed.
python search.py validate

# 2a. Level 6 (minutes, ~430 MB grid output)
python search.py run --level 6

# 2b. Level 7 (hours; packed output, a few GB)
python search.py run --level 7 --cube-bits 16
```

Runs are safe under nohup/tmux: one checkpoint file per finished cube
(written atomically), so Ctrl-C and rerun resumes where it stopped. `run`
auto-merges when all cubes are complete; `merge` re-assembles from
checkpoints without re-solving.

### Options

| Flag | Default | Notes |
|---|---|---|
| `--workers N` | all cores | worker processes |
| `--cube-bits K` | 10 | 2^K independent subproblems; 14–16 for level 7 |
| `--work-dir DIR` | `work` | checkpoint location (`work/level_L/cubes/*.npy`) |
| `--solver NAME` | `cadical195` | any pysat solver name; `glucose42` as fallback |
| `--packed` | auto (level ≥ 7) | save packed orbit bits instead of full grids |
| `--output PATH` | `solutions_pattern_level_{L}[_orbits].npy` | final artifact |

Level-7 output stays as **packed orbit bits** (11 bytes/tile; raw 42×42
grids would be hundreds of GB). Expand in chunks with
`gol_mosaics.tile_domain.unpack_solutions(packed[i:j], level=7)`.

## Audit battery

```bash
python search.py validate                      # byte-exact levels 3-5 + calibration
python search.py validate --solver glucose42   # cross-solver determinism
python -m gol_mosaics.sat_search bruteforce --level 4   # SAT-free 2^22 check (~1 min)
python -m pytest tests/test_tiling.py tests/test_patterns.py  # from the repo root:
                                               # whole-mosaic stability, dead-edge
                                               # regression, packed round trip
```

## Integrity guarantees

- **Exhaustive**: each cube's solver runs to UNSAT; the cubes partition the
  assignment space exactly (no overlap, no gap), so the union is complete.
- **Exact**: the model has no auxiliary variables, so each blocking clause
  removes exactly one assignment — no duplicate or phantom solutions.
- **Verified**: `merge` re-checks every solution in chunks with the
  independent verifier (still-life rules, D4 symmetry, all forcings) and
  asserts global uniqueness before saving; any failure aborts.
- **Checkpoint safety**: cube files are written atomically; the manifest
  pins the CNF SHA-256 fingerprint (which includes the rule and the derived
  dead edges), so a checkpoint made with different code or parameters is
  refused instead of silently mixed.

## Files

| File | Role |
|---|---|
| `search.py` | CLI: `validate` \| `run` \| `merge` (parallelism, checkpoints, streaming) |
| `verify.py` | independent solution checker + reference comparison |
| `geometry.py`, `encoding.py` | thin shims re-exporting `gol_mosaics.tile_domain` / `.sat_search` |
| `pack_level6.py` | one-off: grids → packed orbit bits for the package data dir |
| `reference/` | optional local copy of the level 3–5 databases; when absent, `validate` reads the byte-identical files shipped in `src/gol_mosaics/data/` |
