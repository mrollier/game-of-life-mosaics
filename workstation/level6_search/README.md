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
- **Level 7 is done, twice**: **108,492,376** tiles (84 free orbits, 42×42
  grids), first with `search.py run --level 7 --cube-bits 16` on a 10-core
  laptop (~2.25 h of solving), then reproduced on a 36-thread Xeon
  workstation with the adaptive runner below in **36 min** of solving plus
  90 s of merging (75,000 CPU-s; 34,966 leaves, deepest at 44 fixed
  variables). The 1.19 GB packed artifact
  (`solutions_pattern_level_7_orbits.npy`, sha256
  `d39bbd0aa05be20ea7689f7f4a5e69e6c4da4a9cfb281d96d9cef671ea885b80`) is
  gitignored; `count.py` reads it as ground truth when present.
- **The counts are confirmed by #SAT model counting**: sharpSAT-td on the
  DIMACS export of `count.py` returns 7, 85, 2632, 332,321 and 108,492,376
  for levels 3–7 (level 7 in 117 s single-core, 0.18 GB) — a method that
  shares nothing with the enumeration — and extends the sequence with
  **level 8 = 172,693,540,438** (6.6 h single-core), see below.

## Usage

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Mandatory validation gate: re-derives levels 3-5 byte-exactly and
#    prints calibration timings. Nonzero exit = do not proceed.
python search.py validate

# 2a. Level 6 (minutes, ~430 MB grid output)
python search.py run --level 6

# 2b. Level 7 with the fixed-prefix cubes (hours; packed output, 1.2 GB)
python search.py run --level 7 --cube-bits 16

# 2c. Level 7 with adaptive cubes (~40 min on 36 threads; same output)
python search_adaptive.py self-test         # levels 5-6 through the adaptive path
python search_adaptive.py run --level 7 --nice 10 --expect 108492376 \
    --work-dir work/adaptive --output solutions_pattern_level_7_orbits.npy
```

### Why the adaptive runner

`search.py` fixes the first `--cube-bits` decision variables. Those are the
tile's apex orbits, which are so constrained that at level 7 only **65 of
65,536** cubes are non-empty, and (as at level 6, where 12 of 1,024 cubes
hold everything and the largest holds 18 %) a few cubes carry the census.
Two things then go wrong at once: parallelism is capped by the handful of
big cubes, and inside a big cube the AllSAT loop slows from ~30k to <6k
tiles/s as millions of 84-literal blocking clauses accumulate. On top of
that, `merge` runs the independent verifier serially, at ~3k tiles/s for
42×42 grids — about nine CPU-hours for 10^8 tiles.

`search_adaptive.py` keeps the encoding, the checker, the packed format and
the canonical order, but partitions adaptively: a node (the first *k*
variables fixed) is enumerated with a cap (default 50,000); if the cap is
reached the node is discarded and split into 2^`--split-bits` children on
the next variables. Leaves stay in the solver's fast regime, the tree
balances itself across workers, and every leaf is verified *inside* its
worker in 2,048-tile chunks (3× faster per tile than 64k chunks) before it
is written. The merge then only checks uniqueness, computes live-cell
counts from the orbit sizes (cross-checked against expanded grids on a
sample) and sorts. Capped-and-discarded probes cost about 20 % of the CPU;
the run is resumable from `nodes.jsonl` plus the atomically written leaf
files. `self-test` reproduces level 5 as a set and the shipped level-6
packed file byte-for-byte (including its order).

### Counting without enumerating

The CNF has no auxiliary variables, so its model count *is* the census.
`count.py` exports DIMACS and drives an exact model counter:

```bash
git clone https://github.com/Laakeri/sharpsat-td tools/sharpsat-td   # needs gmp + mpfr
(cd tools/sharpsat-td && ./setupdev.sh)
cd tools/sharpsat-td/bin        # sharpSAT execs ./flow_cutter_pace17 from its cwd
PYTHONPATH=../../../../../src:../../.. python ../../../count.py --level 7 \
    --counter ./sharpSAT --counter-args "-decot 1 -decow 100 -tmpdir ../../../counts -cs 3500" \
    --out-dir ../../../counts
```

Levels 3–6 count in seconds and level 7 in two minutes, all matching the
enumerated censuses. **Level 8 counts to 172,693,540,438 tiles**
(1.73 × 10^11; 114 free orbits, 16,196 clauses, 48×48 grids) in 6 h 35 min
on one core with a 3.5 GB component cache (6.2 GB peak RSS; the
log-quadratic extrapolation from levels 3–7 had predicted 1.2 × 10^11).
A larger cache does not help: with `-cs 12000` the process passed 20 GB
of RSS within two hours and was stopped. Level 8 is a counting problem
only — enumerating it would take weeks and ~2.5 TB of packed output
(15 bytes/tile) — so the census sequence now reads

    1, 2, 7, 85, 2632, 332321, 108492376, 172693540438   (levels 1–8)

with the last term known only as a count.

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
| `search_adaptive.py` | CLI: `self-test` \| `run` \| `merge` — adaptive cube-and-conquer with capped leaves and in-worker verification (level 7 in ~40 min on 36 threads) |
| `count.py` | DIMACS export + exact #SAT model counting (sharpSAT-td, ganak, …); confirms censuses without enumeration |
| `verify.py` | independent solution checker + reference comparison |
| `certify.py` | emits a DRAT proof that CNF + one blocking clause per census tile is UNSAT (completeness certificate); checks it with drat-trim or the bundled checker |
| `rup_check.py` | self-contained forward RUP/DRAT proof checker (no shared code; for small proofs — use drat-trim at level-6 scale) |
| `requirements-lock.txt` | exact package versions used for the paper's measurements |
| `geometry.py`, `encoding.py` | thin shims re-exporting `gol_mosaics.tile_domain` / `.sat_search` |
| `pack_level6.py` | one-off: grids → packed orbit bits for the package data dir |
| `reference/` | optional local copy of the level 3–5 databases; when absent, `validate` reads the byte-identical files shipped in `src/gol_mosaics/data/` |
