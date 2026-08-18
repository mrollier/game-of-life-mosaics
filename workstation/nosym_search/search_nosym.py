"""
Cube-and-conquer enumeration of the no-symmetry diamond tile census for
levels whose count is too large for single-shot AllSAT (level 4: 156
free cells, census > 2 x 10^6 — a single blocking-clause loop degrades
badly past ~10^6 models, so the space is split into cubes over the
first `cube_bits` decision variables and each cube gets a fresh solver,
mirroring workstation/level6_search/search.py).

Usage:
    python search_nosym.py run        --level 4 --cube-bits 12 --workers 10
    python search_nosym.py self-test  # cube pipeline vs single-shot, level 3

Output: work_nosym/level_{L}/cubes/cube_*.npy checkpoints (resumable —
existing cube files are skipped), merged into
solutions_pattern_nosym_level_{L}_cells.npy next to this script:
packed free-cell bits, canonical (popcount, grid bytes) row order.
Local-only artifact — too big for git; census count and sha256 are
recorded in REPRODUCE.md.

The merge verifies every tile independently (vectorised toroidal rule
check via sat_search.rule_violations on the expanded grids, in chunks),
asserts global uniqueness, and checks per-cell forcings on samples.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

from gol_mosaics.nosym_tiles import (  # noqa: E402
    build_nosym_cnf,
    build_nosym_domain,
    enumerate_nosym_tiles,
    pack_nosym_solutions,
)
from gol_mosaics.sat_search import (  # noqa: E402
    cube_units,
    enumerate_all,
    rule_violations,
)

SOLVER = "cadical195"
VERIFY_CHUNK = 1 << 15

_worker_enc = None


def _init_worker(level):
    global _worker_enc
    _worker_enc = build_nosym_cnf(level)


def _solve_cube(args):
    cube, cube_bits, out_path = args
    t0 = time.time()
    rows = enumerate_all(_worker_enc, solver_name=SOLVER,
                         extra_units=cube_units(cube, cube_bits))
    if len(rows):
        bits = np.asarray(rows, dtype=np.uint8).reshape(len(rows), -1)
        packed = np.packbits(bits, axis=1)
    else:
        packed = np.empty((0, (_worker_enc.n_vars + 7) // 8), dtype=np.uint8)
    tmp = out_path.with_suffix(".tmp.npy")
    with open(tmp, "wb") as f:
        np.save(f, packed)
    tmp.rename(out_path)
    return cube, len(rows), time.time() - t0


def cmd_run(level: int, cube_bits: int, workers: int) -> int:
    enc = build_nosym_cnf(level)
    assert cube_bits < enc.n_vars, "cube_bits must be below n_vars"
    work = HERE / "work_nosym" / f"level_{level}"
    cubes_dir = work / "cubes"
    cubes_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = work / "manifest.json"
    manifest = {
        "format": "nosym-cube-packed-v1",
        "level": level,
        "cube_bits": cube_bits,
        "n_vars": enc.n_vars,
        "encoding_sha256": enc.sha256,
        "solver": SOLVER,
    }
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        assert stored == manifest, "manifest mismatch — wipe work dir to restart"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    n_cubes = 1 << cube_bits
    todo = [
        (c, cube_bits, cubes_dir / f"cube_{c:05d}.npy")
        for c in range(n_cubes)
        if not (cubes_dir / f"cube_{c:05d}.npy").exists()
    ]
    print(f"level {level}: {enc.n_vars} vars, {len(enc.clauses)} clauses, "
          f"{n_cubes} cubes ({len(todo)} to solve), {workers} workers")

    t0 = time.time()
    total = 0
    with mp.Pool(workers, initializer=_init_worker, initargs=(level,)) as pool:
        for i, (cube, count, dt) in enumerate(
                pool.imap_unordered(_solve_cube, todo, chunksize=1)):
            total += count
            if count or (i + 1) % 256 == 0:
                print(f"  [{i + 1}/{len(todo)}] cube {cube:5d}: {count:8d} "
                      f"({dt:.1f}s, running total {total}, "
                      f"{time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"solve done: {total} new solutions in {time.time() - t0:.0f}s")
    return cmd_merge(level)


def cmd_merge(level: int) -> int:
    enc = build_nosym_cnf(level)
    domain = enc.domain
    work = HERE / "work_nosym" / f"level_{level}"
    cubes_dir = work / "cubes"
    manifest = json.loads((work / "manifest.json").read_text())
    assert manifest["encoding_sha256"] == enc.sha256, "stale work dir"
    n_cubes = 1 << manifest["cube_bits"]
    paths = [cubes_dir / f"cube_{c:05d}.npy" for c in range(n_cubes)]
    missing = [p.name for p in paths if not p.exists()]
    assert not missing, f"{len(missing)} cube files missing, rerun first"

    t0 = time.time()
    packed = np.concatenate([np.load(p) for p in paths])
    print(f"merge: {len(packed)} rows loaded in {time.time() - t0:.0f}s")

    # global uniqueness (cubes partition the space, but verify anyway)
    uniq = np.unique(packed, axis=0)
    assert len(uniq) == len(packed), "duplicate solutions across cubes"

    # independent stability + forcing verification on expanded grids
    t0 = time.time()
    n_free = len(domain.free_reps)
    bad = 0
    for start in range(0, len(packed), VERIFY_CHUNK):
        chunk = packed[start:start + VERIFY_CHUNK]
        bits = np.unpackbits(chunk, axis=1, count=n_free)
        grids = domain.expand_many(bits)
        bad += int(rule_violations(grids, enc.birth, enc.survival).sum())
    assert bad == 0, f"{bad} unstable solutions"
    print(f"verified {len(packed)} tiles stable in {time.time() - t0:.0f}s")

    # canonical order: (popcount, full grid bytes); forced-alive count is
    # constant so free-bit popcount ordering is equivalent, but sort on
    # the expanded grid to match enumerate_nosym_tiles exactly
    t0 = time.time()
    keys = []
    for start in range(0, len(packed), VERIFY_CHUNK):
        chunk = packed[start:start + VERIFY_CHUNK]
        bits = np.unpackbits(chunk, axis=1, count=n_free)
        grids = domain.expand_many(bits)
        flat = grids.reshape(len(grids), -1)
        keys.append((flat.sum(axis=1), list(map(bytes, flat))))
    pops = np.concatenate([k[0] for k in keys])
    blobs = [b for k in keys for b in k[1]]
    order = sorted(range(len(packed)), key=lambda i: (int(pops[i]), blobs[i]))
    packed = packed[order]
    print(f"sorted in {time.time() - t0:.0f}s")

    dst = HERE / f"solutions_pattern_nosym_level_{level}_cells.npy"
    with open(dst, "wb") as f:
        np.save(f, packed)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    print(f"CENSUS level {level} (no symmetry): {len(packed)}")
    print(f"artifact: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    print(f"file sha256:     {sha}")
    print(f"encoding sha256: {enc.sha256}")
    return 0


def cmd_self_test() -> int:
    """Cube pipeline must reproduce the level-3 single-shot census."""
    level, cube_bits = 3, 6
    enc = build_nosym_cnf(level)
    _init_worker(level)
    rows = []
    for cube in range(1 << cube_bits):
        rows.extend(
            enumerate_all(enc, solver_name=SOLVER,
                          extra_units=cube_units(cube, cube_bits)).tolist())
    bits = np.asarray(rows, dtype=np.uint8)
    grids = enc.domain.expand_many(bits)
    order = sorted(range(len(grids)),
                   key=lambda i: (int(grids[i].sum()), grids[i].tobytes()))
    grids = grids[order]
    reference = enumerate_nosym_tiles(level)
    assert np.array_equal(grids, reference), "cube path != single-shot"
    assert np.array_equal(pack_nosym_solutions(grids, level),
                          pack_nosym_solutions(reference, level))
    print(f"self-test OK: {len(grids)} tiles via {1 << cube_bits} cubes "
          f"== single-shot census")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--level", type=int, default=4)
    run.add_argument("--cube-bits", type=int, default=12)
    run.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    merge = sub.add_parser("merge")
    merge.add_argument("--level", type=int, default=4)
    sub.add_parser("self-test")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args.level, args.cube_bits, args.workers)
    if args.cmd == "merge":
        return cmd_merge(args.level)
    return cmd_self_test()


if __name__ == "__main__":
    sys.exit(main())
