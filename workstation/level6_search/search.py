"""
Exhaustive enumeration of symmetric still-life tiles via SAT.

Subcommands:
  validate  re-enumerate levels 3-5 and compare byte-exactly against the
            shipped reference files (mandatory gate before a big run);
            prints calibration timings
  run       cube-and-conquer search for one level, parallel + checkpointed
  merge     assemble, verify, sort, and save the final artifact from cubes

Typical session:
  python search.py validate
  python search.py run --level 6            # -> solutions_pattern_level_6.npy
  python search.py run --level 7            # -> solutions_pattern_level_7_orbits.npy

Levels >= 7 default to --packed output: the artifact holds one bit per free
symmetry orbit per tile (level 7: 11 bytes/tile instead of a 42x42 grid),
because ~10^8 expected level-7 tiles would be hundreds of GB as raw grids.
Expand with gol_mosaics.tile_domain.unpack_solutions (in chunks!).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from encoding import Encoding, build_cnf, cube_units, enumerate_all
from verify import check_batch, compare_sets

HERE = Path(__file__).resolve().parent
DEFAULT_SOLVER = "cadical195"
CHECKPOINT_FORMAT = "cube-packed-v2"
VERIFY_CHUNK = 1 << 16


# ---------------------------------------------------------------- worker pool

_worker_state = {}


def _worker_init(level: int, solver_name: str, cubes_dir: str):
    _worker_state["enc"] = build_cnf(level)
    _worker_state["solver"] = solver_name
    _worker_state["cubes_dir"] = Path(cubes_dir)


def _run_cube(args):
    cube, cube_bits = args
    enc = _worker_state["enc"]
    bits = enumerate_all(enc, _worker_state["solver"],
                         extra_units=cube_units(cube, cube_bits))
    packed = np.packbits(bits, axis=1)
    path = _worker_state["cubes_dir"] / f"cube_{cube:05d}.npy"
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, packed)
    os.replace(tmp, path)
    return cube, len(bits)


# ---------------------------------------------------------------- run

def manifest_for(level: int, cube_bits: int, solver_name: str, enc: Encoding) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "level": level,
        "n": enc.domain.n,
        "birth": list(enc.birth),
        "survival": list(enc.survival),
        "cube_bits": cube_bits,
        "n_vars": enc.n_vars,
        "free_reps": [list(rep) for rep in enc.domain.free_reps],
        "encoding_sha256": enc.sha256,
        "solver": solver_name,
    }


def cmd_run(args) -> int:
    level = args.level
    enc = build_cnf(level)
    if args.cube_bits >= enc.n_vars:
        print(f"--cube-bits must be < {enc.n_vars} (free variables)")
        return 2
    workers = args.workers or default_workers()

    work_dir = Path(args.work_dir) / f"level_{level}"
    cubes_dir = work_dir / "cubes"
    cubes_dir.mkdir(parents=True, exist_ok=True)

    manifest = manifest_for(level, args.cube_bits, args.solver, enc)
    manifest_path = work_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing.pop("created_at", None)
        if existing != manifest:
            print(f"manifest mismatch in {work_dir} — the checkpoint was made "
                  f"with different code/parameters. Remove the directory to "
                  f"start over, or rerun with the original parameters.")
            return 2
        print(f"resuming from checkpoint in {work_dir}")
    else:
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2))

    total = 2 ** args.cube_bits
    pending = [c for c in range(total)
               if not (cubes_dir / f"cube_{c:05d}.npy").exists()]
    done_before = total - len(pending)
    found_before = sum(
        len(np.load(cubes_dir / f"cube_{c:05d}.npy")) for c in range(total)
        if (cubes_dir / f"cube_{c:05d}.npy").exists()
    )
    print(f"level {level}: {enc.n_vars} free variables, "
          f"{len(enc.clauses)} clauses, {total} cubes "
          f"({done_before} already done), {workers} workers, "
          f"solver {args.solver}")

    if pending:
        t0 = time.time()
        done, found = done_before, found_before
        is_tty = sys.stderr.isatty()
        last_report = 0.0
        with mp.Pool(workers, initializer=_worker_init,
                     initargs=(level, args.solver, str(cubes_dir))) as pool:
            tasks = ((c, args.cube_bits) for c in pending)
            for cube, count in pool.imap_unordered(_run_cube, tasks,
                                                   chunksize=4):
                done += 1
                found += count
                elapsed = time.time() - t0
                # Throttle progress lines so nohup/log output stays readable
                if done == total or elapsed - last_report >= (0.5 if is_tty else 10.0):
                    last_report = elapsed
                    rate = (done - done_before) / elapsed
                    eta = (total - done) / rate if rate > 0 else float("inf")
                    prefix = "\r" if is_tty else ""
                    print(f"{prefix}cubes {done}/{total}  "
                          f"solutions {found}  elapsed {elapsed:7.1f}s  "
                          f"ETA {eta:7.1f}s",
                          end="" if is_tty else "\n",
                          file=sys.stderr, flush=True)
        if is_tty:
            print(file=sys.stderr)
        print(f"solve phase: {time.time() - t0:.1f}s wall on {workers} workers "
              f"({done - done_before} cubes, {found - found_before} solutions)")

    return cmd_merge(args)


# ---------------------------------------------------------------- merge

def _wants_packed(args) -> bool:
    return args.packed or (args.level >= 7 and not args.grids)


def cmd_merge(args) -> int:
    level = args.level
    enc = build_cnf(level)
    domain = enc.domain
    work_dir = Path(args.work_dir) / f"level_{level}"
    cubes_dir = work_dir / "cubes"

    manifest = json.loads((work_dir / "manifest.json").read_text())
    if manifest.get("format") != CHECKPOINT_FORMAT:
        print(f"checkpoint format {manifest.get('format')!r} not supported "
              f"(expected {CHECKPOINT_FORMAT!r}); re-run the search")
        return 2
    if manifest["encoding_sha256"] != enc.sha256:
        print("manifest encoding fingerprint does not match current code")
        return 2
    total = 2 ** manifest["cube_bits"]
    missing = [c for c in range(total)
               if not (cubes_dir / f"cube_{c:05d}.npy").exists()]
    if missing:
        print(f"{len(missing)} of {total} cubes incomplete "
              f"(e.g. cube {missing[0]}) — rerun `run` to finish them")
        return 2

    ncols = (enc.n_vars + 7) // 8
    t_load = time.time()
    packed = np.vstack([
        np.load(cubes_dir / f"cube_{c:05d}.npy").reshape(-1, ncols)
        for c in range(total)
    ])
    m = len(packed)
    t_load = time.time() - t_load

    # Chunked verification with the independent checker, plus alive counts
    # (needed for sorting) computed per chunk without keeping grids around.
    t_verify = time.time()
    alive_counts = np.empty(m, dtype=np.int64)
    for start in range(0, m, VERIFY_CHUNK):
        stop = min(start + VERIFY_CHUNK, m)
        bits = np.unpackbits(packed[start:stop], axis=1, count=enc.n_vars)
        grids = domain.expand_many(bits)
        ok = check_batch(grids, level)
        if not ok.all():
            bad = start + int(np.where(~ok)[0][0])
            print(f"FATAL: solution #{bad} fails independent verification")
            return 1
        alive_counts[start:stop] = grids.sum(axis=(1, 2))
    t_verify = time.time() - t_verify
    t_dedup = time.time()
    if len(np.unique(packed, axis=0)) != m:
        print("FATAL: duplicate solutions across cubes")
        return 1
    t_dedup = time.time() - t_dedup
    print(f"merge phases: load {t_load:.1f}s, verify {t_verify:.1f}s, "
          f"uniqueness {t_dedup:.1f}s")

    if _wants_packed(args):
        # Sort by (alive count, packed-bit row); packbits is bitorder='big',
        # so byte-lexicographic equals bit-lexicographic.
        order = np.lexsort(
            tuple(packed[:, c] for c in reversed(range(ncols))) + (alive_counts,)
        )
        packed = packed[order]
        out = Path(args.output or f"solutions_pattern_level_{level}_orbits.npy")
        with open(out, "wb") as f:
            np.save(f, packed)
        n_free = enc.n_vars
        print(f"saved {out}: {m} solutions as packed orbit bits "
              f"({n_free} bits/tile), {out.stat().st_size / 1e6:.1f} MB. "
              f"Expand with gol_mosaics.tile_domain.unpack_solutions "
              f"(in chunks: full level-{level} grids would be "
              f"{m * domain.n * domain.n / 1e9:.0f} GB).")
        return 0

    # Grid output (levels <= 6): sort by (alive count, grid bytes) — the
    # historical order of the shipped data files.
    t_sort = time.time()
    bits = np.unpackbits(packed, axis=1, count=enc.n_vars)
    grids = domain.expand_many(bits)
    order = sorted(range(m),
                   key=lambda i: (int(alive_counts[i]), grids[i].tobytes()))
    grids = grids[order]
    t_sort = time.time() - t_sort
    t_write = time.time()
    out = Path(args.output or f"solutions_pattern_level_{level}.npy")
    with open(out, "wb") as f:
        np.save(f, grids)
    t_write = time.time() - t_write
    print(f"merge phases: expand+sort {t_sort:.1f}s, write {t_write:.1f}s")
    densities = grids.mean(axis=(1, 2))
    print(f"saved {out}: {m} solutions, shape {grids.shape}, "
          f"dtype {grids.dtype}, {out.stat().st_size / 1e6:.1f} MB, "
          f"density {densities.min():.3f}..{densities.max():.3f}")
    return 0


# ---------------------------------------------------------------- validate

def cmd_validate(args) -> int:
    import shutil
    import tempfile

    overall_ok = True
    t5 = n5 = None

    # Levels 3 and 4: single-shot enumeration
    for level in (3, 4):
        t0 = time.time()
        enc = build_cnf(level)
        grids = enc.domain.expand_many(
            enumerate_all(enc, args.solver))
        elapsed = time.time() - t0
        ok = _check_against_reference(grids, level, elapsed)
        overall_ok &= ok

    # Level 5: through the full cube + multiprocessing + checkpoint path
    tmp = tempfile.mkdtemp(prefix="l6search_validate_")
    try:
        run_args = argparse.Namespace(
            level=5, cube_bits=6, workers=args.workers, solver=args.solver,
            work_dir=tmp, output=os.path.join(tmp, "level5.npy"),
            packed=False, grids=False)
        t0 = time.time()
        rc = cmd_run(run_args)
        elapsed = time.time() - t0
        if rc != 0:
            print("level 5 parallel run FAILED")
            overall_ok = False
        else:
            grids = np.load(run_args.output)
            ok = _check_against_reference(grids, 5, elapsed,
                                          suffix=" (parallel cube path)")
            overall_ok &= ok
            t5, n5 = elapsed, len(grids)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if overall_ok and t5:
        per_sol = t5 / n5
        print(f"\ncalibration: level 5 took {t5:.1f}s for {n5} solutions "
              f"({per_sol * 1e3:.2f} ms/solution through the parallel path)")
        for level, n_est in ((6, 3.3e5), (7, 1.5e8)):
            est = per_sol * n_est * 3  # ~3x for the larger formulas
            print(f"  level-{level} estimate at {n_est:.1e} solutions: "
                  f"~{est:.0f}s ({est / 3600:.1f} h)")
        print("\nVALIDATION PASSED — safe to run: python search.py run --level 6")
        return 0
    print("\nVALIDATION FAILED — do not run higher levels until this is fixed")
    return 1


def _reference_path(level: int) -> Path:
    """Local reference/ copy if present, else the package's shipped data
    (byte-identical; the pipeline runs from a repo checkout anyway)."""
    local = HERE / "reference" / f"solutions_pattern_level_{level}.npy"
    if local.exists():
        return local
    return (HERE.parents[1] / "src" / "gol_mosaics" / "data"
            / f"solutions_pattern_level_{level}.npy")


def _check_against_reference(grids, level, elapsed, suffix="") -> bool:
    n_bad = int((~check_batch(grids, level)).sum())
    ref = np.load(_reference_path(level))
    ok, msg = compare_sets(grids, ref)
    status = "PASS" if ok and not n_bad else "FAIL"
    print(f"level {level}{suffix}: {status} — {msg}, "
          f"{n_bad} verifier failures, {elapsed:.1f}s")
    return ok and not n_bad


# ---------------------------------------------------------------- CLI

def default_workers() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        return os.cpu_count() or 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--solver", default=DEFAULT_SOLVER,
                        help=f"pysat solver name (default: {DEFAULT_SOLVER}; "
                             f"try glucose42 if unavailable)")
    common.add_argument("--workers", type=int, default=None,
                        help="worker processes (default: all cores)")

    p = sub.add_parser("validate", parents=[common],
                       help="re-derive levels 3-5 and compare to reference")
    p.set_defaults(func=cmd_validate)

    out_opts = argparse.ArgumentParser(add_help=False)
    out_opts.add_argument("--level", type=int, default=6)
    out_opts.add_argument("--work-dir", default="work")
    out_opts.add_argument("--output", default=None)
    out_opts.add_argument("--packed", action="store_true",
                          help="save packed orbit bits instead of grids "
                               "(default for level >= 7)")
    out_opts.add_argument("--grids", action="store_true",
                          help="force full-grid output even for level >= 7 "
                               "(hundreds of GB — you almost never want this)")

    p = sub.add_parser("run", parents=[common, out_opts],
                       help="checkpointed parallel search for one level")
    p.add_argument("--cube-bits", type=int, default=10,
                   help="split into 2^k cubes (default 10; use 14-16 for level 7)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("merge", parents=[common, out_opts],
                       help="assemble final artifact from completed cubes")
    p.set_defaults(func=cmd_merge)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
