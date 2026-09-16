"""
Adaptive cube-and-conquer enumeration of symmetric still-life tiles.

Same encoding (encoding.build_cnf), same independent checker (verify.py)
and same output format and ordering as search.py — but the assignment
space is partitioned *adaptively* instead of on a fixed 2^k prefix:

    node (k, cube)  = the first k decision variables fixed to cube's bits
    enumerate with a cap; fewer than --cap solutions  -> leaf (kept)
                          the cap is reached          -> split into
                          2^--split-bits children on the next variables

Why: the fixed prefix of search.py lands on the tile's apex orbits, so
almost every cube is immediately UNSAT and the census concentrates in a few
huge cubes (level 7: 65 of 65,536 cubes are non-empty). Inside a huge cube
the AllSAT loop slows from ~30k to <6k tiles/s as blocking clauses pile up.
Capped leaves keep every solver instance in its fast regime, the tree
balances itself over the workers, and verification runs *inside* the
workers in small chunks (3x faster per tile than 64k chunks, and parallel)
instead of serially in the merge.

Exhaustiveness argument (unchanged in spirit): every node is either a leaf
whose solver ran to UNSAT with all solutions recorded, or was replaced by
children that partition its assignments exactly; leaves have distinct
prefixes and so are disjoint. Their union is the census.

Checkpointing: one atomically written file per leaf plus an append-only
nodes.jsonl; rerunning `run` resumes from the log. The manifest pins the
CNF fingerprint so mismatched code/parameters are refused.

  python search_adaptive.py self-test                    # levels 5-6 byte-exact
  python search_adaptive.py run --level 7 --expect 108492376
  python search_adaptive.py merge --level 7
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from encoding import Encoding, build_cnf, cube_units, enumerate_all
from search import default_workers, manifest_for
from verify import check_batch, compare_sets

HERE = Path(__file__).resolve().parent
DEFAULT_SOLVER = "cadical195"
CHECKPOINT_FORMAT = "adaptive-leaves-v1"
VERIFY_CHUNK = 2048


def leaf_name(k: int, cube: int) -> str:
    return f"d{k:03d}_{cube:x}.npy"


# ---------------------------------------------------------------- worker

_ws = {}


def _worker_init(level, solver_name, leaves_dir, cap, nice):
    if nice:
        os.nice(nice)
    _ws["enc"] = build_cnf(level)
    _ws["level"] = level
    _ws["solver"] = solver_name
    _ws["leaves"] = Path(leaves_dir)
    _ws["cap"] = cap


def _run_node(args):
    """Enumerate one node with a cap. Returns (k, cube, status, count, cpu)."""
    k, cube = args
    enc: Encoding = _ws["enc"]
    cap = _ws["cap"]
    t0 = time.process_time()
    bits = enumerate_all(enc, _ws["solver"], extra_units=cube_units(cube, k),
                         limit=cap)
    if len(bits) >= cap:
        return k, cube, "split", len(bits), time.process_time() - t0

    # Leaf: verify every solution with the independent checker before it
    # can reach disk (small chunks: much faster per tile than 64k chunks).
    for start in range(0, len(bits), VERIFY_CHUNK):
        chunk = bits[start:start + VERIFY_CHUNK]
        ok = check_batch(enc.domain.expand_many(chunk), _ws["level"])
        if not ok.all():
            raise RuntimeError(
                f"FATAL: node k={k} cube={cube:x}: solution "
                f"#{start + int(np.where(~ok)[0][0])} fails independent "
                f"verification")
    packed = np.packbits(bits, axis=1)
    path = _ws["leaves"] / leaf_name(k, cube)
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, packed)
    os.replace(tmp, path)
    return k, cube, "leaf", len(bits), time.process_time() - t0


# ---------------------------------------------------------------- tree

def children(k: int, cube: int, split_bits: int, n_vars: int):
    b = min(split_bits, n_vars - k)
    return [(k + b, cube | (x << k)) for x in range(1 << b)]


def replay_log(log_path: Path):
    """Return (leaves: {(k,cube): count}, splits: set[(k,cube)])."""
    leaves, splits = {}, set()
    if not log_path.exists():
        return leaves, splits
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["k"], int(rec["cube"], 16))
            if rec["status"] == "leaf":
                leaves[key] = rec["count"]
            else:
                splits.add(key)
    return leaves, splits


def pending_nodes(root_bits, split_bits, n_vars, leaves, splits, leaves_dir):
    """Walk the tree from the root; nodes neither logged as leaf nor split
    are pending. A leaf logged without its file (should not happen: the
    file is written first) is re-run rather than trusted."""
    pending = []
    stack = children(0, 0, root_bits, n_vars)
    while stack:
        node = stack.pop()
        if node in leaves and (leaves_dir / leaf_name(*node)).exists():
            continue
        if node in splits:
            stack.extend(children(node[0], node[1], split_bits, n_vars))
            continue
        pending.append(node)
    # deepest first: those are the cheapest, but more importantly a
    # resumed run should finish subtrees before starting new ones
    pending.sort(key=lambda n: (-n[0], n[1]))
    return pending


# ---------------------------------------------------------------- run

def cmd_run(args) -> int:
    level = args.level
    enc = build_cnf(level)
    n_vars = enc.n_vars
    if args.cap < 2:
        print("--cap must be >= 2")
        return 2
    if not (1 <= args.root_bits < n_vars and 1 <= args.split_bits <= n_vars):
        print(f"--root-bits must be in 1..{n_vars - 1}, --split-bits in 1..{n_vars}")
        return 2
    workers = args.workers or default_workers()

    work_dir = Path(args.work_dir) / f"level_{level}"
    leaves_dir = work_dir / "leaves"
    leaves_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "nodes.jsonl"

    manifest = manifest_for(level, args.root_bits, args.solver, enc)
    manifest.update(format=CHECKPOINT_FORMAT, cube_bits=None,
                    root_bits=args.root_bits, split_bits=args.split_bits,
                    cap=args.cap)
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

    leaves, splits = replay_log(log_path)
    pending = pending_nodes(args.root_bits, args.split_bits, n_vars,
                            leaves, splits, leaves_dir)
    found_before = sum(leaves.values())
    print(f"level {level}: {n_vars} free variables, {len(enc.clauses)} clauses, "
          f"cap {args.cap}, root 2^{args.root_bits} cubes, split 2^{args.split_bits}, "
          f"{workers} workers, solver {args.solver}, nice {args.nice}")
    print(f"resume state: {len(leaves)} leaves ({found_before} tiles), "
          f"{len(splits)} splits, {len(pending)} pending nodes")
    if not pending:
        return cmd_merge(args)

    if args.nice:
        os.nice(args.nice)
    t0 = time.time()
    n_leaf = n_split = n_empty = 0
    found = found_before
    cpu_total = wasted = 0.0
    is_tty = sys.stderr.isatty()
    last_report = 0.0
    max_depth = 0
    inflight = {}
    log = open(log_path, "a")
    try:
        with ProcessPoolExecutor(
                workers, initializer=_worker_init,
                initargs=(level, args.solver, str(leaves_dir), args.cap,
                          args.nice)) as pool:
            def submit(node):
                inflight[pool.submit(_run_node, node)] = node

            # Keep a bounded number of tasks queued so that splits found
            # early get their children scheduled promptly.
            queue_target = 4 * workers
            while pending and len(inflight) < queue_target:
                submit(pending.pop())
            while inflight:
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in done:
                    inflight.pop(fut)
                    k, cube, status, count, cpu = fut.result()  # raises on FATAL
                    cpu_total += cpu
                    max_depth = max(max_depth, k)
                    if status == "leaf":
                        n_leaf += 1
                        found += count
                        if count == 0:
                            n_empty += 1
                    else:
                        n_split += 1
                        wasted += cpu
                        pending.extend(
                            children(k, cube, args.split_bits, n_vars))
                    log.write(json.dumps({"k": k, "cube": f"{cube:x}",
                                          "status": status,
                                          "count": count}) + "\n")
                    log.flush()
                while pending and len(inflight) < queue_target:
                    submit(pending.pop())

                elapsed = time.time() - t0
                if not inflight or elapsed - last_report >= (0.5 if is_tty else 15.0):
                    last_report = elapsed
                    rate = (found - found_before) / elapsed
                    eta = ""
                    if args.expect and rate > 0:
                        eta = (f"  ETA {(args.expect - found) / rate:7.0f}s "
                               f"({100 * found / args.expect:5.1f}%)")
                    prefix = "\r" if is_tty else ""
                    print(f"{prefix}leaves {n_leaf} (empty {n_empty})  splits {n_split}  "
                          f"pending {len(pending) + len(inflight)}  depth<={max_depth}  "
                          f"tiles {found}  {rate:8.0f}/s  cpu {cpu_total:7.0f}s "
                          f"(discarded {wasted:5.0f}s)  elapsed {elapsed:7.1f}s{eta}",
                          end="" if is_tty else "\n", file=sys.stderr, flush=True)
    finally:
        log.close()
    if is_tty:
        print(file=sys.stderr)
    wall = time.time() - t0
    print(f"solve phase: {wall:.1f}s wall on {workers} workers, {cpu_total:.0f}s cpu "
          f"({100 * cpu_total / wall / workers:.0f}% busy; {wasted:.0f}s cpu on capped "
          f"nodes discarded); {n_leaf} leaves ({n_empty} empty), {n_split} splits, "
          f"max depth {max_depth}, {found - found_before} new tiles (total {found})")
    return cmd_merge(args)


# ---------------------------------------------------------------- merge

def orbit_sizes(enc: Encoding):
    """Live-cell count of a tile = base + bits . sizes, from the domain."""
    base = int(enc.domain.expand_many(
        np.zeros((1, enc.n_vars), dtype=np.uint8))[0].sum())
    eye = np.eye(enc.n_vars, dtype=np.uint8)
    sizes = enc.domain.expand_many(eye).sum(axis=(1, 2)).astype(np.int64) - base
    return base, sizes


def cmd_merge(args) -> int:
    level = args.level
    enc = build_cnf(level)
    ncols = (enc.n_vars + 7) // 8
    work_dir = Path(args.work_dir) / f"level_{level}"
    leaves_dir = work_dir / "leaves"
    manifest = json.loads((work_dir / "manifest.json").read_text())
    if manifest.get("format") != CHECKPOINT_FORMAT:
        print(f"checkpoint format {manifest.get('format')!r} not supported")
        return 2
    if manifest["encoding_sha256"] != enc.sha256:
        print("manifest encoding fingerprint does not match current code")
        return 2
    leaves, splits = replay_log(work_dir / "nodes.jsonl")
    pending = pending_nodes(manifest["root_bits"], manifest["split_bits"],
                            enc.n_vars, leaves, splits, leaves_dir)
    if pending:
        print(f"{len(pending)} nodes incomplete (e.g. k={pending[0][0]} "
              f"cube={pending[0][1]:x}) — rerun `run` to finish them")
        return 2

    t = time.time()
    parts = []
    for (k, cube), count in sorted(leaves.items()):
        if count == 0:
            continue
        arr = np.load(leaves_dir / leaf_name(k, cube)).reshape(-1, ncols)
        if len(arr) != count:
            print(f"FATAL: leaf k={k} cube={cube:x} holds {len(arr)} rows, "
                  f"log says {count}")
            return 1
        parts.append(arr)
    packed = np.vstack(parts) if parts else np.empty((0, ncols), np.uint8)
    del parts
    m = len(packed)
    t_load = time.time() - t
    print(f"loaded {m} tiles from {len(leaves)} leaves in {t_load:.1f}s")

    # Every tile was verified in its worker; here: uniqueness across leaves,
    # live-cell counts (needed for the canonical order), and one more
    # independent spot check on a random sample of the assembled array.
    t = time.time()
    rows = np.ascontiguousarray(packed).view(np.dtype((np.void, ncols))).ravel()
    n_unique = len(np.unique(rows))
    del rows
    if n_unique != m:
        print(f"FATAL: {m - n_unique} duplicate solutions across leaves")
        return 1
    t_dedup = time.time() - t

    t = time.time()
    base, sizes = orbit_sizes(enc)
    alive = np.empty(m, dtype=np.int64)
    step = 1 << 20
    for s in range(0, m, step):
        bits = np.unpackbits(packed[s:s + step], axis=1, count=enc.n_vars)
        alive[s:s + step] = base + bits.astype(np.int64) @ sizes
    # cross-check the arithmetic against real grids on a sample
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(m, size=min(m, 4096), replace=False)) if m else []
    if m:
        grids = enc.domain.expand_many(
            np.unpackbits(packed[idx], axis=1, count=enc.n_vars))
        if not np.array_equal(grids.sum(axis=(1, 2)), alive[idx]):
            print("FATAL: orbit-size live-cell counts disagree with expanded grids")
            return 1
        ok = check_batch(grids, level)
        if not ok.all():
            print("FATAL: sampled tile fails independent verification after merge")
            return 1
    t_alive = time.time() - t

    t = time.time()
    order = np.lexsort(tuple(packed[:, c] for c in reversed(range(ncols))) + (alive,))
    packed = packed[order]
    t_sort = time.time() - t
    out = Path(args.output or f"solutions_pattern_level_{level}_orbits.npy")
    tmp = out.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, packed)
    os.replace(tmp, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"merge phases: load {t_load:.1f}s, uniqueness {t_dedup:.1f}s, "
          f"alive counts {t_alive:.1f}s, sort {t_sort:.1f}s")
    print(f"saved {out}: {m} solutions as packed orbit bits ({enc.n_vars} bits/tile), "
          f"{out.stat().st_size / 1e6:.1f} MB, sha256 {sha}")
    return 0


# ---------------------------------------------------------------- self-test

def cmd_self_test(args) -> int:
    """Levels 5 and 6 through the adaptive path, with small caps so the tree
    really splits, compared byte-exactly with the shipped censuses (level 6
    against the shipped *packed* file, which also checks the ordering)."""
    import shutil
    import tempfile
    from gol_mosaics.tile_domain import unpack_solutions

    data = HERE.parents[1] / "src" / "gol_mosaics" / "data"
    ok_all = True
    for level, cap, root_bits, split_bits in ((5, 200, 4, 2), (6, 3000, 8, 3)):
        tmp = tempfile.mkdtemp(prefix="adaptive_selftest_")
        try:
            out = os.path.join(tmp, f"level{level}.npy")
            run_args = argparse.Namespace(
                level=level, cap=cap, root_bits=root_bits, split_bits=split_bits,
                workers=args.workers, solver=args.solver, work_dir=tmp,
                output=out, expect=None, nice=args.nice)
            t0 = time.time()
            rc = cmd_run(run_args)
            elapsed = time.time() - t0
            if rc != 0:
                print(f"level {level}: FAIL (rc={rc})")
                ok_all = False
                continue
            mine = np.load(out)
            if level == 6:
                ref = np.load(data / "solutions_pattern_level_6_orbits.npy")
                same = mine.shape == ref.shape and mine.tobytes() == ref.tobytes()
                what = "packed file byte-identical"
            else:
                # the level <= 5 files are grid-ordered, so compare as sets
                # (as search.py validate does)
                ref = np.load(data / f"solutions_pattern_level_{level}.npy")
                same, what = compare_sets(unpack_solutions(mine, level=level), ref)
            print(f"level {level} (adaptive, cap {cap}): "
                  f"{'PASS' if same else 'FAIL'} — {len(mine)} tiles, {what}: "
                  f"{same}, {elapsed:.1f}s")
            ok_all &= same
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("\nSELF-TEST PASSED" if ok_all else "\nSELF-TEST FAILED")
    return 0 if ok_all else 1


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--solver", default=DEFAULT_SOLVER)
    common.add_argument("--workers", type=int, default=None)
    common.add_argument("--nice", type=int, default=0,
                        help="os.nice() increment for the main and worker processes")

    p = sub.add_parser("self-test", parents=[common],
                       help="levels 5-6 through the adaptive path, byte-exact")
    p.set_defaults(func=cmd_self_test)

    out_opts = argparse.ArgumentParser(add_help=False)
    out_opts.add_argument("--level", type=int, default=7)
    out_opts.add_argument("--work-dir", default="work_adaptive")
    out_opts.add_argument("--output", default=None)

    p = sub.add_parser("run", parents=[common, out_opts],
                       help="adaptive, checkpointed parallel search; merges when complete")
    p.add_argument("--cap", type=int, default=50000,
                   help="split a node once it yields this many solutions (default 50000)")
    p.add_argument("--root-bits", type=int, default=12,
                   help="the root is split into 2^k cubes up front (default 12)")
    p.add_argument("--split-bits", type=int, default=4,
                   help="a capped node splits into 2^k children (default 4)")
    p.add_argument("--expect", type=int, default=None,
                   help="expected census size, for the ETA only")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("merge", parents=[common, out_opts],
                       help="assemble the final artifact from completed leaves")
    p.set_defaults(func=cmd_merge)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
