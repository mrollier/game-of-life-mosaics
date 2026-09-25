"""
Write and check the tile databases shipped in src/gol_mosaics/data/.

Every database is stored as packed free-orbit bits (one bit per free
symmetry orbit per tile, see gol_mosaics.tile_domain.Domain.pack) and
expanded when the library loads it. This tool writes them and checks them.

Usage (from the repository root):

    python tools/pack_tiles.py check
        Every shipped file loads, re-packs to exactly its own bytes, and
        has the expected tile count.

    python tools/pack_tiles.py diamond --level 6 --source grids.npy
        Pack full (m, n, n) grids, e.g. the output of
        search/tiles/search.py run, keeping their row order.

    python tools/pack_tiles.py square --level 5
    python tools/pack_tiles.py nosym --level 3
        Enumerate with SAT (needs python-sat) and pack in canonical order.

A file is written only if it expands back byte-identically to the tiles it
was made from.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "gol_mosaics" / "data"

# Tile counts per shipped database: the census.
EXPECTED = {
    ("diamond", 1): 1, ("diamond", 2): 2, ("diamond", 3): 7,
    ("diamond", 4): 85, ("diamond", 5): 2632, ("diamond", 6): 332321,
    ("square", 3): 3, ("square", 4): 65, ("square", 5): 10398,
    ("nosym", 3): 1061,
}


def data_file(kind: str, level: int) -> Path:
    """Where the package looks for one database."""
    if kind == "square":
        return DATA / f"tiles_square_level_{level}_orbits.npy"
    if kind == "nosym":
        return DATA / f"tiles_diamond_nosym_level_{level}_cells.npy"
    return DATA / f"tiles_diamond_level_{level}_orbits.npy"


def domain_for(kind: str, level: int):
    from gol_mosaics.nosym_tiles import build_nosym_domain
    from gol_mosaics.tile_domain import build_domain
    from gol_mosaics.tile_scheme import build_scheme_domain, pond_square_scheme

    if kind == "square":
        return build_scheme_domain(pond_square_scheme(level))
    if kind == "nosym":
        return build_nosym_domain(level)
    return build_domain(level)


def write(kind: str, level: int, tiles: np.ndarray) -> Path:
    from gol_mosaics._atomic import atomic_save

    domain = domain_for(kind, level)
    tiles = np.asarray(tiles, dtype=np.uint8)
    packed = domain.pack(tiles)
    if domain.unpack(packed).tobytes() != tiles.tobytes():
        raise SystemExit(f"FATAL: {kind} level {level} does not expand back "
                         f"byte-identically")
    dst = atomic_save(data_file(kind, level), packed)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    print(f"{kind} level {level}: {len(packed)} tiles x {packed.shape[1]} "
          f"bytes -> {dst.relative_to(REPO)} ({dst.stat().st_size / 1e3:.1f} "
          f"KB, sha256 {sha[:16]})")
    return dst


def cmd_pack(args) -> int:
    t0 = time.time()
    if args.source:
        tiles = np.load(args.source)
    elif args.kind == "square":
        from gol_mosaics.tile_scheme import (enumerate_scheme_tiles,
                                             pond_square_scheme)
        tiles = enumerate_scheme_tiles(pond_square_scheme(args.level))
    elif args.kind == "nosym":
        from gol_mosaics.nosym_tiles import enumerate_nosym_tiles
        tiles = enumerate_nosym_tiles(args.level)
    else:
        from gol_mosaics.sat_search import enumerate_tiles
        tiles = enumerate_tiles(args.level)
    write(args.kind, args.level, tiles)
    print(f"done in {time.time() - t0:.1f}s")
    return 0


def cmd_check(args) -> int:
    failures = 0
    for (kind, level), count in sorted(EXPECTED.items()):
        path = data_file(kind, level)
        if not path.is_file():
            print(f"MISSING {path.relative_to(REPO)}")
            failures += 1
            continue
        packed = np.load(path)
        domain = domain_for(kind, level)
        tiles = domain.unpack(packed)
        ok = (len(tiles) == count
              and np.array_equal(domain.pack(tiles), packed))
        print(f"{'ok  ' if ok else 'FAIL'} {kind:7s} level {level}: "
              f"{len(tiles):>7} tiles  {path.name}")
        failures += not ok
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify every shipped database"
                   ).set_defaults(func=cmd_check)
    for kind in ("diamond", "square", "nosym"):
        p = sub.add_parser(kind, help=f"write the {kind} database of a level")
        p.add_argument("--level", type=int, required=True)
        p.add_argument("--source", type=Path,
                       help="full (m, n, n) grids to pack, in their order; "
                            "default: enumerate with SAT")
        p.set_defaults(func=cmd_pack, kind=kind)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
