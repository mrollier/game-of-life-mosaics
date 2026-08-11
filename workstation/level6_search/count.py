"""
Exact census counting via #SAT model counting.

The encoding has no auxiliary variables, so the model count of the level-L
CNF *is* the census size — no enumeration, no storage. This script exports
the CNF in DIMACS, runs an external exact model counter (sharpSAT-td,
ganak, or anything printing the standard `c s exact arb int N` line), and
compares the reported count against the shipped census when one exists.

  python count.py --level 7 --counter tools/sharpsat-td/build/sharpSAT \
                  --counter-args "-decot 1 -decow 100 -tmpdir counts -cs 3500"

Without --counter it only writes counts/level_L.cnf (with the `c t mc`
model-counting header) and prints the encoding fingerprint.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from encoding import build_cnf

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "src" / "gol_mosaics" / "data"

# Ground truth: shipped censuses (levels 1-6) and the local level-7 artifact.
KNOWN = {1: 1, 2: 2, 3: 7, 4: 85, 5: 2632, 6: 332321}
LEVEL7_ARTIFACT = HERE / "solutions_pattern_level_7_orbits.npy"

COUNT_PATTERNS = [
    re.compile(r"^c?\s*s\s+(?:exact|approx)?\s*(?:arb|log10)?\s*(?:int)?\s+(\d+)\s*$"),
    re.compile(r"^s\s+mc\s+(\d+)\s*$"),
    re.compile(r"^\s*(\d+)\s*$"),  # bare count after '# solutions' banner
]


def known_count(level: int):
    if level in KNOWN:
        return KNOWN[level]
    if level == 7 and LEVEL7_ARTIFACT.exists():
        packed = np.load(LEVEL7_ARTIFACT, mmap_mode="r")
        return len(packed)
    return None


def write_dimacs(level: int, out_dir: Path) -> Path:
    enc = build_cnf(level)
    path = out_dir / f"level_{level}.cnf"
    with open(path, "w") as f:
        f.write(f"c gol-mosaics level {level} tile census "
                f"(model count = census size; no auxiliary variables)\n")
        f.write(f"c encoding sha256 {enc.sha256}\n")
        f.write("c t mc\n")
        f.write(f"p cnf {enc.n_vars} {len(enc.clauses)}\n")
        f.write("".join(" ".join(map(str, cl)) + " 0\n" for cl in enc.clauses))
    print(f"level {level}: {enc.n_vars} vars, {len(enc.clauses)} clauses -> "
          f"{path} (encoding sha256 {enc.sha256[:16]}...)")
    return path


def parse_count(text: str):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "# solutions" in line:  # classic sharpSAT banner: count on next line
            for after in lines[i + 1:]:
                after = after.strip()
                if after.isdigit():
                    return int(after)
        for pat in COUNT_PATTERNS[:2]:
            m = pat.match(line.strip())
            if m:
                return int(m.group(1))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--counter", default=None,
                    help="path to an exact model counter binary")
    ap.add_argument("--counter-args", default="",
                    help="extra arguments passed to the counter")
    ap.add_argument("--out-dir", default=str(HERE / "counts"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cnf_path = write_dimacs(args.level, out_dir)
    if not args.counter:
        return 0

    cmd = (["/usr/bin/time", "-l", args.counter]
           + args.counter_args.split() + [str(cnf_path)])
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0

    count = parse_count(proc.stdout)
    rss = None
    m = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
    if m:
        rss = int(m.group(1)) / 1e9  # bytes on macOS
    if count is None:
        print(f"level {args.level}: FAILED to parse a count "
              f"(exit {proc.returncode}, {wall:.1f}s). Last output lines:")
        print("\n".join(proc.stdout.splitlines()[-15:]))
        print(proc.stderr.splitlines()[-3:] if proc.stderr else "")
        return 1

    ref = known_count(args.level)
    if ref is not None:
        status = "PASS" if count == ref else f"FAIL (expected {ref})"
    else:
        status = "NEW"
    print(f"level {args.level}: count = {count}  [{status}]  "
          f"wall {wall:.1f}s"
          + (f"  peak RSS {rss:.2f} GB" if rss is not None else ""))
    return 0 if status.startswith(("PASS", "NEW")) else 1


if __name__ == "__main__":
    sys.exit(main())
