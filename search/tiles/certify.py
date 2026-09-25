"""
Emit and check UNSAT certificates for the completeness of a census.

For level L, the certificate is a DRAT proof that

    CNF(L)  ∧  ¬t_1 ∧ ... ∧ ¬t_N        (one blocking clause per census tile)

is unsatisfiable: together with the independent per-tile verification (the
SAT direction), this certifies that the shipped census is exactly the
solution set — with no residual trust in the enumerating solver or in the
cube partition.

Usage:
    python certify.py --level 5
    python certify.py --level 6 --checker drat-trim   # needs drat-trim in PATH

Outputs certificates/level_L.cnf and certificates/level_L.drat, then checks
the proof with drat-trim when available (or when --checker names a binary),
falling back to the bundled pure-Python rup_check.py for proofs small
enough to check that way (< ~50k lemmas).
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from encoding import build_cnf  # noqa: E402

RUP_CHECK_MAX_LEMMAS = 50_000


def census_bits(level, enc):
    """Free-orbit bits of the shipped census for this level."""
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from gol_mosaics.patterns import PatternLibrary
    sols = PatternLibrary.load(level).solutions
    return enc.domain.extract_bits(sols)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    # Glucose emits pure-CDCL DRAT proofs whose final conflict is always
    # reachable by unit propagation (CaDiCaL's inprocessing can leave tails
    # that RUP checking alone cannot finish); it is also a different
    # codebase from the enumeration solver, which separates concerns.
    ap.add_argument("--solver", default="glucose42")
    ap.add_argument("--out-dir", default=str(HERE / "certificates"))
    ap.add_argument("--checker", default=None,
                    help="proof checker binary (default: drat-trim if in "
                         "PATH, else the bundled rup_check.py for small "
                         "proofs)")
    args = ap.parse_args(argv)

    from pysat.solvers import Solver

    level = args.level
    enc = build_cnf(level)
    bits = census_bits(level, enc)
    nvars = enc.n_vars
    print(f"level {level}: {nvars} variables, {len(enc.clauses)} clauses, "
          f"{len(bits)} census tiles")

    var_range = np.arange(1, nvars + 1)
    blockers = np.where(bits.astype(bool), -var_range, var_range).tolist()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cnf_path = out_dir / f"level_{level}.cnf"
    proof_path = out_dir / f"level_{level}.drat"

    with open(cnf_path, "w") as f:
        f.write(f"c completeness certificate formula, level {level}\n"
                f"c encoding sha256 {enc.sha256}\n")
        f.write(f"p cnf {nvars} {len(enc.clauses) + len(blockers)}\n")
        for c in enc.clauses:
            f.write(" ".join(map(str, c)) + " 0\n")
        for c in blockers:
            f.write(" ".join(map(str, c)) + " 0\n")

    solver = Solver(name=args.solver, with_proof=True)
    for c in enc.clauses:
        solver.add_clause(list(c))
    solver.append_formula(blockers)
    t0 = time.time()
    sat = solver.solve()
    solve_s = time.time() - t0
    assert sat is False, "certificate formula is SAT — census incomplete?!"
    proof = solver.get_proof()
    solver.delete()
    lemmas = sum(1 for line in proof if not line.startswith("d "))
    with open(proof_path, "w") as f:
        f.write("\n".join(proof) + "\n0\n")   # ensure the empty clause
    sha = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    print(f"UNSAT in {solve_s:.1f}s; proof {lemmas} lemmas "
          f"({proof_path.stat().st_size / 1e6:.1f} MB), sha256 {sha[:16]}…")

    checker = args.checker or ("drat-trim" if shutil.which("drat-trim")
                               else None)
    t0 = time.time()
    if checker:
        res = subprocess.run([checker, str(cnf_path), str(proof_path)],
                             capture_output=True, text=True)
        ok = "s VERIFIED" in res.stdout
        print(res.stdout.strip().splitlines()[-1])
        print(f"{checker}: {'VERIFIED' if ok else 'NOT VERIFIED'} "
              f"({time.time() - t0:.1f}s)")
        return 0 if ok else 1
    if lemmas <= RUP_CHECK_MAX_LEMMAS:
        res = subprocess.run([sys.executable, str(HERE / "rup_check.py"),
                              str(cnf_path), str(proof_path)],
                             capture_output=True, text=True)
        print(res.stdout.strip())
        print(f"rup_check.py: exit {res.returncode} "
              f"({time.time() - t0:.1f}s)")
        return res.returncode
    print(f"proof too large for rup_check.py ({lemmas} lemmas); install "
          f"drat-trim (github.com/marijnheule/drat-trim) and rerun with "
          f"--checker drat-trim")
    return 2


if __name__ == "__main__":
    sys.exit(main())
