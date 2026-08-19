"""One-off cross-check: does a MIP solver's LP bound do any better?

The CP-SAT lower bound sits near zero at 400² because the half-reified
stability constraints vanish at fractional values. Indicator constraints
in a MIP have exactly the same weakness, so the expectation is a root
bound of ~0 here too — this script exists to record that negative result
with numbers rather than assert it.

    python experiments/beyond_tiles/gurobi_check.py --size 200 --time 600
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--time", type=float, default=600.0)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    import gurobipy as gp
    from gurobipy import GRB

    from beyond_tiles.bench import marilyn_inputs
    from beyond_tiles.targets import cell_targets, window_slices, window_targets

    grey, free = marilyn_inputs(args.size)
    cell_t = cell_targets(grey, 0.45)
    windows = window_slices(grey.shape, 8, 8)
    targets, kept = window_targets(cell_t, free, windows)
    h, w = grey.shape

    t0 = time.perf_counter()
    m = gp.Model("still_image")
    m.Params.TimeLimit = args.time
    m.Params.Threads = args.threads
    x = m.addVars(h + 2, w + 2, vtype=GRB.BINARY)

    fixed_dead = np.ones((h + 2, w + 2), dtype=bool)
    fixed_dead[1:-1, 1:-1] = ~free
    for i in range(h + 2):
        for j in range(w + 2):
            if fixed_dead[i, j]:
                x[i, j].ub = 0
            nbrs = [
                x[i + di, j + dj]
                for di in (-1, 0, 1)
                for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)
                and 0 <= i + di < h + 2
                and 0 <= j + dj < w + 2
                and not fixed_dead[i + di, j + dj]
            ]
            s = gp.quicksum(nbrs)
            if fixed_dead[i, j]:
                if nbrs:
                    # no-birth: sum != 3 via indicator on an aux binary
                    z = m.addVar(vtype=GRB.BINARY)
                    m.addGenConstrIndicator(z, True, s <= 2)
                    m.addGenConstrIndicator(z, False, s >= 4)
                continue
            m.addGenConstrIndicator(x[i, j], True, s >= 2)
            m.addGenConstrIndicator(x[i, j], True, s <= 3)
            z = m.addVar(vtype=GRB.BINARY)  # dead branch: sum <= 2 or >= 4
            m.addGenConstrIndicator(x[i, j], False, s - 6 * z <= 2)
            m.addGenConstrIndicator(x[i, j], False, s + 6 * (1 - z) >= 4)

    devs = []
    for t, (si, sj) in zip(targets, kept):
        cells = [
            x[i + 1, j + 1]
            for i in range(si.start, si.stop)
            for j in range(sj.start, sj.stop)
            if free[i, j]
        ]
        live = gp.quicksum(cells)
        dev = m.addVar(lb=0.0)
        m.addConstr(dev >= live - float(t))
        m.addConstr(dev >= float(t) - live)
        devs.append(dev)
    m.setObjective(gp.quicksum(devs), GRB.MINIMIZE)
    build_s = time.perf_counter() - t0

    m.optimize()
    out = {
        "size": args.size,
        "build_time_s": build_s,
        "status": int(m.Status),
        "objective": m.ObjVal if m.SolCount else None,
        "best_bound": m.ObjBound,
        "runtime_s": m.Runtime,
        "gap": m.MIPGap if m.SolCount else None,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
