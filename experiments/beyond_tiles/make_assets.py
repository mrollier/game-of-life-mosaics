"""Extract the committed high-resolution assets from a local results tree.

`experiments/beyond_tiles/results/` is gitignored (renders, .cells files and
convergence logs add up), but the notebook needs the two headline solves. This
script distils them into ~60 kB of versioned files: bit-packed patterns,
gzipped convergence logs and the metrics dicts.

    python experiments/beyond_tiles/make_assets.py
"""

import gzip
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from beyond_tiles.artifacts import ASSETS, save_pattern_asset

RESULTS = Path(__file__).resolve().parent / "results" / "e2"
RUNS = [
    ("marilyn_200_s0_force_dead_eq_k8s8", "200", "optimal"),
    ("marilyn_400_s0_force_dead_eq_k8s8", "400", "feasible"),
]


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for run, size, tag in RUNS:
        src = RESULTS / run
        if not src.exists():
            raise SystemExit(f"missing run: {src} (re-run the e2 experiment first)")

        pattern = np.load(src / "pattern.npy")
        out = save_pattern_asset(ASSETS / f"marilyn_{size}_{tag}.npz", pattern)

        with open(src / "convergence.csv", "rb") as fh, gzip.open(
            ASSETS / f"convergence_{size}.csv.gz", "wb"
        ) as gz:
            shutil.copyfileobj(fh, gz)

        metrics = json.loads((src / "metrics.json").read_text())
        (ASSETS / f"metrics_{size}.json").write_text(json.dumps(metrics, indent=2))

        print(
            f"{run}: {pattern.shape} -> {out.name} "
            f"({out.stat().st_size / 1024:.1f} kB), "
            f"{metrics['status']} obj={metrics['objective']}"
        )


if __name__ == "__main__":
    main()
