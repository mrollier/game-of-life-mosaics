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

from beyond_tiles.artifacts import (
    ASSETS,
    load_snapshots,
    save_pattern_asset,
    save_snapshots,
)
from beyond_tiles.animate import select_frames

RESULTS = Path(__file__).resolve().parent / "results"
E2 = RESULTS / "e2"
E6 = RESULTS / "e6"
RUNS = [
    ("marilyn_200_s0_force_dead_eq_k8s8", "200", "optimal"),
    ("marilyn_400_s0_force_dead_eq_k8s8", "400", "feasible"),
]
MOVIES = [100, 200, 400]  # e6 runs whose incumbents feed the notebook filmstrip
MOVIE_FRAMES = 60


def _gzip_copy(src: Path, dst: Path) -> None:
    # mtime=0 keeps the bytes reproducible, so re-running this script does not
    # show up as a diff when nothing actually changed.
    with open(src, "rb") as fh, open(dst, "wb") as out:
        with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
            shutil.copyfileobj(fh, gz)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for run, size, tag in RUNS:
        src = E2 / run
        if not src.exists():
            raise SystemExit(f"missing run: {src} (re-run the e2 experiment first)")

        pattern = np.load(src / "pattern.npy")
        out = save_pattern_asset(ASSETS / f"marilyn_{size}_{tag}.npz", pattern)

        _gzip_copy(src / "convergence.csv", ASSETS / f"convergence_{size}.csv.gz")

        metrics = json.loads((src / "metrics.json").read_text())
        (ASSETS / f"metrics_{size}.json").write_text(json.dumps(metrics, indent=2))

        print(
            f"{run}: {pattern.shape} -> {out.name} "
            f"({out.stat().st_size / 1024:.1f} kB), "
            f"{metrics['status']} obj={metrics['objective']}"
        )

    for size in MOVIES:
        src = E6 / f"marilyn_{size}"
        if not src.exists():
            print(f"skipping movie assets for {size}²: no {src}")
            continue

        # Thin the incumbents out to a filmstrip-sized, evenly timed subset.
        snapshots = load_snapshots(src / "snapshots.npz")
        picks = select_frames(
            [t for t, _, _ in snapshots], MOVIE_FRAMES, pacing="time"
        )
        out = save_snapshots(
            ASSETS / f"snapshots_{size}.npz", [snapshots[i] for i in picks]
        )
        _gzip_copy(src / "convergence.csv", ASSETS / f"movie_curve_{size}.csv.gz")
        metrics = json.loads((src / "metrics.json").read_text())
        (ASSETS / f"movie_metrics_{size}.json").write_text(
            json.dumps(metrics, indent=2)
        )
        print(
            f"movie {size}²: {len(picks)} of {len(snapshots)} incumbents -> "
            f"{out.name} ({out.stat().st_size / 1024:.1f} kB), "
            f"{metrics['status']} obj={metrics['objective']}"
        )


if __name__ == "__main__":
    main()
