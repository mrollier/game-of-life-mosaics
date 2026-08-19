"""Benchmark harness for the optimization campaign.

Every claimed speed-up gets an A/B measurement here, against a fixed
protocol:

  * quick suite    — Marilyn 200², k=8 stride=8, tone=eq, 300 s wall,
                     seeds {0,1,2}; screening judge is the median
                     time-to-optimal, falling back to objective@budget.
  * decisive suite — Marilyn 400², 600 s, seeds {0,1}; judge is
                     objective@budget + mad_darkest_quartile + best_bound.

Raw `objective` is only comparable between runs with identical window
geometry, targets and slack; across configurations the judges are the
density-fidelity metrics (mad, mad_darkest_quartile, pearson) plus
time-to-optimal and best_bound.
"""

import dataclasses
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MARILYN = REPO / "input" / "images" / "marilyn.png"

# The judging columns of `compare`, in display order.
COLUMNS = [
    "status",
    "objective",
    "best_bound",
    "wall_time_s",
    "build_time_s",
    "time_to_first_s",
    "time_to_optimal_s",
    "mad",
    "mad_darkest_quartile",
    "pearson",
    "max_rss_mb",
]


@dataclass
class BenchCase:
    name: str
    size: int
    budget_s: float
    seed: int
    overrides: Dict[str, object] = field(default_factory=dict)


def parse_overrides(pairs: List[str]) -> Dict[str, object]:
    """"key=value" CLI strings -> typed SpikeConfig overrides."""
    out: Dict[str, object] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not _ or not key:
            raise ValueError(f"override must look like key=value, got {pair!r}")
        if raw.lower() in ("true", "false"):
            out[key] = raw.lower() == "true"
            continue
        for cast in (int, float):
            try:
                out[key] = cast(raw)
                break
            except ValueError:
                continue
        else:
            out[key] = raw
    return out


def derive_timings(
    obj_history: List[Tuple[float, int]], status: str, wall_time_s: float
) -> Dict[str, Optional[float]]:
    """First-incumbent and proven-optimal times from the objective log.

    CP-SAT returns the moment optimality is proven, so for an OPTIMAL run
    the proof time is the solve wall time itself; FEASIBLE runs get None.
    """
    return {
        "time_to_first_s": obj_history[0][0] if obj_history else None,
        "time_to_optimal_s": wall_time_s if status == "OPTIMAL" else None,
    }


def marilyn_inputs(size: int, tone: str = "eq"):
    """The campaign's fixed test image at a given canvas size."""
    from beyond_tiles.targets import (
        equalize_grey,
        grey_and_mask_from_image,
        normalize_grey,
    )

    grey, free = grey_and_mask_from_image(MARILYN, size)
    if tone == "norm":
        grey = normalize_grey(grey, free)
    elif tone == "eq":
        grey = equalize_grey(grey, free)
    return grey, free


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_case(case: BenchCase, outdir: Path, grey=None, free=None) -> dict:
    """Solve one case, persist it via save_run, extend metrics.json."""
    import ortools

    from beyond_tiles.artifacts import save_run
    from beyond_tiles.still_image import SpikeConfig, solve_image

    if grey is None:
        grey, free = marilyn_inputs(case.size)
    kw = dict(k=8, stride=8, time_limit_s=case.budget_s, seed=case.seed)
    kw.update(case.overrides)
    cfg = SpikeConfig(**kw)
    result = solve_image(grey, free, cfg)

    case_dir = Path(outdir) / case.name
    metrics = save_run(case_dir, result, grey, free)
    metrics.update(derive_timings(result.obj_history, result.status, result.wall_time_s))
    metrics["bench"] = {
        "case": dataclasses.asdict(case),
        "git_sha": _git_sha(),
        "ortools": ortools.__version__,
    }
    with open(case_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(
        f"{case.name}: {metrics['status']} obj={metrics['objective']} "
        f"bound={metrics['best_bound']} wall={metrics['wall_time_s']:.1f}s "
        f"build={metrics['build_time_s']:.1f}s "
        f"mad={metrics['deviation']['mad']:.4f}",
        flush=True,
    )
    return metrics


def quick_suite(
    overrides: Optional[Dict[str, object]] = None, seeds=(0, 1, 2)
) -> List[BenchCase]:
    """Screening: 200² at 300 s across seeds."""
    return [
        BenchCase(f"q200_s{s}", 200, 300.0, s, dict(overrides or {}))
        for s in seeds
    ]


def decisive_suite(
    overrides: Optional[Dict[str, object]] = None, seeds=(0, 1)
) -> List[BenchCase]:
    """Winners only: 400² at 600 s."""
    return [
        BenchCase(f"d400_s{s}", 400, 600.0, s, dict(overrides or {}))
        for s in seeds
    ]


def _flatten(metrics: dict) -> Dict[str, object]:
    row: Dict[str, object] = {}
    for key in COLUMNS:
        if key in metrics:
            row[key] = metrics[key]
        elif key in metrics.get("deviation", {}):
            row[key] = metrics["deviation"][key]
        else:
            row[key] = None
    return row


def compare(dirs: List[Path]) -> str:
    """Markdown table over every metrics.json below the given tag dirs."""
    rows = []
    for tag_dir in map(Path, dirs):
        for path in sorted(tag_dir.rglob("metrics.json")):
            metrics = json.loads(path.read_text())
            label = f"{tag_dir.name}/{path.parent.name}"
            rows.append((label, _flatten(metrics)))
    if not rows:
        return "no metrics.json found"

    header = ["run"] + COLUMNS
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for label, row in rows:
        cells = [label]
        for key in COLUMNS:
            v = row[key]
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def median_summary(tag_dir: Path) -> Dict[str, float]:
    """Median judge metrics across the cases of one tag (for A/B calls)."""
    tto, obj, mad = [], [], []
    for path in sorted(Path(tag_dir).rglob("metrics.json")):
        metrics = json.loads(path.read_text())
        if metrics.get("time_to_optimal_s") is not None:
            tto.append(metrics["time_to_optimal_s"])
        obj.append(metrics["objective"])
        mad.append(metrics["deviation"]["mad"])
    return {
        "n_cases": len(obj),
        "n_optimal": len(tto),
        "median_time_to_optimal_s": float(np.median(tto)) if tto else None,
        "median_objective": float(np.median(obj)) if obj else None,
        "median_mad": float(np.median(mad)) if mad else None,
    }
