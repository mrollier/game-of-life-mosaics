"""CLI driver for the beyond-tiles spike experiments.

Run from anywhere:
    python experiments/beyond_tiles/run_experiment.py e2 --size 100 --time 600
Subcommands: e1 (ramps/uniforms), e2 (Marilyn headline), verify (recheck a
saved pattern). Later phases (e3/e4/e5) are added as the spike progresses.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from beyond_tiles.artifacts import save_run
from beyond_tiles.still_image import SpikeConfig, solve_image, verify_still_life
from beyond_tiles.targets import (
    equalize_grey,
    grey_and_mask_from_image,
    normalize_grey,
    ramp_grey,
    uniform_grey,
)

MARILYN = REPO / "input" / "images" / "marilyn.png"
RESULTS = Path(__file__).resolve().parent / "results"


def _cfg(args, **overrides) -> SpikeConfig:
    kw = dict(
        k=args.k,
        stride=args.stride,
        d_max=args.dmax,
        time_limit_s=args.time,
        workers=args.workers,
        seed=args.seed,
        mask_mode=args.mask_mode,
        snapshot_gap_s=getattr(args, "snapshot_gap", 0.0),
    )
    kw.update(overrides)
    return SpikeConfig(**kw)


def _run_one(name: str, grey, free, cfg, outbase: Path) -> dict:
    result = solve_image(grey, free, cfg)
    metrics = save_run(outbase / name, result, grey, free)
    dev = metrics["deviation"]
    print(
        f"{name}: {result.status} obj={result.objective} "
        f"wall={result.wall_time_s:.1f}s mad={dev['mad']:.4f} "
        f"pearson={dev['pearson']:.3f} verify={metrics['verify']}",
        flush=True,
    )
    return metrics


def cmd_e1(args) -> None:
    out = RESULTS / "e1"
    free = np.ones((args.size, args.size), dtype=bool)
    for d in (0.1, 0.2, 0.3, 0.4, 0.5):
        grey_val = int(round(255 * (1 - min(d / args.dmax, 1.0))))
        cfg = _cfg(args, d_max=max(args.dmax, d))
        _run_one(f"uniform_d{d:.1f}", uniform_grey(args.size, grey_val), free, cfg, out)
    _run_one("ramp_linear", ramp_grey(args.size), free, _cfg(args), out)
    yy, xx = np.mgrid[0 : args.size, 0 : args.size]
    r = np.hypot(yy - args.size / 2, xx - args.size / 2)
    radial = np.clip(255 * r / r.max(), 0, 255).astype(np.uint8)
    _run_one("ramp_radial", radial, free, _cfg(args), out)


def _marilyn(args):
    grey, free = grey_and_mask_from_image(MARILYN, args.size)
    if args.tone == "norm":
        grey = normalize_grey(grey, free)
    elif args.tone == "eq":
        grey = equalize_grey(grey, free)
    return grey, free


def cmd_e2(args) -> None:
    grey, free = _marilyn(args)
    name = (
        f"marilyn_{args.size}_s{args.seed}_{args.mask_mode}_{args.tone}"
        f"_k{args.k}s{args.stride}"
    )
    _run_one(name, grey, free, _cfg(args), RESULTS / "e2")


def cmd_e3(args) -> None:
    """One-factor-at-a-time knob study around the defaults, short budget."""
    grey, free = _marilyn(args)
    out = RESULTS / "e3"
    variants = [
        ("default_s0", {}),
        ("seed1", {"seed": 1}),
        ("seed2", {"seed": 2}),
        ("k6", {"k": 6, "stride": 3}),
        ("k12", {"k": 12, "stride": 6}),
        ("stride_full", {"stride": args.k}),
        ("dmax40", {"d_max": 0.40}),
        ("dmax50", {"d_max": 0.50}),
    ]
    for name, overrides in variants:
        _run_one(name, grey, free, _cfg(args, **overrides), out)


def cmd_e4(args) -> None:
    for size in (150, 200):
        sized = argparse.Namespace(**{**vars(args), "size": size})
        grey, free = _marilyn(sized)
        _run_one(f"marilyn_{size}", grey, free, _cfg(args), RESULTS / "e4")


def cmd_e5(args) -> None:
    """Texture comparison: free-form solve vs a square-tile mosaic."""
    from PIL import Image

    from gol_mosaics.mosaic import MosaicGenerator

    from beyond_tiles.metrics import motif_stats, spectral, tile_db_overlap

    free_form = np.load(Path(args.pattern))
    gen = MosaicGenerator(
        level=3, grid_size=args.tile_grid, tile_shape="square"
    )
    _, tile_mosaic, _ = gen.generate_from_pil(
        Image.open(MARILYN), no_eca=True, return_arrays=True, seed=0
    )

    report = {}
    for name, pat in [("free_form", free_form), ("tile_mosaic", tile_mosaic)]:
        report[name] = {
            "shape": list(pat.shape),
            "motif_4": motif_stats(pat, block=4),
            "motif_6": motif_stats(pat, block=6),
            "tile_db_overlap_L1": tile_db_overlap(pat, level=1),
        }
    out = RESULTS / "e5"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "texture.json", "w") as fh:
        json.dump(report, fh, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, pat in [("free_form", free_form), ("tile_mosaic", tile_mosaic)]:
        spec = spectral(pat)
        ax.semilogy(spec["radius"], np.array(spec["power"]) + 1e-9, label=name)
    ax.set_xlabel("spatial frequency (radial)")
    ax.set_ylabel("power")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "spectra.png", dpi=150)
    np.save(out / "tile_mosaic.npy", tile_mosaic)
    print(json.dumps(report, indent=2))


def cmd_e6(args) -> None:
    """Convergence movies: re-solve at three sizes, keeping the incumbents.

    The snapshot gap is scaled to the run length so every movie ends up with
    roughly a hundred frames spread evenly over its wall time.
    """
    out = RESULTS / "e6"
    plan = [(100, 60.0, 0.25), (200, 300.0, 1.0), (400, 2400.0, 10.0)]
    if args.only:
        plan = [row for row in plan if row[0] in args.only]
    for size, time_limit, gap in plan:
        sized = argparse.Namespace(**{**vars(args), "size": size})
        grey, free = _marilyn(sized)
        cfg = _cfg(
            args,
            k=8,
            stride=8,
            time_limit_s=time_limit,
            snapshot_gap_s=gap,
        )
        metrics = _run_one(f"marilyn_{size}", grey, free, cfg, out)
        print(f"  snapshots: {(out / f'marilyn_{size}' / 'snapshots.npz').exists()}", flush=True)
        del metrics


def cmd_gif(args) -> None:
    from beyond_tiles.animate import movie_from_run

    movie_from_run(
        args.run_dir,
        max_frames=args.max_frames,
        pacing=args.pacing,
        fps=args.fps,
        panel=not args.no_panel,
    )


def cmd_verify(args) -> None:
    pattern = np.load(args.pattern)
    checks = verify_still_life(pattern)
    print(json.dumps({"pattern": str(args.pattern), **checks}))
    if not all(checks.values()):
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [
        ("e1", cmd_e1),
        ("e2", cmd_e2),
        ("e3", cmd_e3),
        ("e4", cmd_e4),
        ("e6", cmd_e6),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--snapshot-gap", type=float, default=0.0,
                       help="keep an incumbent pattern at most every N seconds")
        if name == "e6":
            p.add_argument("--only", type=int, nargs="*", default=None,
                           help="restrict to these canvas sizes")
        p.add_argument("--size", type=int, default=100)
        p.add_argument("--k", type=int, default=8)
        p.add_argument("--stride", type=int, default=4)
        p.add_argument("--dmax", type=float, default=0.45)
        p.add_argument("--time", type=float, default=600.0)
        p.add_argument("--workers", type=int, default=10)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--mask-mode", default="force_dead",
                       choices=["force_dead", "soft_zero", "none"])
        p.add_argument("--tone", default="eq", choices=["raw", "norm", "eq"])
        p.set_defaults(fn=fn)
    p5 = sub.add_parser("e5")
    p5.add_argument("pattern", help="pattern.npy of the free-form solve")
    p5.add_argument("--tile-grid", type=int, default=16)
    p5.set_defaults(fn=cmd_e5)
    pg = sub.add_parser("gif")
    pg.add_argument("run_dir", help="run directory holding snapshots.npz")
    pg.add_argument("--max-frames", type=int, default=80)
    pg.add_argument("--pacing", default="time", choices=["time", "index", "log"])
    pg.add_argument("--fps", type=float, default=8.0)
    pg.add_argument("--no-panel", action="store_true",
                    help="pattern only, without the convergence curve")
    pg.set_defaults(fn=cmd_gif)
    pv = sub.add_parser("verify")
    pv.add_argument("pattern")
    pv.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
