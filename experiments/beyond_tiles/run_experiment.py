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
        slack=getattr(args, "slack", 0),
        dither=getattr(args, "dither", "round"),
        edge_windows=getattr(args, "edge_windows", "clamp"),
        add_lb_subsolvers=getattr(args, "lb_subsolvers", False),
        num_violation_ls=getattr(args, "violation_ls", 0),
        symmetry_level=getattr(args, "symmetry_level", None),
        log_to=getattr(args, "log_to", None),
        hint_mode=getattr(args, "hint_mode", "none"),
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


E7_VARIANTS = [
    ("A_base", {}),
    ("B_slack1", {"slack": 1}),
    ("C_slack2", {"slack": 2}),
    ("D_dither", {"dither": "fs"}),
    ("E_slack1_dither", {"slack": 1, "dither": "fs"}),
    ("F_lbsub", {"add_lb_subsolvers": True}),
    ("G_vls3", {"num_violation_ls": 3}),
    ("H_sym0", {"symmetry_level": 0}),
]


def cmd_e7(args) -> None:
    """Stage-2 parameter study: quick suite per variant, tagged e7_<name>."""
    from beyond_tiles import bench

    variants = E7_VARIANTS
    if args.only:
        variants = [(n, ov) for n, ov in variants if n[0] in args.only]
    seeds = tuple(args.seeds) if args.seeds else (0, 1, 2)
    for name, overrides in variants:
        suite = (
            bench.quick_suite if args.suite == "quick" else bench.decisive_suite
        )
        for case in suite(overrides, seeds=seeds):
            bench.run_case(case, RESULTS / "bench" / f"e7_{name}")
    print(bench.compare(sorted((RESULTS / "bench").glob("e7_*"))))


def cmd_bench(args) -> None:
    """Fixed-protocol A/B benchmark runs (see bench.py for the protocol)."""
    from beyond_tiles import bench

    if args.compare:
        print(bench.compare([Path(p) for p in args.compare]))
        for p in args.compare:
            print(f"\n{p}: {json.dumps(bench.median_summary(Path(p)))}")
        return

    overrides = bench.parse_overrides(args.overrides or [])
    suite_fn = bench.quick_suite if args.suite == "quick" else bench.decisive_suite
    cases = suite_fn(overrides, seeds=tuple(args.seeds)) if args.seeds else suite_fn(overrides)
    outdir = RESULTS / "bench" / args.tag
    for case in cases:
        bench.run_case(case, outdir)
    print(json.dumps(bench.median_summary(outdir), indent=2))


def cmd_e9(args) -> None:
    """Strip decomposition at scale: solve (restriction) and/or bound."""
    from beyond_tiles.decompose import lower_bound_strips, plan_strips, solve_strips
    from beyond_tiles.metrics import deviation_stats
    from beyond_tiles.targets import cell_targets, window_slices

    grey, free = _marilyn(args)
    cfg = _cfg(args, k=8, stride=8)
    plan = plan_strips(args.size, cfg.k, args.strip_rows, args.gap)
    out = RESULTS / "e9" / f"marilyn_{args.size}_r{args.strip_rows}"
    out.mkdir(parents=True, exist_ok=True)
    report = {"plan": {"spans": plan.spans, "gap": plan.gap}}

    if args.mode in ("solve", "both"):
        solved = solve_strips(grey, free, cfg, plan, n_procs=args.procs)
        pattern = solved.pop("pattern")
        checks = verify_still_life(pattern)
        assert all(checks.values()), checks
        np.save(out / "pattern.npy", pattern)
        cell_t = cell_targets(grey, cfg.d_max)
        windows = window_slices(grey.shape, cfg.k, cfg.stride, edge=cfg.edge_windows)
        solved["deviation_vs_full_targets"] = deviation_stats(
            pattern, cell_t, free, windows
        )
        solved["verify"] = checks
        report["solve"] = solved
        print(
            f"solve: strips={len(plan.spans)} objective={solved['objective']} "
            f"wall={solved['wall_time_s']:.1f}s "
            f"mad={solved['deviation_vs_full_targets']['mad']:.4f} verify={checks}"
        )

    if args.mode in ("bound", "both"):
        bound = lower_bound_strips(grey, free, cfg, plan, n_procs=args.procs)
        report["bound"] = bound
        print(
            f"bound: LB={bound['lower_bound']} all_optimal={bound['all_optimal']} "
            f"wall={bound['wall_time_s']:.1f}s"
        )

    with open(out / "strips.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)


def cmd_e10(args) -> None:
    """Annealing chain: agar seed -> parallel tempering -> exact repair."""
    import time as time_mod

    from beyond_tiles.anneal import AnnealConfig, anneal, instability, kill_repair
    from beyond_tiles.lns import LnsConfig, improve, window_devs
    from beyond_tiles.seeds import best_seed
    from beyond_tiles.targets import cell_targets, window_slices, window_targets

    grey, free = _marilyn(args)
    cell_t = cell_targets(grey, 0.45)
    windows = window_slices(grey.shape, 8, 8)
    targets, kept = window_targets(cell_t, free, windows, dither=args.dither)

    seed, seed_obj = best_seed(free, kept, targets, slack=args.slack)
    print(f"seed objective: {seed_obj}", flush=True)
    cfg = AnnealConfig(
        sweeps=args.sweeps,
        replicas=args.replicas,
        lam=args.lam,
        slack=args.slack,
        seed=args.seed,
    )
    t0 = time_mod.perf_counter()
    pattern, info = anneal(seed, free, kept, targets, cfg)
    print(
        f"anneal: best E={info['best_energy']:.0f} "
        f"unstable={info['unstable_cells']} "
        f"{info['updates_per_s']:.2e} updates/s",
        flush=True,
    )
    repaired = kill_repair(pattern)
    assert instability(np.pad(repaired, 1)) == 0
    obj = int(window_devs(repaired, free, kept, targets, args.slack).sum())
    print(f"after repair: objective {obj}", flush=True)

    if args.lns_polish > 0:
        res = improve(
            repaired,
            free,
            kept,
            targets,
            LnsConfig(budget_s=args.lns_polish, slack=args.slack, seed=args.seed),
        )
        repaired, obj = res.pattern, res.objective
        print(f"after lns: objective {obj}", flush=True)

    checks = verify_still_life(repaired)
    assert all(checks.values()), checks
    out = RESULTS / "e10" / f"marilyn_{args.size}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pattern.npy", repaired)
    from beyond_tiles.metrics import deviation_stats

    stats = deviation_stats(repaired, cell_t, free, kept)
    report = {
        "seed_objective": seed_obj,
        "anneal": {k: v for k, v in info.items() if k != "history"},
        "objective_after_repair": obj,
        "deviation": stats,
        "verify": checks,
        "wall_time_s": time_mod.perf_counter() - t0,
        "config": vars(args),
    }
    with open(out / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ("objective_after_repair", "deviation")}, indent=2))


def cmd_lns(args) -> None:
    """Polish a saved run with rectangular LNS; writes RUN_DIR/lns/."""
    import dataclasses
    import resource

    from beyond_tiles import lns as lns_mod
    from beyond_tiles.still_image import SpikeConfig, SpikeResult
    from beyond_tiles.targets import cell_targets, window_slices, window_targets

    run_dir = Path(args.run_dir)
    pattern = np.load(run_dir / "pattern.npy")
    saved = json.loads((run_dir / "metrics.json").read_text())
    known = {f.name for f in dataclasses.fields(SpikeConfig)}
    cfg = SpikeConfig(**{k: v for k, v in saved["config"].items() if k in known})

    sized = argparse.Namespace(size=pattern.shape[0], tone=args.tone)
    grey, free = _marilyn(sized)
    cell_t = cell_targets(grey, cfg.d_max)
    if cfg.mask_mode == "none":
        free = np.ones_like(free, dtype=bool)
    elif cfg.mask_mode == "soft_zero":
        cell_t = np.where(free, cell_t, 0.0)
        free = np.ones_like(free, dtype=bool)
    windows = window_slices(grey.shape, cfg.k, cfg.stride, edge=cfg.edge_windows)
    targets, kept = window_targets(cell_t, free, windows, dither=cfg.dither)

    lcfg = lns_mod.LnsConfig(
        patch_windows=args.patch_windows,
        patch_time_s=args.patch_time,
        budget_s=args.budget,
        n_procs=args.procs,
        seed=cfg.seed,
        slack=cfg.slack,
    )
    before = int(
        lns_mod.window_devs(pattern, free, kept, targets, cfg.slack).sum()
    )
    res = lns_mod.improve(pattern, free, kept, targets, lcfg)
    out = SpikeResult(
        pattern=res.pattern,
        status="LNS",
        objective=res.objective,
        best_bound=saved.get("best_bound", 0),
        wall_time_s=res.obj_history[-1][0],
        obj_history=res.obj_history,
        max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        config=cfg,
        windows=kept,
        targets=targets,
    )
    metrics = save_run(run_dir / "lns", out, grey, free)
    print(
        f"lns: objective {before} -> {res.objective} in {res.rounds} rounds "
        f"({res.patches_improved}/{res.patches_solved} patches), "
        f"verify={metrics['verify']}"
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
        p.add_argument("--slack", type=int, default=0,
                       help="free deviation per window, in cells")
        p.add_argument("--dither", default="round", choices=["round", "fs"])
        p.add_argument("--edge-windows", default="clamp",
                       choices=["clamp", "partial"])
        p.add_argument("--lb-subsolvers", action="store_true",
                       help="schedule lb_tree_search + objective_lb_search")
        p.add_argument("--violation-ls", type=int, default=0,
                       help="Feasibility-Jump local-search workers")
        p.add_argument("--symmetry-level", type=int, default=None)
        p.add_argument("--log-to", default=None,
                       help="write the CP-SAT search log to this file")
        p.add_argument("--hint", dest="hint_mode", default="none",
                       choices=["none", "agar"],
                       help="warm-start the solve with a constructive seed")
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
    p7 = sub.add_parser("e7")
    p7.add_argument("--suite", default="quick", choices=["quick", "decisive"])
    p7.add_argument("--only", nargs="*", default=None, metavar="LETTER",
                    help="variant letters to run, e.g. --only B E")
    p7.add_argument("--seeds", type=int, nargs="*", default=None)
    p7.set_defaults(fn=cmd_e7)
    pb = sub.add_parser("bench")
    pb.add_argument("--suite", default="quick", choices=["quick", "decisive"])
    pb.add_argument("--tag", default="untagged",
                    help="results land in results/bench/<tag>/")
    pb.add_argument("--overrides", nargs="*", default=None, metavar="KEY=VAL",
                    help="SpikeConfig overrides, e.g. workers=4 d_max=0.4")
    pb.add_argument("--seeds", type=int, nargs="*", default=None)
    pb.add_argument("--compare", nargs="*", default=None, metavar="TAG_DIR",
                    help="print a markdown table over these tag dirs instead")
    pb.set_defaults(fn=cmd_bench)
    p10 = sub.add_parser("e10")
    p10.add_argument("--size", type=int, default=400)
    p10.add_argument("--sweeps", type=int, default=4000)
    p10.add_argument("--replicas", type=int, default=4)
    p10.add_argument("--lam", type=float, default=4.0)
    p10.add_argument("--slack", type=int, default=0)
    p10.add_argument("--seed", type=int, default=0)
    p10.add_argument("--dither", default="round", choices=["round", "fs"])
    p10.add_argument("--tone", default="eq", choices=["raw", "norm", "eq"])
    p10.add_argument("--lns-polish", type=float, default=0.0,
                     help="seconds of LNS after the repair pass")
    p10.set_defaults(fn=cmd_e10)
    p9 = sub.add_parser("e9")
    p9.add_argument("--size", type=int, default=400)
    p9.add_argument("--strip-rows", type=int, default=48)
    p9.add_argument("--gap", type=int, default=2)
    p9.add_argument("--mode", default="both", choices=["solve", "bound", "both"])
    p9.add_argument("--procs", type=int, default=4)
    p9.add_argument("--time", type=float, default=120.0,
                    help="per-strip time limit")
    p9.add_argument("--workers", type=int, default=2,
                    help="CP-SAT workers per strip (times --procs processes)")
    p9.add_argument("--seed", type=int, default=0)
    p9.add_argument("--k", type=int, default=8)
    p9.add_argument("--stride", type=int, default=8)
    p9.add_argument("--dmax", type=float, default=0.45)
    p9.add_argument("--mask-mode", default="force_dead",
                    choices=["force_dead", "soft_zero", "none"])
    p9.add_argument("--tone", default="eq", choices=["raw", "norm", "eq"])
    p9.add_argument("--slack", type=int, default=0)
    p9.add_argument("--dither", default="round", choices=["round", "fs"])
    p9.set_defaults(fn=cmd_e9)
    pl = sub.add_parser("lns")
    pl.add_argument("run_dir", help="saved run directory to polish")
    pl.add_argument("--budget", type=float, default=600.0)
    pl.add_argument("--patch-windows", type=int, default=5,
                    help="patch side length in windows")
    pl.add_argument("--patch-time", type=float, default=2.0)
    pl.add_argument("--procs", type=int, default=4)
    pl.add_argument("--tone", default="eq", choices=["raw", "norm", "eq"])
    pl.set_defaults(fn=cmd_lns)
    pv = sub.add_parser("verify")
    pv.add_argument("pattern")
    pv.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
