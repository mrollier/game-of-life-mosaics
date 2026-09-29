"""Jury prints (A3, or A2 from the same solve): the thesis cover's Paraty
Mirim in more detail, the sea packed with the cover's tiles, the land one
free-form still life whose density rises with the altitude.

**Elevation.** AWS's open Terrarium tiles (elevation-tiles-prod, zoom 14,
about 9 m per pixel here; SRTM on land), fetched into `--dem` with curl.
The cover's silhouette c (studies/cover/assets/silhouettes) is registered
on the tiles' own coastline, centred near the Pico do Pão de Açúcar
(-23.264146, -44.615687), and the page's window is taken around it, so the
print shows the cover's framing with the elevation model's coast.

**Canvas.** A3 portrait plus 3 mm bleed, cells of 1/60 inch: 720 x 1008
cells (multiples of 6 for the pond lattice and of 8 for the free-form
windows). The same pattern prints on A2 with cells of 0.60 mm.

**One still life.** The sea tiles are packed first (the cover's packer,
padded and cut); the land's free cells are those at least three cells from
any tile cell, so two dead lines always separate the two, and the land is
solved with `freeform.solve_poster`.

**The print** (the user's pick): both land and sea packed with tiles,
their level rising with the altitude and with the distance from the
shore; every level in one colour of its family (denim steps at sea,
terracotta on land); the relief smoothed a little, pale seams, a gentle
sea; tiling T2 "calm" (12 tiles per level, little jitter). The free-form
land (`solve`, `render`) and the other variants are kept for comparison.

    python studies/postcards/jury.py dem --dem DIR          # stitch the tiles
    python studies/postcards/jury.py register --dem DIR     # fit the silhouette
    python studies/postcards/jury.py refine --dem DIR       # polish the fit on the hills
    python studies/postcards/jury.py layout --dem DIR       # land, altitude, the cover's sea tiles
    python studies/postcards/jury.py tilings t2-calm --profile ICC   # the print, about 2 min
    python studies/postcards/jury.py solve                  # free-form land instead (about an hour)
    python studies/postcards/jury.py render --profile ICC   # its PDFs
"""

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

import postcards as P  # noqa: E402
from common import log  # noqa: E402

C = P.C
Z = 14
PEAK = (-23.264146, -44.615687)
A3 = (297.0, 420.0)
A2 = (420.0, 594.0)
OUT = HERE / "output" / "jury"


def global_px(lat, lon, z=Z):
    """Web Mercator pixel coordinates (x, y) at zoom z."""
    n = 256 * 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def cmd_dem(dem: Path):
    """Stitch the downloaded tiles t_{x}_{y}.png into one elevation grid (m)."""
    tiles = {tuple(map(int, f.stem.split("_")[1:])): f for f in dem.glob("t_*.png")}
    xs, ys = sorted({k[0] for k in tiles}), sorted({k[1] for k in tiles})
    grid = np.zeros((256 * len(ys), 256 * len(xs)), np.float32)
    for (tx, ty), f in tiles.items():
        a = np.asarray(Image.open(f).convert("RGB"), np.float64)
        e = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
        j, i = xs.index(tx), ys.index(ty)
        grid[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] = e
    np.savez_compressed(dem / "dem.npz", elev=grid, x0=xs[0] * 256, y0=ys[0] * 256)
    log(f"dem {grid.shape}, {grid.min():.0f} to {grid.max():.0f} m")


def load_dem(dem: Path):
    d = np.load(dem / "dem.npz")
    return d["elev"], float(d["x0"]), float(d["y0"])


def sample(elev, x0, y0, cx, cy, w, shape):
    """Elevation on a (rows, cols) grid covering a window of width `w`
    DEM pixels centred on global pixel (cx, cy), bilinear."""
    rows, cols = shape
    h = w * rows / cols
    gy = cy - h / 2 + (np.arange(rows) + 0.5) * h / rows - y0
    gx = cx - w / 2 + (np.arange(cols) + 0.5) * w / cols - x0
    yy, xx = np.meshgrid(gy - 0.5, gx - 0.5, indexing="ij")
    return ndi.map_coordinates(elev, [yy, xx], order=1, mode="nearest")


def cmd_register(dem: Path):
    """Scale and shift of silhouette c on the elevation model's coastline:
    the window (centre, width in DEM pixels) where the two land masks
    overlap best (intersection over union)."""
    elev, x0, y0 = load_dem(dem)
    rgb = np.asarray(Image.open(C.SILHOUETTES / "paraty-c.png").convert("RGB"), float)
    small = (192, 128)
    sil = np.asarray(Image.fromarray((rgb[..., 1] - rgb[..., 2]).astype(np.float32), "F")
                     .resize(small[::-1], Image.BOX)) > 0
    px, py = global_px(*PEAK)
    best = (0, None)
    for w in np.geomspace(300, 2400, 40):
        for dx in np.linspace(-0.2, 0.2, 17) * w:
            for dy in np.linspace(-0.2, 0.2, 17) * w:
                land = sample(elev, x0, y0, px + dx, py + dy, w, small) > 0.5
                iou = (land & sil).sum() / max(1, (land | sil).sum())
                if iou > best[0]:
                    best = (iou, (px + dx, py + dy, w))
    iou, (cx, cy, w) = best
    # refine around the best
    for w2 in np.linspace(w * 0.93, w * 1.07, 15):
        for dx in np.linspace(-0.025, 0.025, 11) * w:
            for dy in np.linspace(-0.025, 0.025, 11) * w:
                land = sample(elev, x0, y0, cx + dx, cy + dy, w2, small) > 0.5
                v = (land & sil).sum() / max(1, (land | sil).sum())
                if v > best[0]:
                    best = (v, (cx + dx, cy + dy, w2))
    iou, (cx, cy, w) = best
    m_per_px = 156543.03392 * math.cos(math.radians(PEAK[0])) / 2 ** Z
    reg = dict(cx=cx, cy=cy, w=w, iou=iou, width_km=w * m_per_px / 1000,
               offset_px=(cx - px, cy - py))
    (dem / "register.json").write_text(json.dumps(reg, indent=1))
    land = sample(elev, x0, y0, cx, cy, w, (384, 256)) > 0.5
    sil_big = np.asarray(Image.fromarray((rgb[..., 1] - rgb[..., 2]).astype(np.float32), "F")
                         .resize((256, 384), Image.BOX)) > 0
    over = np.zeros((384, 256, 3), np.uint8) + 255
    over[land & sil_big] = (90, 150, 90)
    over[land & ~sil_big] = (220, 60, 60)
    over[~land & sil_big] = (60, 60, 220)
    Image.fromarray(over).resize((512, 768), Image.NEAREST).save(dem / "register.png")
    log(f"registered: IoU {iou:.3f}, window {reg['width_km']:.2f} km wide, "
        f"centre {cx - px:+.0f}, {cy - py:+.0f} px from the peak")


def refine(dem: Path, shape=(384, 256)):
    """Polish the registration on the hills: the SRTM land is clean but the
    blended bathymetry puts false shallows all over the sea, so the fit
    rewards hills (over 25 m) on the silhouette's land and penalises hills
    in its sea."""
    elev, x0, y0 = load_dem(dem)
    reg = json.loads((dem / "register.json").read_text())
    rgb = np.asarray(Image.open(C.SILHOUETTES / "paraty-c.png").convert("RGB"), float)
    sil = np.asarray(Image.fromarray((rgb[..., 1] - rgb[..., 2]).astype(np.float32), "F")
                     .resize(shape[::-1], Image.BOX)) > 0

    def score(cx, cy, w):
        hill = sample(elev, x0, y0, cx, cy, w, shape) > 25
        n = max(1, hill.sum())
        return (hill & sil).sum() / n - 2 * (hill & ~sil).sum() / n + 0.5 * (hill & sil).sum() / sil.sum()

    best = (score(reg["cx"], reg["cy"], reg["w"]), (reg["cx"], reg["cy"], reg["w"]))
    for _ in range(2):
        cx, cy, w = best[1]
        for w2 in np.linspace(w * 0.9, w * 1.1, 21):
            for dx in np.linspace(-0.05, 0.05, 21) * w:
                for dy in np.linspace(-0.05, 0.05, 21) * w:
                    v = score(cx + dx, cy + dy, w2)
                    if v > best[0]:
                        best = (v, (cx + dx, cy + dy, w2))
    cx, cy, w = best[1]
    reg.update(cx=cx, cy=cy, w=w, hill_score=best[0], width_km=w * reg["width_km"] / reg["w"])
    (dem / "register.json").write_text(json.dumps(reg, indent=1))
    log(f"refined: {reg['width_km']:.2f} km wide")


def a3():
    P.set_canvas(60, trim=A3, mult=24)          # 720 x 1008 cells


def cmd_layout(dem: Path, seed=96):
    """Land (the cover's silhouette c), elevation on it, and the sea tiles."""
    a3()
    H, W = P.H, P.W
    elev, x0, y0 = load_dem(dem)
    reg = json.loads((dem / "register.json").read_text())
    rgb = np.asarray(Image.open(C.SILHOUETTES / "paraty-c.png").convert("RGB"), float)
    g = rgb[..., 1] - rgb[..., 2]
    ch = g.shape[1] * H / W                      # the canvas's share of the silhouette's height
    t = (g.shape[0] - ch) / 2
    land = np.asarray(Image.fromarray(g.astype(np.float32), "F")
                      .resize((W, H), Image.BOX, box=(0, t, g.shape[1], t + ch))) > 0
    alt = sample(elev, x0, y0, reg["cx"], reg["cy"], reg["w"], (H, W))
    alt = np.where(land, ndi.gaussian_filter(np.clip(alt, 0, None), 1.0), 0)
    p = P.Packed(~land, seed)
    free = land & ~ndi.binary_dilation(p.live > 0, np.ones((3, 3), bool), iterations=2)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "layout.npz", land=land, alt=alt.astype(np.float32), free=free,
                        live=p.live, tile=p.tile, labels=p.labels)
    log(f"layout {W}x{H}: land {land.mean():.0%}, free {free.mean():.0%}, altitude to "
        f"{alt.max():.0f} m, tiles {p.placed}")


def target(alt, free, lo=0.03, hi=0.31, gamma=0.7, d_max=0.40):
    """Grey image for solve_poster: density from `lo` at sea level to `hi`
    on the highest ground (99.5th percentile), 255 where nothing lives."""
    top = np.quantile(alt[free], 0.995)
    d = lo + (hi - lo) * np.clip(alt / top, 0, 1) ** gamma
    return np.where(free, np.round(255 * (1 - d / d_max)), 255).astype(np.uint8)


def cmd_solve(procs=4):
    from gol_mosaics.freeform.poster import PosterConfig, solve_poster
    d = np.load(OUT / "layout.npz")
    grey = target(d["alt"], d["free"])
    cfg = PosterConfig(d_max=0.40, dither="fs", strip_procs=procs, strip_workers=2, isolate=True,
                       polish_procs=3, strip_time=240.0, polish_budget=1500.0, seam_budget=600.0)
    res = solve_poster(grey, d["free"], cfg, out=OUT / "solve", resume=True, log=log)
    np.savez_compressed(OUT / "land.npz", live=res.pattern)
    log(f"land solved: deviation {res.report.get('deviation')}")


INKS = {  # the land's live cells
    "denim": (0.39, 0.09, 260),                  # the ramp's darkest step, the sea's family
    "terracotta": (0.50, 0.13, 40),
}


def check(live, what):
    """Count the cells that break the still life, and those of them away
    from the canvas edge (where the cut tiles are)."""
    g = np.pad(live.astype(np.int16), 1)
    n = sum(np.roll(np.roll(g, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)[1:-1, 1:-1]
    bad = (live & ~((n == 2) | (n == 3))) | (~live & (n == 3))
    inner = np.zeros_like(bad)
    inner[3:-3, 3:-3] = True
    log(f"{what}: {int(live.sum())} live cells; cells breaking the still life "
        f"{int(bad.sum())}, of which away from the cut edge {int((bad & inner).sum())}")


def write_print(ls, stem):
    """A 600 dpi PNG at A3, a small preview, and vector PDFs at A3 and A2
    (the same cells, 0.42 and 0.60 mm)."""
    import matplotlib.pyplot as plt
    a3()
    shape = (P.H, P.W)
    img = P.save_png(ls, stem.with_suffix(".png"))
    P.preview(img, stem.parent / f"{stem.name}_preview.png", scale=0.12)
    for name, trim, cpi in (("A3", A3, 60), ("A2", A2, 60 / math.sqrt(2))):
        P.set_canvas(cpi, trim=trim, mult=24)
        assert (P.H, P.W) == shape, (P.W, P.H)
        fig, ax = P.new_page()
        P.draw_layers(ax, ls)
        fig.savefig(stem.parent / f"{stem.name}_{name}.pdf")
        plt.close(fig)
        a3()
    log(f"{stem.name}: {len(ls)} colours, pdfs written")


def cmd_render(inks=tuple(INKS)):
    """The sea as on the cover (denim on apricot, COAST steps), the land's
    free-form cells in one ink on the same apricot."""
    P.C.PROFILE = P.PROFILE
    a3()
    d = np.load(OUT / "layout.npz")
    land = np.load(OUT / "land.npz")["live"] > 0
    check(land | (d["live"] > 0), "free-form land + tiled sea")
    sea = C.layers(d["labels"], P.PAL, "denim", "apricot", C.COAST)
    for ink in inks:
        col = P.safe([P.oklch_hex(*INKS[ink])])[0]
        write_print(P.compress(sea + [(col, land)]), OUT / f"jury-paraty_{ink}")


def cmd_tiles(seed=97, gamma=0.7):
    """No free-form: the land packed with tiles too, their level (and so
    their size and darkness) rising with the altitude, in the cover's
    terracotta ramp, as the sea's rises with the distance from the shore."""
    P.C.PROFILE = P.PROFILE
    a3()
    d = np.load(OUT / "layout.npz")
    land, alt = d["land"], d["alt"]
    top = np.quantile(alt[land], 0.995)
    Z = 1.6 + 5.6 * np.clip(alt / top, 0, 1) ** gamma
    p = P.Packed(land, seed, Z=Z)
    np.savez_compressed(OUT / "land_tiles.npz", live=p.live, tile=p.tile, labels=p.labels)
    check((p.live > 0) | (d["live"] > 0), "tiled land + tiled sea")
    sea = C.layers(d["labels"], P.PAL, "denim", "apricot", C.COAST)
    earth = C.layers(p.labels, P.PAL, "terracotta", "apricot", C.COAST)[1:]
    write_print(P.compress(sea + earth), OUT / "jury-paraty_tiles")


LEVEL_COLOURS = {  # one colour per level 1..7, land and sea alike (OKLCH)
    "ramp": [(0.85, 0.09, 70), (0.78, 0.11, 52), (0.70, 0.13, 35), (0.61, 0.13, 12),
             (0.52, 0.11, 342), (0.44, 0.10, 295), (0.36, 0.10, 262)],
    "legend": [(0.86, 0.15, 95), (0.76, 0.15, 60), (0.64, 0.17, 30), (0.60, 0.17, 350),
               (0.55, 0.14, 305), (0.55, 0.13, 250), (0.62, 0.11, 190)],
}


def level_layers(labels, colours, ink=0.10):
    """Paint layers colouring every tile by its own level (small tiles in
    holes too), its live cells a tone darker; the seams stay field."""
    lab = labels.astype(int)
    lvl = np.where(lab < 20, lab, np.where(lab < 40, lab - 20, 0))
    small = np.where((lab >= 40) & (lab < 100), (lab - 40) % 10, np.where((lab >= 140) & (lab < 200), (lab - 140) % 10, 0))
    lvl = np.where(small > 0, small, lvl)
    live = ((lab >= 1) & (lab <= 7)) | ((lab >= 40) & (lab < 100))
    inks = P.safe([P.tone(c, ink) for c in colours])
    return ([(colours[L - 1], lvl == L) for L in range(1, 8)] +
            [(inks[L - 1], (lvl == L) & live) for L in range(1, 8)])


def cmd_levels(names=tuple(LEVEL_COLOURS)):
    """The tiles-only print (the same tiles) with one fixed colour per level."""
    P.C.PROFILE = P.PROFILE
    a3()
    sea = np.load(OUT / "layout.npz")["labels"]
    land = np.load(OUT / "land_tiles.npz")["labels"]
    field = P.PAL["fields"]["apricot"]["hex"]
    for name in names:
        cols = P.safe([P.oklch_hex(*c) for c in LEVEL_COLOURS[name]])
        log(f"{name}: {' '.join(cols)}")
        ls = [(field, np.ones(sea.shape, bool))] + level_layers(sea, cols) + level_layers(land, cols)
        write_print(P.compress(ls), OUT / f"jury-paraty_levels-{name}")


def seven_steps(ramp, first=3):
    """Seven distinct colours, one per level, spread evenly (in OKLCH) over
    the ramp's steps `first`..7: the cover's COAST range, without levels
    sharing a colour."""
    pts = np.array([C.oklch(C._hex(h)) for h in P.PAL["ramps"][ramp]["grounds"][first - 1:]])
    t = np.linspace(0, len(pts) - 1, 7)
    cols = [P.oklch_hex(*(pts[int(np.floor(x))] + (x - np.floor(x)) *
                          (pts[min(int(np.floor(x)) + 1, len(pts) - 1)] - pts[int(np.floor(x))])))
            for x in t]
    return P.safe(cols)


def cmd_fixed():
    """The tiles-only print with every level in one colour of its family:
    denim steps in the sea, terracotta steps on land, small tiles in holes
    in their own level's colour."""
    P.C.PROFILE = P.PROFILE
    a3()
    sea = np.load(OUT / "layout.npz")["labels"]
    land = np.load(OUT / "land_tiles.npz")["labels"]
    blue, earth = seven_steps("denim"), seven_steps("terracotta")
    log(f"sea  {' '.join(blue)}\nland {' '.join(earth)}")
    ls = ([(P.PAL["fields"]["apricot"]["hex"], np.ones(sea.shape, bool))]
          + level_layers(sea, blue, ink=0.09) + level_layers(land, earth, ink=0.09))
    write_print(P.compress(ls), OUT / "jury-paraty_tiles-fixed")


VARIANTS = {  # name: changes from option 1 ("fixed")
    "bolder": dict(smooth=6.0, gamma=0.6),                  # more big tiles on the massif and ridges
    "finer": dict(land_max=4.4, gamma=1.0),                 # land capped near level 5
    "crisper": dict(seam="#FFF3E2", ink=0.15),              # pale seams, stronger tile patterns
    "wider": dict(first=2),                                 # colour ramps from one step lighter
    "gentler": dict(sea_scale=15.0),                        # a wide pale band along the coast
}


BASE = dict(smooth=0.0, gamma=0.7, land_max=5.6, sea_scale=10.0, first=3, seam=None, ink=0.09,
            seeds=(96, 97), pool=(48, (0.3, 0.8)), jitter=0.2, noise=0.0, reach=C.REACH)
STYLE = dict(VARIANTS["bolder"], **VARIANTS["crisper"], **VARIANTS["gentler"])   # the user's 1 + 3 + 5
TILINGS = {  # name: packing settings on top of STYLE, from ordered to wild
    "final": {},                                          # option 1's seeds
    "t1-ordered": dict(seeds=(201, 202), pool=(3, (0.45, 0.6)), jitter=0.0, noise=0.0, reach=0.3),
    "t2-calm": dict(seeds=(203, 204), pool=(12, (0.35, 0.75)), jitter=0.1, noise=0.1, reach=0.5),
    "t3-balanced": dict(seeds=(205, 206), jitter=0.2, noise=0.2, reach=0.6),
    "t4-lively": dict(seeds=(207, 208), pool=(48, (0.15, 0.9)), jitter=0.4, noise=0.4, reach=0.8),
    "t5-wild": dict(seeds=(209, 210), pool=(48, (0.05, 0.97)), jitter=0.8, noise=0.7, reach=1.0),
}


def make(name, v):
    """One tile print with settings `v` (see BASE): A3 PDF, 600 dpi PNG,
    a preview and a close-up."""
    import matplotlib.pyplot as plt
    P.C.PROFILE = P.PROFILE
    a3()
    d = np.load(OUT / "layout.npz")
    land, alt = d["land"], d["alt"]
    C._POOLS.clear()                                  # tile pools: how many tiles per level, which densities
    C.tile_pool.__defaults__ = (v["pool"][0], v["pool"][1])
    try:
        a = ndi.gaussian_filter(alt, v["smooth"]) if v["smooth"] else alt
        top = np.quantile(a[land], 0.995)
        Z = 1.6 + v["land_max"] * np.clip(a / top, 0, 1) ** v["gamma"]
        if v["noise"]:
            n = ndi.gaussian_filter(np.random.default_rng(v["seeds"][1]).normal(size=land.shape), 10)
            Z = Z + v["noise"] * n / n.std()
        pl = P.Packed(land, v["seeds"][1], Z=Z, reach=v["reach"])
        ps = P.Packed(~land, v["seeds"][0], reach=v["reach"], scale=v["sea_scale"], jitter=v["jitter"])
    finally:
        C._POOLS.clear()
        C.tile_pool.__defaults__ = (48, (0.3, 0.8))
    check((pl.live > 0) | (ps.live > 0), name)
    seam = v["seam"] or P.PAL["fields"]["apricot"]["hex"]
    blue, earth = seven_steps("denim", v["first"]), seven_steps("terracotta", v["first"])
    ls = P.compress([(seam, np.ones(land.shape, bool))]      # the seams: every cell outside a tile
                    + level_layers(ps.labels, blue, ink=v["ink"]) + level_layers(pl.labels, earth, ink=v["ink"]))
    stem = OUT / f"jury-paraty_{name}"
    img = P.save_png(ls, stem.with_suffix(".png"))
    P.preview(img, OUT / f"jury-paraty_{name}_preview.png", scale=0.12)
    cx, cy = round(0.58 * img.width), round(0.60 * img.height)
    img.crop((cx - 700, cy - 700, cx + 700, cy + 700)).save(OUT / f"jury-paraty_{name}_detail.png")
    fig, ax = P.new_page()
    P.draw_layers(ax, ls)
    fig.savefig(OUT / f"jury-paraty_{name}_A3.pdf")
    plt.close(fig)
    np.savez_compressed(OUT / f"tiling_{name}.npz", sea=ps.labels, land=pl.labels,
                        live=(pl.live > 0) | (ps.live > 0))
    log(f"{name}: sea {ps.placed}, land {pl.placed}")


def cmd_variants(names=tuple(VARIANTS)):
    """Slight variations of the fixed-colour tile print (round of 2026-09-29), A3 only."""
    for name in names:
        make(f"v-{name}", dict(BASE, **VARIANTS[name]))


def cmd_tilings(names=tuple(TILINGS)):
    """The chosen style (1 + 3 + 5) on option 1's tiling and on five new ones."""
    for name in names:
        make(name, dict(BASE, **STYLE, **TILINGS[name]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["dem", "register", "refine", "layout", "solve", "render", "tiles", "levels", "fixed", "variants", "tilings"])
    ap.add_argument("--dem", type=Path)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--profile", type=Path, help="Coated FOGRA39 ICC profile")
    ap.add_argument("names", nargs="*", help="tilings to render (default: all)")
    a = ap.parse_args()
    P.PROFILE = a.profile
    if a.cmd == "solve":
        cmd_solve(a.procs)
    elif a.cmd == "render":
        cmd_render()
    elif a.cmd == "tiles":
        cmd_tiles()
    elif a.cmd == "levels":
        cmd_levels()
    elif a.cmd == "fixed":
        cmd_fixed()
    elif a.cmd == "variants":
        cmd_variants()
    elif a.cmd == "tilings":
        cmd_tilings(a.names or tuple(TILINGS))
    else:
        {"dem": cmd_dem, "register": cmd_register, "refine": refine, "layout": cmd_layout}[a.cmd](a.dem)


if __name__ == "__main__":
    main()
