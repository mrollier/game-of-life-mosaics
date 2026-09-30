"""Sacred-geometry posters (A2): images as gapless mosaics of tiles of levels 1-7.

Interpretations of photographs of sacred geometry (a Persian dome, three
Himalayan mandala thangkas, a panel of Persian tilework) as posters of Game
of Life still lifes, in the manner of the thesis cover and the Paraty jury
print: the picture is packed with the cover's tiles (studies/cover/cover.py),
levels 1 to 7, without a gap.

**Frame.** Each mosaic keeps its source's proportions and sits on an A2
sheet with a margin of paper around it. Tiles are seated only where they
lie wholly inside the mosaic's rectangle, so the sheet is one finite still
life: nothing is cut at the edge.

**Tracing.** Each source is reduced to a small palette of its own inks
(k-means in OKLab), and the inks' one-hot maps are carried onto the
poster's cells. The traced map is then made exactly symmetric: 4-fold with
mirrors (D4) inside a central square, mirrored outside it (the dome is a
kaleidoscope of one wedge of its own 16-fold pattern). The figures, which
are not symmetric, dissolve into colour fields.

**Levels by detail.** A level-L tile may sit wherever one ink covers at
least THETA[L] of its cells, so the big tiles fill the calm fields and the
small ones trace the busy parts. Levels 7 down to 2 are packed greedily
(each tile where its rim touches the most tiles, as on the cover), a whole
orbit at a time, so the layout is exactly as symmetric as the tracing;
every tile still draws its own still life. Every free pond vertex inside
the frame then gets a pond.

**Colour.** Every tile is one pane of the ink that covers most of it, its
live cells a tone darker. Palettes (PALETTES): the source's own inks, the
same made fresh, or the liquid-light dyes of the defence postcards, each
ink moved onto the dye nearest its hue at its lightness and lit from the
centre, every orbit of panes a little different. All colours are moved
into Coated FOGRA39.

**Canvas.** A2 plus 3 mm bleed, cells of 1/60 inch (0.42 mm): 1008 x 1422
cells (the dome, landscape: 1422 x 1008), the jury print's cell. A lead
line FRAME cells wide runs round each mosaic.

    python studies/sacred/sacred.py build [name ...] --profile ICC    # trace, pack, render
    python studies/sacred/sacred.py render [name ...] --profile ICC   # recolour saved packs
    python studies/sacred/sacred.py print --profile ICC               # the PRINTS: PDF, PNG

The print check needs pdftoppm (poppler) on the PATH.

The sources (assets/sources) are kept out of git: their rights are unknown.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", HERE.parent, HERE.parent / "cover", HERE.parent / "postcards"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

import postcards as P  # noqa: E402
from common import log  # noqa: E402

C = P.C
SOURCES = HERE / "assets" / "sources"
OUT = HERE / "output"
SHEETS = {"portrait": (420.0, 594.0), "landscape": (594.0, 420.0)}      # A2, mm
MARGIN_MM = 24.0                # paper round the mosaic, at its tighter side
TOP_SHARE = 0.45                # of the free height, above the mosaic
PAPER = "#FFFFFF"
FRAME = 2                       # cells of lead between the outermost tiles and the paper
M = C.MARGIN


# --- Colour spaces -------------------------------------------------------------------

def oklab(rgb):
    """sRGB in 0..1, shape (..., 3) -> OKLab."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lms = np.cbrt(c @ np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                                [0.2119034982, 0.6806995451, 0.1073969566],
                                [0.0883024619, 0.2817188376, 0.6299787005]]).T)
    return lms @ np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                           [1.9779984951, -2.4285922050, 0.4505937099],
                           [0.0259040371, 0.7827717662, -0.8086757660]]).T


def lch(lab):
    L, a, b = lab
    return float(L), float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)) % 360)


def lab_hex(lab, dL=0.0, gain=1.0):
    L, Cc, h = lch(lab)
    return C.from_oklch(L + dL, Cc * gain, h)


# --- Designs --------------------------------------------------------------------------
# Source geometry is in source pixels, (y, x). `centre` is the symmetry
# centre, `half` the half-width of the mosaic about it, `rows` its first and
# last row; `square` the half-size of the D4 square; `sym` the symmetry
# outside it (D1: left-right, D2: and up-down).

DESIGNS = {
    "mandala-blue": dict(src="mandala-blue", k=13, centre=(512.0, 442.75), half=435.0, rows=(8, 1015),
                         square=425.0, sym="D1", seam=("dark",)),
    "mandala-gold": dict(src="mandala-gold", k=13, centre=(327.0, 274.0), half=264.0, rows=(10, 630),
                         square=264.0, sym="D1", seam=("dark",)),
    "mandala-dark": dict(src="mandala-dark", k=12, centre=(474.0, 401.0), half=344.0, rows=(48, 990),
                         square=344.0, sym="D1", seam=("dark",)),
    # The dome is a kaleidoscope: one wedge of the photo between two
    # neighbouring mirror axes (source angles 8.94 + k 11.25 degrees, y
    # down), mirrored round the circle, a strap of the star upright. The
    # centre drifts down with the radius (the photo is a little off-axis).
    # Where the wedge leaves the photo, the photo's own top-left corner.
    "dome": dict(src="dome", kind="dome", k=14, centre=(268.0, 365.0), drift=0.05, wedge=211.44, flip=True,
                 half=363.0, rows=(0, 536), square=268.0, sym="D2", sheet="landscape",
                 seam=("ink", "#3FB6E0"), smooth=1.0, min_area=20, close="majority"),
    # The same, traced as loosely as the thangkas: calmer panels, bigger tiles.
    "dome-calm": dict(src="dome", kind="dome", k=14, centre=(268.0, 365.0), drift=0.05, wedge=211.44, flip=True,
                      half=363.0, rows=(0, 536), square=268.0, sym="D2", sheet="landscape",
                      seam=("ink", "#3FB6E0"), close="majority"),
    "tilework": dict(src="tilework", k=9, centre=(511.0, 407.0), half=407.0, rows=(0, 1022),
                     square=407.0, sym="D2", seam=("dark",)),
}
# The share of a level-L tile's cells that one ink must cover.
THETA = {7: 0.74, 6: 0.75, 5: 0.76, 4: 0.78, 3: 0.78, 2: 0.75}
SMOOTH = 2.5                    # cells: the soft ink map's blur before symmetrising
MIN_AREA = 60                   # cells: smaller patches of one ink join their surroundings


def layout(d):
    """Set the A2 canvas for the design and place the mosaic on it: the
    symmetry centre (Yc, Xc) and the mosaic's box (y0, y1, x0, x1) in cells,
    the scale `s` (cells per source pixel), the D4 square's half-size `R`,
    the `region` inside the box and the `inner` region the tiles keep to
    (FRAME cells in from the box: a lead frame)."""
    P.set_canvas(60, trim=SHEETS[d.get("sheet", "portrait")], mult=6)
    H, W = P.H, P.W
    cy, cx = d["centre"]
    r0, r1 = d["rows"]
    tw, th = (v / P.MM_PER_CELL for v in P.TRIM)
    s = min((tw - 2 * MARGIN_MM / P.MM_PER_CELL) / (2 * d["half"]),
            (th - 2 * MARGIN_MM / P.MM_PER_CELL) / (r1 - r0))
    Xc = W // 2
    top = (H - th) / 2 + TOP_SHARE * (th - (r1 - r0) * s)
    Yc = int(round((top + (cy - r0) * s) / 3)) * 3
    up, down, hx = int((cy - r0) * s), int((r1 - cy) * s), int(d["half"] * s)
    if d["sym"] == "D2":
        up = down = min(up, down)
    R = min(int(d["square"] * s), up, down, hx)
    region = np.zeros((H, W), bool)
    region[Yc - up:Yc + down, Xc - hx:Xc + hx] = True
    inner = np.zeros((H, W), bool)
    inner[Yc - up + FRAME:Yc + down - FRAME, Xc - hx + FRAME:Xc + hx - FRAME] = True
    return SimpleNamespace(Yc=Yc, Xc=Xc, box=(Yc - up, Yc + down, Xc - hx, Xc + hx), s=s, R=R,
                           region=region, inner=inner, sym=d["sym"])


# --- Tracing ------------------------------------------------------------------------

def load_source(name, pre=0.6):
    rgb = np.asarray(Image.open(SOURCES / f"{name}.png").convert("RGB"), float) / 255
    return oklab(ndi.gaussian_filter(rgb, (pre, pre, 0)))


def inks(lab, region, k, seed=1):
    """k inks of the source (OKLab), from k-means on its painted area."""
    from sklearn.cluster import KMeans
    px = lab[region]
    rng = np.random.default_rng(seed)
    px = px[rng.choice(len(px), min(len(px), 150000), replace=False)]
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(px * np.array([1.0, 1.4, 1.4]))
    cen = km.cluster_centers_ / np.array([1.0, 1.4, 1.4])
    return cen[np.argsort(cen[:, 0])]                # darkest first


def one_hot(lab, cen, blur=0.8):
    w = np.array([1.0, 1.4, 1.4], np.float32)
    lab, cen = lab.astype(np.float32) * w, cen.astype(np.float32) * w
    idx = np.argmin([((lab - c) ** 2).sum(-1) for c in cen], axis=0)
    return np.stack([ndi.gaussian_filter((idx == j).astype(np.float32), blur) for j in range(len(cen))], -1)


def sample(oh, ys, xs):
    """Bilinear samples of every channel of `oh` at (ys, xs)."""
    return np.stack([ndi.map_coordinates(oh[..., j], [ys, xs], order=1, mode="nearest")
                     for j in range(oh.shape[-1])], -1)


def trace(d, lay):
    """The poster's soft ink map (H, W, k) before symmetrising, and the inks."""
    lab = load_source(d["src"])
    h, w = lab.shape[:2]
    yy, xx = np.mgrid[0:P.H, 0:P.W].astype(float) + 0.5
    dy, dx = (yy - lay.Yc) / lay.s, (xx - lay.Xc) / lay.s
    cy0, cx0 = d["centre"]
    r0, r1 = d["rows"]
    region = np.zeros((h, w), bool)
    region[r0:r1, max(0, int(cx0 - d["half"])):int(np.ceil(cx0 + d["half"]))] = True
    cen = inks(lab, region, d["k"])
    oh = one_hot(lab, cen)
    if d.get("kind") != "dome":
        return sample(oh, np.clip(cy0 + dy, 0, h - 1), np.clip(cx0 + dx, 0, w - 1)), cen
    r, th = np.hypot(dy, dx), np.arctan2(dy, dx)
    step = np.pi / 16                             # between neighbouring mirror axes
    rel = np.mod(th + np.pi / 2, 2 * step)        # from the output's vertical axis
    rel = np.where(rel > step, 2 * step - rel, rel)
    t = np.radians(d["wedge"]) + (step - rel if d.get("flip") else rel)
    ys, xs = cy0 + d["drift"] * r + r * np.sin(t), cx0 + r * np.cos(t)
    seen = (ys >= 0) & (ys <= h - 1) & (xs >= 0) & (xs <= w - 1)
    ys = np.where(seen, ys, cy0 - np.abs(dy))
    xs = np.where(seen, xs, cx0 - np.abs(dx))
    log(f"dome: cells of the mosaic outside the wedge's reach {1 - seen[lay.region].mean():.1%}")
    return sample(oh, np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)), cen


def symmetrise(Pk, lay, smooth=SMOOTH):
    """Exact symmetry about (Yc, Xc): left-right mirrors everywhere, up-down
    too for D2, D4 in the square. Float addition is commutative, so the
    mirrored cells get identical sums."""
    H, W = Pk.shape[:2]
    Yc, Xc, R = lay.Yc, lay.Xc, lay.R
    if smooth:
        Pk = ndi.gaussian_filter(Pk, (smooth, smooth, 0))
    Pk = Pk.copy()
    hx = min(Xc, W - Xc)
    if lay.sym == "D2":
        hy = min(Yc, H - Yc)
        win = (slice(Yc - hy, Yc + hy), slice(Xc - hx, Xc + hx))
    else:
        win = (slice(None), slice(Xc - hx, Xc + hx))
    S = Pk[win]
    S = S + S[:, ::-1]
    if lay.sym == "D2":
        S = S + S[::-1]
    Pk[win] = S
    if R:
        sq = (slice(Yc - R, Yc + R), slice(Xc - R, Xc + R))
        S = Pk[sq]
        if lay.sym != "D2":
            S = S + S[::-1]
        S = S + S.transpose(1, 0, 2)
        Pk[sq] = S                                # the scale does not matter to argmax
    return Pk


def despeckle(lab, lay, k, min_area=MIN_AREA, rounds=3):
    """Patches of one ink smaller than `min_area` cells take the ink around
    them (re-symmetrised exactly, so the map stays symmetric)."""
    for _ in range(rounds):
        small = np.zeros(lab.shape, bool)
        for j in range(k):
            cc, n = ndi.label(lab == j)
            if n:
                small |= (np.bincount(cc.ravel())[cc] < min_area) & (cc > 0)
        if not small.any():
            break
        oh = np.stack([(lab == j) & ~small for j in range(k)], -1).astype(np.float64)
        lab = np.where(small, symmetrise(oh, lay, smooth=3.0).argmax(-1), lab).astype(np.uint8)
    return lab


def plan(d, lay):
    """The traced, symmetric ink map (H, W) and the inks (OKLab)."""
    Pk, cen = trace(d, lay)
    lab = symmetrise(Pk.astype(np.float64), lay, smooth=d.get("smooth", SMOOTH)).argmax(-1).astype(np.uint8)
    return despeckle(lab, lay, len(cen), d.get("min_area", MIN_AREA)), cen


# --- Symmetric packing ----------------------------------------------------------------

class SymPack:
    """The cover's gapless packing inside the frame, a whole orbit of tiles
    at a time, on the canvas padded by M dead cells. `labels` is the
    (symmetric) ink map; a level-L tile may sit where it lies wholly inside
    the frame (`lay.inner`) and one ink covers THETA[L] of its cells."""

    def __init__(self, labels, k, lay, seed, theta=THETA, levels=(7, 6, 5, 4, 3, 2)):
        H, W = labels.shape
        pad = np.pad(labels, M, mode="edge")
        Hp, Wp = H + 2 * M, W + 2 * M
        self.Yc, self.Xc, self.R, self.sym = lay.Yc + M, lay.Xc + M, lay.R, lay.sym
        assert self.Yc % 3 == 0 and self.Xc % 3 == 0
        C.set_page(Wp, Hp)
        try:
            self.cov = C.Cover(seed)
            self.rng = self.cov.rng
            spec = [np.fft.rfft2((pad == j).astype(float)) for j in range(k)]
            inside = np.fft.rfft2(np.pad(lay.inner, M).astype(float))
            self.tile_ink = []
            for L in levels:
                cnt, fits = self.counts(spec, L), self.counts([inside], L)[0] == len(C.shape(L)[0])
                self.pack(L, cnt, theta[L], fits)
            best = self.counts(spec, 1).argmax(0)
            fits = self.counts([inside], 1)[0] == len(C.shape(1)[0])
            for a, b in np.argwhere(C.PONDS & (self.cov.vlevel == 0) & fits):
                self.cov.put(1, a, b)
                self.tile_ink.append(int(best[a, b]))
            c = self.cov
            self.padded_ok = c.verify()
            tile = np.full((C.H, C.W), -1, np.int32)
            for t, (L, a, b) in enumerate(c.tiles):
                sy, sx = C.shape(L)
                tile[(3 * a + sy) % C.H, (3 * b - 3 * L + 3 + sx) % C.W] = t
            self.key = np.array([(L,) + self.orbit_key(L, a, b) for L, a, b in c.tiles], np.int32)
            self.symmetric = self.closed(c.tiles)
        finally:
            C.set_page(*C.PAGE)
        cut = (slice(M, M + H), slice(M, M + W))
        self.live, self.level, self.tile = c.live[cut], c.ground[cut], tile[cut]
        self.tiles = c.tiles
        self.placed = dict(sorted(c.placed.items()))
        self.still = P.still_life(self.live)
        self.outside = int(c.live.sum() - self.live.sum() + (self.live & ~lay.region).sum())
        self.symmetric &= mirrored(self.level, lay)
        log(f"packed: {self.placed}, padded still life {self.padded_ok}, sheet still life {self.still}, "
            f"live cells outside the frame {self.outside}, layout symmetric {self.symmetric}")

    def counts(self, spec, L):
        """(k, A, B): how many of a level-L tile's cells each map covers, at
        every top vertex."""
        sy, sx = C.shape(L)
        ker = np.zeros((C.H, C.W))
        ker[sy % C.H, sx % C.W] = 1
        kf = np.conj(np.fft.rfft2(ker))
        (ky, kx), _ = C.corner(L)
        return np.stack([np.rint(np.fft.irfft2(s * kf, s=(C.H, C.W)))[ky, kx] for s in spec])

    def offsets(self, L, a, b):
        """The orbit of the level-L tile at top vertex (a, b): its centre's
        offsets from the symmetry centre."""
        cy, cx = 3 * a + 3 * L, 3 * b + 3
        dy = (cy - self.Yc + C.H // 2) % C.H - C.H // 2
        dx = (cx - self.Xc + C.W // 2) % C.W - C.W // 2
        pts = {(dy, dx), (dy, -dx)}
        if self.sym == "D2":
            pts |= {(-dy, dx), (-dy, -dx)}
        ext = 3 * L - 1
        if self.R and abs(dy) + ext <= self.R and abs(dx) + ext <= self.R:
            pts |= {(-dy, dx), (-dy, -dx)}
            pts |= {(x, y) for y, x in pts}
        return pts

    def orbit_key(self, L, a, b):
        return min(self.offsets(L, a, b))

    def closed(self, tiles):
        """True if the orbit of every tile of level 2 and up is among the
        tiles, and every pond's mirror images (the ponds fill whatever is
        left, so along the rim of the D4 square, where bigger tiles from
        outside reach in, a pond's transpose may be taken)."""
        have = set()
        for L, a, b in tiles:
            cy, cx = 3 * a + 3 * L, 3 * b + 3
            have.add((L, cy - self.Yc, cx - self.Xc))
        for L, dy, dx in have:
            pts = {(dy, -dx)} | ({(-dy, dx), (-dy, -dx)} if self.sym == "D2" else set())
            if L > 1:
                pts = self.offsets(L, (dy + self.Yc - 3 * L) // 3, (dx + self.Xc - 3) // 3)
            if any((L, *p) not in have for p in pts):
                return False
        return True

    def images(self, L, a, b):
        """The orbit as top vertices (a, b)."""
        out = []
        for ey, ex in sorted(self.offsets(L, a, b)):
            a2_, b2_ = ((self.Yc + ey - 3 * L) // 3) % C.A, ((self.Xc + ex - 3) // 3) % C.B
            assert (a2_ + b2_) % 2 == 0
            out.append((a2_, b2_))
        return out

    def pack(self, L, cnt, theta, fits):
        n = len(C.shape(L)[0])
        best, ink = cnt.max(0), cnt.argmax(0)
        sites = C.PONDS & fits & (best >= theta * n)
        nr = len(C.rim(L)[0])
        da, db = C.block(L)
        rejected = np.zeros_like(sites)
        placed = 0
        while True:
            occ = self.cov.vlevel > 0
            cand = sites & ~rejected & (C.vcorr(occ, L, "block") < 0.5)
            if not cand.any():
                break
            score = np.where(cand, np.round(C.vcorr(occ, L, "rim") / nr, 6), -np.inf)
            ties = np.argwhere(score >= score.max() - 1e-9)
            a, b = ties[self.rng.integers(len(ties))]
            orbit = self.images(L, a, b)
            verts = [set(zip(((a_ + da) % C.A).tolist(), ((b_ + db) % C.B).tolist())) for a_, b_ in orbit]
            ok = all(cand[a_, b_] for a_, b_ in orbit) and len(set().union(*verts)) == sum(map(len, verts))
            if not ok:
                for a_, b_ in orbit:
                    rejected[a_, b_] = True
                continue
            for a_, b_ in orbit:
                self.cov.put(L, a_, b_)
                self.tile_ink.append(int(ink[a_, b_]))
            placed += len(orbit)
        log(f"  level {L}: {placed} tiles, {int(sites.sum())} sites")


def mirrored(level, lay):
    """True if the tile levels, cell by cell, mirror about the centre."""
    y0, y1, x0, x1 = lay.box
    g = level[y0:y1, x0:x1]
    return bool(np.array_equal(g, g[:, ::-1]) and (lay.sym != "D2" or np.array_equal(g, g[::-1])))


# --- Colour ---------------------------------------------------------------------------
# A palette gives every pane a colour from its ink j, its distance r from
# the centre (1 at the mosaic's half-width) and its orbit's key.

LEAD = P.LEAD
PALETTES = ["source", "fresh", "liquid", "cathedral", "seventies", "brasil"]
LIGHT = 0.35                    # how far the dyes darken from the centre out (ramp units)
JITTER = (0.03, 7.0)            # pane to pane: lightness, hue (degrees)


def seam_colour(spec, hexes, cen):
    """("ink", hex): the ink nearest that colour; ("dark",): a lead deeper
    than the darkest ink."""
    if spec[0] == "ink":
        target = oklab(np.array(C._hex(spec[1])) / 255)
        j = int(np.argmin(((cen - target) ** 2).sum(-1)))
        return hexes[j]
    return P.safe([lab_hex(cen[0], dL=-0.06, gain=0.8)])[0]          # dark lead: the darkest ink, deeper


def ink_hexes(cen):
    return P.safe([lab_hex(c) for c in cen])


def dye_map(cen, dyes, area, least=0.6, weight=10.0):
    """Each ink's dye and place on its ramp. The ink's lightness is spread
    over the dyes' range; on each dye the ink takes the place that best
    matches that lightness and its hue (neutral inks: the lightness only);
    then inks move between dyes, cheapest first, until every dye covers at
    least `least` / (number of dyes) of the mosaic (as the dye pools of
    the postcards share the card)."""
    ts = np.linspace(0, 1.1, 111)
    curves = [np.array([P.ramp(dy, t) for t in ts]) for dy in dyes]
    lo, hi = min(c[:, 0].min() for c in curves), max(c[:, 0].max() for c in curves)
    Ls = cen[:, 0]
    k, D = len(cen), len(dyes)
    cost, place = np.zeros((k, D)), np.zeros((k, D))
    for i, lab in enumerate(cen):
        L, Cc, h = lch(lab)
        target = lo + (L - Ls.min()) / (Ls.max() - Ls.min()) * (hi - lo)
        for j, c in enumerate(curves):
            err = 4 * np.abs(c[:, 0] - target)
            if Cc > 0.035:
                err = err + np.abs((c[:, 2] - h + 180) % 360 - 180) / 90
            cost[i, j], place[i, j] = err.min(), ts[err.argmin()]
    share = np.asarray(area, float) / np.sum(area)

    def total(a):
        cover = np.bincount(a, share, D)
        return (share * cost[np.arange(k), a]).sum() + weight * np.clip(least / D - cover, 0, None).sum()

    a = cost.argmin(1)
    while True:
        moves = [(total(np.where(np.arange(k) == i, j, a)), i, j) for i in range(k) for j in range(D) if j != a[i]]
        best = min(moves)
        if best[0] >= total(a) - 1e-12:
            break
        a[best[1]] = best[2]
    return [(int(a[i]), float(place[i, a[i]])) for i in range(k)]


class Palette:
    def __init__(self, name, cen, seam_spec, area):
        self.name = name
        if name in ("source", "fresh"):
            lab = cen if name == "source" else np.array([fresh(c) for c in cen])
            self.hexes = ink_hexes(lab)
            self.seam = seam_colour(seam_spec, self.hexes, lab)
            return
        self.dyes = P.PALETTES[name]
        self.map = dye_map(cen, self.dyes, area)
        self.seam = P.safe([LEAD])[0]

    def lit(self, j, r):
        dye, t0 = self.map[j]
        return self.dyes[dye], float(np.clip(t0 + LIGHT * (r - 0.5), 0, 1.1))

    def panes(self, inks, rs, keys, seed=5):
        """A colour per pane (same ink, distance and key: same colour)."""
        if hasattr(self, "hexes"):
            return [self.hexes[j] for j in inks]
        cols = []
        for j, r, key in zip(inks, rs, keys):
            ramp, t = self.lit(j, r)
            L, Cc, h = P.ramp(ramp, t)
            rng = np.random.default_rng([seed, *(int(v) + 4096 for v in key)])
            L += rng.normal(0, JITTER[0])
            h += rng.normal(0, JITTER[1])
            cols.append((round(float(np.clip(L, .2, .98)) * 50) / 50, round(Cc * 50) / 50, 3 * round(h / 3) % 360))
        uniq = sorted(set(cols))
        lut = dict(zip(uniq, P.safe([P.oklch_hex(*c) for c in uniq])))
        return [lut[c] for c in cols]

    def field(self, j, r):
        """The pane colour without the jitter (for closed seams)."""
        if hasattr(self, "hexes"):
            return self.hexes[j]
        ramp, t = self.lit(j, round(r * 20) / 20)
        return P.safe([P.oklch_hex(*P.ramp(ramp, t))])[0]


def fresh(lab, gain=1.45, lift=0.03):
    """An ink made fresh: brighter and more saturated (pigment, not age)."""
    L, a, b = lab
    return np.array([L + lift * (1 - L), a * gain, b * gain])


def poster_layers(pk, pal, rule="same", ink_step=0.10, close_max=2):
    """Paper, the seam colour inside the frame, but seams among small tiles
    (levels up to `close_max`) closed, so a clump of ponds reads as one
    field (the cover's hole grout): with rule "same" only where the tiles
    around share one ink, with "majority" always, in the ink most of the
    tiles around have. Then the panes and their live cells."""
    has = pk.tile >= 0
    ink = np.where(has, np.asarray(pk.tile_ink)[np.maximum(pk.tile, 0)], -1)
    big = ndi.maximum_filter(np.where(has, pk.level, 0), size=5)
    small = pk.inner & ~has & (big <= close_max)
    k = int(max(pk.tile_ink)) + 1
    if rule == "same":
        lo = ndi.minimum_filter(np.where(has, ink, 255), size=5)
        close = small & (lo == ndi.maximum_filter(ink, size=5))
    else:
        counts = np.stack([ndi.correlate((ink == j).astype(np.int32), np.ones((5, 5), np.int32), mode="constant")
                           for j in range(k)])
        lo = counts.argmax(0)
        close = small & (counts.max(0) > 0)
    yy, xx = np.mgrid[0:pk.live.shape[0], 0:pk.live.shape[1]]
    rq = np.round(np.hypot(yy + 0.5 - pk.Yc, xx + 0.5 - pk.Xc) / pk.rscale * 20).astype(int)
    code = np.where(close, lo * 1000 + rq, -1)
    layers = [(PAPER, np.ones(pk.live.shape, bool)), (pal.seam, pk.region)]
    for c in np.unique(code[close]):
        layers.append((pal.field(int(c // 1000), (c % 1000) / 20), code == c))
    rs = np.hypot(pk.key[:, 1], pk.key[:, 2]) / pk.rscale
    colours = pal.panes(pk.tile_ink, rs, pk.key)
    return layers + P.tile_layers(pk.tile, pk.live, colours, ink_step)


class Saved:
    """A packed poster read back from its npz."""

    def __init__(self, path):
        z = np.load(path)
        for f in z.files:
            setattr(self, f, z[f])
        self.tile_ink = list(self.tile_ink)
        self.Yc, self.Xc, self.rscale = int(self.Yc), int(self.Xc), float(self.rscale)


def plan_image(lab, hexes, region):
    img = np.array([C._hex(h) for h in hexes], np.uint8)[lab]
    img[~region] = 255
    return img


def build(names, out: Path, seed=301):
    P.C.PROFILE = P.PROFILE
    out.mkdir(parents=True, exist_ok=True)
    C._POOLS.clear()
    C.tile_pool.__defaults__ = (12, (0.35, 0.75))       # the jury print's calm tiling
    for name in names:
        d = DESIGNS[name]
        lay = layout(d)
        lab, cen = plan(d, lay)
        hexes = ink_hexes(cen)
        log(f"{name}: {P.W} x {P.H} cells, mosaic {lay.box[3] - lay.box[2]} x {lay.box[1] - lay.box[0]} "
            f"at {lay.s:.3f} cells per px, inks {' '.join(hexes)}")
        Image.fromarray(plan_image(lab, hexes, lay.region)).save(out / f"{name}_plan.png")
        pk = SymPack(lab, len(cen), lay, seed)
        np.savez_compressed(out / f"{name}.npz", labels=lab, live=pk.live, tile=pk.tile, level=pk.level,
                            tile_ink=np.array(pk.tile_ink), key=pk.key, inks=cen, region=lay.region,
                            inner=lay.inner, Yc=lay.Yc, Xc=lay.Xc, rscale=(lay.box[3] - lay.box[2]) / 2)
        (out / f"{name}.json").write_text(json.dumps(dict(
            placed=pk.placed, padded_ok=pk.padded_ok, still_life=pk.still, outside=pk.outside,
            symmetric=pk.symmetric, box=lay.box, scale=lay.s, canvas=(P.W, P.H)), indent=1))
        render([name], out)


def render(names, out: Path, palettes=PALETTES):
    """Every palette of saved packs: the sheet at one pixel per cell, a
    half-size preview and a close-up of the centre."""
    P.C.PROFILE = P.PROFILE
    for name in names:
        d = DESIGNS[name]
        layout(d)
        pk = Saved(out / f"{name}.npz")
        meta = json.loads((out / f"{name}.json").read_text())
        meta["palettes"] = {}
        for pn in palettes:
            pal = Palette(pn, pk.inks, d["seam"], np.bincount(pk.labels[pk.region], minlength=len(pk.inks)))
            ls = P.compress(poster_layers(pk, pal, rule=d.get("close", "same")))
            rgb = P.flatten(ls)
            img = Image.fromarray(rgb)
            img.save(out / f"{name}_{pn}.png")
            img.resize((P.W // 2, P.H // 2), Image.LANCZOS).save(out / f"{name}_{pn}_preview.png")
            crop = rgb[pk.Yc - 60:pk.Yc + 60, pk.Xc - 90:pk.Xc + 90]
            Image.fromarray(crop).resize((1080, 720), Image.NEAREST).save(out / f"{name}_{pn}_detail.png")
            meta["palettes"][pn] = dict(colours=len(ls), seam=pal.seam)
            log(f"{name} {pn}: {len(ls)} colours")
        (out / f"{name}.json").write_text(json.dumps(meta, indent=1))


# The sheets chosen for print, each in its palette.
PRINTS = [("mandala-gold", "fresh"), ("mandala-blue", "fresh"), ("mandala-dark", "fresh"),
          ("dome-calm", "fresh"), ("tilework", "fresh")]


def write_print(name, pal_name, src: Path, out: Path):
    """One sheet for print: a vector PDF of the A2 page with 3 mm bleed, a
    600 dpi PNG of it and a small preview; the PDF is rasterised again and
    checked cell by cell against the cells."""
    import matplotlib.pyplot as plt
    d = DESIGNS[name]
    layout(d)
    pk = Saved(src / f"{name}.npz")
    assert P.still_life(pk.live) and not (pk.live & ~pk.inner).any()
    area = np.bincount(pk.labels[pk.region], minlength=len(pk.inks))
    ls = P.compress(poster_layers(pk, Palette(pal_name, pk.inks, d["seam"], area), rule=d.get("close", "same")))
    stem = out / f"sacred-{name}_A2"
    img = P.save_png(ls, stem.with_suffix(".png"))
    P.preview(img, out / f"{stem.name}_preview.png", scale=0.08)
    fig, ax = P.new_page()
    P.draw_layers(ax, ls)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    wrong, cells = check_pdf(stem.with_suffix(".pdf"), P.flatten(ls))
    log(f"{stem.name}: {P.PAGE_MM[0]:.0f} x {P.PAGE_MM[1]:.0f} mm, {len(ls)} colours, "
        f"{int(pk.live.sum())} live cells, PDF {stem.with_suffix('.pdf').stat().st_size / 1e6:.1f} MB, "
        f"cells differing from the render {wrong} of {cells}")
    return dict(page_mm=P.PAGE_MM, colours=len(ls), live=int(pk.live.sum()), wrong=wrong, cells=cells)


def check_pdf(pdf, rgb, dpi=300):
    """Rasterise the PDF with pdftoppm and compare the pixel at every cell
    centre on the page with the cell's colour: (cells that differ, cells
    compared). Without anti-aliasing, which only touches the cells' edges
    (and which poppler 20.10 gets wrong on these pages: white slivers)."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-aaVector", "no", "-png", "-singlefile", str(pdf), f"{tmp}/page"],
                       check=True)
        page = np.asarray(Image.open(f"{tmp}/page.png").convert("RGB"))
    c, ox, oy = P.canvas_to_mm()
    px = dpi / 25.4
    assert all(abs(n - v * px) <= 1 for n, v in zip(page.shape, P.PAGE_MM[::-1])), page.shape
    ys, xs = ((np.arange(P.H) + 0.5) * c - oy) * px, ((np.arange(P.W) + 0.5) * c - ox) * px
    iy = np.flatnonzero((ys >= 1) & (ys < page.shape[0] - 1))
    ix = np.flatnonzero((xs >= 1) & (xs < page.shape[1] - 1))
    got = page[np.floor(ys[iy]).astype(int)][:, np.floor(xs[ix]).astype(int)]
    return int((got != rgb[iy][:, ix]).any(-1).sum()), len(iy) * len(ix)


def prints(src: Path, out: Path):
    P.C.PROFILE = P.PROFILE
    Image.MAX_IMAGE_PIXELS = None                   # the 600 dpi pages are ours
    out.mkdir(parents=True, exist_ok=True)
    meta = {f"{name}_{pn}": write_print(name, pn, src, out) for name, pn in PRINTS}
    (out / "prints.json").write_text(json.dumps(meta, indent=1))


def plans(names, out: Path):
    """Only the traced maps, in the source's inks."""
    P.C.PROFILE = P.PROFILE
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        d = DESIGNS[name]
        lay = layout(d)
        lab, cen = plan(d, lay)
        Image.fromarray(plan_image(lab, ink_hexes(cen), lay.region)).save(out / f"{name}_plan.png")
        log(f"{name}: plan written")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build", "plans", "render", "print"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", type=Path, default=OUT / "round2")
    ap.add_argument("--profile", type=Path, help="Coated FOGRA39 ICC profile")
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--palettes", nargs="*", default=PALETTES)
    a = ap.parse_args()
    P.PROFILE = a.profile
    names = a.names or list(DESIGNS)
    if a.cmd == "build":
        build(names, a.out, a.seed)
    elif a.cmd == "plans":
        plans(names, a.out)
    elif a.cmd == "print":
        prints(a.out, OUT / "print")
    else:
        render(names, a.out, a.palettes)


if __name__ == "__main__":
    main()
