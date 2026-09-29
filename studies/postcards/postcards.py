"""Defence postcards (A6): five fronts and their backs.

Handed out at the PhD defence (joint UGent / USP, 2 October 2026), one card
per guest. Every card is A6, 105 x 148 mm, delivered with 3 mm bleed
(111 x 154 mm) as a two-page vector PDF (front, back). The design history
is in README.md.

**Cells.** One cell is 1/60 inch (0.423 mm; the Conway card 1/120 inch): a
whole number of device pixels at 300, 600, 1200 and 2400 dpi, so a printer
that rasterises the page never stretches one cell more than its
neighbour. The canvas (multiples of 6 cells, for the pond lattice) is
centred on the bleed page and clipped by it. The PDFs are vector, one
path per colour with the cells traced into polygons; the PNGs are 600 dpi.

**Fronts** (FINAL, `--round 3`):

- conway: the README mosaic's recipe (level-3 tiles, black on white, an
  elementary cellular automaton in UGent yellow and blue), rebuilt for A6.
- dragon: the gilded dragon of the Ghent Belfry, drawn for the card held
  sideways: whole tiles of levels 1-4 each in one gold ("gold panes") from
  a photo (assets/dragon.png, kept out of git), over a night-blue automaton.
- paraty: the thesis cover's Paraty Mirim (design c, tiles in the sea).
- liquid-vortex, liquid-cathedral: stained glass. The whole card is packed
  with tiles of levels 1-7 set in dark lead, each tile one pane coloured
  from a liquid-light dye field (a swirled pool of two dyes; bubbles of
  cobalt and ruby on amber).

**Backs.** The card's field with a band of its front along the bottom
(mirrored, so it lines up with the front through the paper), the thesis
title and a thank-you in English, Dutch and Portuguese in the thesis
cover's inline-code boxes (Courier Prime), a dateline, the website and a
QR code for it.

Colours are kept inside the Coated FOGRA39 gamut; pass the profile
(color.org/registry, Coated_Fogra39L_VIGC_300.icc, not in git) with
--profile:

    python studies/postcards/postcards.py fronts --round 3 --out studies/postcards/output/final --profile ICC
    python studies/postcards/postcards.py cards conway dragon paraty liquid-vortex liquid-cathedral \
        --src studies/postcards/output/final --out studies/postcards/output/final --profile ICC
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ASSETS = HERE / "assets"
for p in (REPO / "src", HERE.parent, HERE.parent / "cover"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from common import log

TRIM = (105.0, 148.0)           # mm
BLEED = 3.0                     # mm, each side
PAGE_MM = (TRIM[0] + 2 * BLEED, TRIM[1] + 2 * BLEED)


def set_canvas(cpi, trim=(105.0, 148.0), mult=6):
    """Cells of 1/cpi inch on a `trim` page (mm, plus the bleed): the canvas
    (multiples of `mult` cells, at least the bleed page) and the PNG's
    pixels per cell (600 dpi)."""
    global CPI, MM_PER_CELL, W, H, PX, TRIM, PAGE_MM
    TRIM = tuple(trim)
    PAGE_MM = (TRIM[0] + 2 * BLEED, TRIM[1] + 2 * BLEED)
    CPI, MM_PER_CELL = cpi, 25.4 / cpi
    W, H = (mult * int(np.ceil(v / MM_PER_CELL / mult)) for v in PAGE_MM)
    PX = max(1, 600 // cpi)


set_canvas(60)                  # 264 x 366 cells of 0.423 mm, 10 px per cell


class turned:
    """Inside the block the canvas and the page are landscape (a design
    made for the card held sideways); `upright` turns its arrays back onto
    the portrait card, the landscape top along the card's left edge."""

    def __enter__(self):
        global W, H, PAGE_MM, TRIM
        self.saved = (W, H, PAGE_MM, TRIM)
        W, H, PAGE_MM, TRIM = H, W, PAGE_MM[::-1], TRIM[::-1]

    def __exit__(self, *exc):
        global W, H, PAGE_MM, TRIM
        W, H, PAGE_MM, TRIM = self.saved

    @staticmethod
    def upright(layers, info):
        return ([(c, np.rot90(m)) for c, m in layers],
                {k: np.rot90(v) if isinstance(v, np.ndarray) else v for k, v in info.items()})


def page_px():
    """The bleed page in PNG pixels, and the canvas offset (it is centred)."""
    pw, ph = (round(v / MM_PER_CELL * PX) for v in PAGE_MM)
    return (pw, ph), ((W * PX - pw) // 2, (H * PX - ph) // 2)


def trim_box_cells():
    """The trim (x0, y0, x1, y1) in canvas cells (floats)."""
    ox, oy = (W - PAGE_MM[0] / MM_PER_CELL) / 2, (H - PAGE_MM[1] / MM_PER_CELL) / 2
    b = BLEED / MM_PER_CELL
    return ox + b, oy + b, W - ox - b, H - oy - b


# --- The dragon ------------------------------------------------------------------

def dragon_cutout():
    """Mask and luminance (0 black .. 1 white) of the dragon photo, which
    is on a white ground."""
    rgb = np.asarray(Image.open(ASSETS / "dragon.png").convert("RGB"), float)
    dist = np.sqrt(((255 - rgb) ** 2).sum(-1))
    mask = dist > 38
    mask = ndi.binary_closing(mask, np.ones((5, 5), bool))
    holes, n = ndi.label(ndi.binary_fill_holes(mask) & ~mask)
    small = 1 + np.flatnonzero(ndi.sum(np.ones_like(mask), holes, range(1, n + 1)) < 3000)
    mask |= np.isin(holes, small)               # rivet gaps closed, the loop by the neck kept
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    mask = np.isin(lab, 1 + np.flatnonzero(sizes > 400))
    lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255
    return mask, lum


def dragon_layout(angle=-40, width=1.45, cx=0.52, cy=0.50):
    """The dragon on the canvas: turned `angle` degrees (negative is
    clockwise, the head rising to the upper left), `width` its span in
    canvas widths, its centre at (cx, cy) of the canvas.

    Returns (mask, lum) at cell resolution: mask True on the dragon, lum
    its luminance there (1 elsewhere)."""
    mask, lum = dragon_cutout()
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8))
    l = Image.fromarray((np.where(mask, lum, 1)[y0:y1, x0:x1] * 255).astype(np.uint8))
    m = m.rotate(angle, Image.BICUBIC, expand=True, fillcolor=0)
    l = l.rotate(angle, Image.BICUBIC, expand=True, fillcolor=255)
    s = width * W * 8 / m.width                     # 8x supersampled, then averaged per cell
    size = (max(1, round(m.width * s)), max(1, round(m.height * s)))
    m, l = m.resize(size, Image.LANCZOS), l.resize(size, Image.LANCZOS)
    big_m = np.zeros((H * 8, W * 8), np.float32)
    big_l = np.ones((H * 8, W * 8), np.float32)
    top, left = round(cy * H * 8 - size[1] / 2), round(cx * W * 8 - size[0] / 2)
    ys = slice(max(0, top), min(H * 8, top + size[1]))
    xs = slice(max(0, left), min(W * 8, left + size[0]))
    src = (slice(ys.start - top, ys.stop - top), slice(xs.start - left, xs.stop - left))
    big_m[ys, xs] = np.asarray(m, np.float32)[src] / 255
    big_l[ys, xs] = np.asarray(l, np.float32)[src] / 255
    cell_m = big_m.reshape(H, 8, W, 8).mean((1, 3))
    cell_l = big_l.reshape(H, 8, W, 8).mean((1, 3))
    return cell_m > 0.5, np.where(cell_m > 0.5, cell_l, 1.0)


def dragon_fit(angle=0.0, margin=6.0, mass=0.5):
    """The dragon turned `angle` degrees (negative is clockwise), as large as
    fits with its centre of mass on the middle of the trim and every part
    at least `margin` mm inside it. `mass` weighs the centre of mass
    against the middle of the bounding box for the point put on the middle
    of the trim (1: the centre of mass exactly). Returns (mask, lum) like
    dragon_layout."""
    mask, lum = dragon_cutout()
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8)).rotate(
        angle, Image.BICUBIC, expand=True, fillcolor=0)
    l = Image.fromarray((np.where(mask, lum, 1)[y0:y1, x0:x1] * 255).astype(np.uint8)).rotate(
        angle, Image.BICUBIC, expand=True, fillcolor=255)
    a = np.asarray(m) > 127
    ys, xs = np.nonzero(a)
    by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy = mass * ys.mean() + (1 - mass) * (by0 + by1) / 2
    cx = mass * xs.mean() + (1 - mass) * (bx0 + bx1) / 2
    tx0, ty0, tx1, ty1 = trim_box_cells()
    g = margin / MM_PER_CELL
    mx, my = (tx0 + tx1) / 2, (ty0 + ty1) / 2
    s = min((mx - tx0 - g) / (cx - bx0), (tx1 - g - mx) / (bx1 - cx),
            (my - ty0 - g) / (cy - by0), (ty1 - g - my) / (by1 - cy))   # cells per photo pixel
    size = (max(1, round(m.width * s * 8)), max(1, round(m.height * s * 8)))
    m, l = m.resize(size, Image.LANCZOS), l.resize(size, Image.LANCZOS)
    left, top = round((mx - cx * s) * 8), round((my - cy * s) * 8)
    big_m = np.zeros((H * 8, W * 8), np.float32)
    big_l = np.ones((H * 8, W * 8), np.float32)
    ys = slice(max(0, top), min(H * 8, top + size[1]))      # the photo's empty corners may
    xs = slice(max(0, left), min(W * 8, left + size[0]))    # stick out of the canvas
    src = (slice(ys.start - top, ys.stop - top), slice(xs.start - left, xs.stop - left))
    big_m[ys, xs] = np.asarray(m, np.float32)[src] / 255
    big_l[ys, xs] = np.asarray(l, np.float32)[src] / 255
    cell_m = big_m.reshape(H, 8, W, 8).mean((1, 3))
    cell_l = big_l.reshape(H, 8, W, 8).mean((1, 3))
    return cell_m > 0.5, np.where(cell_m > 0.5, cell_l, 1.0)


def cmd_dragon_layout(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    mask, lum = dragon_layout()
    np.savez_compressed(out / "dragon_layout.npz", mask=mask, lum=lum)
    prev = np.where(mask, (lum * 200).astype(np.uint8), 255).astype(np.uint8)
    Image.fromarray(prev).resize((W * 3, H * 3), Image.NEAREST).save(out / "dragon_layout.png")
    log(f"dragon layout: {mask.mean():.0%} of the canvas, bbox rows "
        f"{np.nonzero(mask.any(1))[0][[0, -1]]}, cols {np.nonzero(mask.any(0))[0][[0, -1]]}")


# --- Packing (the thesis cover's gapless tiles) ----------------------------------

import cover as C                                   # noqa: E402  (studies/cover/cover.py)

M = C.MARGIN                                        # the packing margin, cut off afterwards


def still_life(g):
    """True if `g` (outside it all dead) is a still life."""
    g = np.pad(g.astype(np.int16), 1)
    n = sum(np.roll(np.roll(g, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    nxt = ((g == 1) & ((n == 2) | (n == 3))) | ((g == 0) & (n == 3))
    return bool(np.array_equal(nxt, g == 1))


class Packed:
    """Whole tiles packed into a region of the canvas: the cover's packer
    run on the canvas padded by M cells (the region mirrored into it), cut
    back to the canvas. Keeps per-cell `live`, `level` (tile ground, 0 in
    the seams) and `tile` (index into `tiles`, -1 in the seams), and per
    tile its level and centre cell."""

    def __init__(self, region, seed, Z=None, levels=(7, 6, 5, 4, 3, 2), reach=C.REACH, **shore):
        pad = np.pad(region, M, mode="symmetric")
        C.set_page(W + 2 * M, H + 2 * M)
        try:
            c = C.Cover(seed)
            Zp = C.shore_field(c.rng, pad, **shore) if Z is None else np.pad(Z, M, mode="symmetric")
            c.fill(pad, Zp, levels=levels, reach=reach)
            self.padded_ok = c.verify()
            tile = np.full((C.H, C.W), -1, np.int32)
            centres = []
            for k, (L, a, b) in enumerate(c.tiles):
                sy, sx = C.shape(L)
                tile[(3 * a + sy) % C.H, (3 * b - 3 * L + 3 + sx) % C.W] = k
                centres.append(((3 * a + 3 * L) % C.H - M, (3 * b + 3) % C.W - M))
            lab = c.labels()
        finally:
            C.set_page(*C.PAGE)
        cut = (slice(M, M + H), slice(M, M + W))
        self.live, self.level, self.tile = c.live[cut], c.ground[cut], tile[cut]
        self.labels = lab[cut]
        self.tiles = [(L, y, x) for (L, _, _), (y, x) in zip(c.tiles, centres)]
        self.placed = dict(sorted(c.placed.items()))
        log(f"packed: tiles {self.placed}, padded page still life {self.padded_ok}, "
            f"canvas still life {still_life(self.live)}")


# --- Colour --------------------------------------------------------------------------

def oklch_hex(L, Cc, h):
    return C.from_oklch(float(L), float(Cc), float(h))


_SAFE = {}
_GAMUT = None
_PROOF = None


def gamut_check(rgb):
    """The cover's `gamut_check` with the proof transform built once (it
    loads the 8.6 MB press profile)."""
    global _PROOF
    from PIL import ImageCms
    if PROFILE is None:
        raise SystemExit("pass --profile Coated_Fogra39L_VIGC_300.icc (from color.org)")
    if _PROOF is None:
        F = ImageCms.Flags
        srgb = ImageCms.createProfile("sRGB")
        _PROOF = ImageCms.buildProofTransform(srgb, srgb, ImageCms.getOpenProfile(str(PROFILE)),
                                              "RGB", "RGB", renderingIntent=0, proofRenderingIntent=1,
                                              flags=F.SOFTPROOFING | F.GAMUTCHECK)
    out = np.asarray(ImageCms.applyTransform(
        Image.fromarray(np.asarray(rgb, np.uint8).reshape(1, -1, 3)), _PROOF))[0]
    return ~np.all(out == 0x7F, axis=-1)


def safe(colours):
    """Print-safe (Coated FOGRA39) versions of hex colours, cached: each
    colour, or the nearest printable one (the cover's `print_safe`, with a
    k-d tree over the printable grid so hundreds of pane colours are quick)."""
    global _GAMUT
    todo = sorted({c for c in colours if c not in _SAFE})
    if todo:
        rgb = np.array([C._hex(h) for h in todo], np.uint8)
        inside = gamut_check(rgb)
        near = {}
        if not inside.all():
            if _GAMUT is None:
                from scipy.spatial import cKDTree
                v = np.arange(0, 256, 4)
                grid = np.stack(np.meshgrid(v, v, v, indexing="ij"), -1).reshape(-1, 3)
                g = grid[gamut_check(grid)]
                _GAMUT = (g, cKDTree(C._lab(g) * np.array([1.3, 1.0, 1.0])))
            g, tree = _GAMUT
            _, idx = tree.query(C._lab(rgb[~inside]) * np.array([1.3, 1.0, 1.0]))
            near = dict(zip(np.flatnonzero(~inside), idx))
        for k, h in enumerate(todo):
            _SAFE[h] = h if inside[k] else "#%02X%02X%02X" % tuple(int(v) for v in g[near[k]])
    return [_SAFE[c] for c in colours]


def tone(hexcol, step):
    """The same colour `step` darker in OKLCH lightness (lighter if too dark)."""
    return C._tone(hexcol, step)


def flatten(layers):
    """The RGB image the layers paint, one pixel per cell."""
    rgb = np.zeros((H, W, 3), np.uint8)
    for c, m in layers:
        rgb[m] = C._hex(c)
    return rgb


def compress(layers):
    """One layer per colour, holding exactly the cells that show it, in
    order of first appearance: the same image with the fewest paths, and
    since no two layers overlap, the order no longer matters."""
    rgb = flatten(layers)
    order = []
    for c, _ in layers:
        if c not in order:
            order.append(c)
    out = []
    for c in order:
        m = np.all(rgb == np.array(C._hex(c), np.uint8), -1)
        if m.any():
            out.append((c, m))
    return out


# --- Output: PNG and vector PDF ----------------------------------------------------

def loops(mask):
    """The outline of `mask`'s cells as closed vertex loops (x, y in cells),
    region on the right of each directed edge, collinear corners dropped."""
    m = np.pad(mask, 1)
    edges = {}
    for (dy, dx), (p, q) in {(-1, 0): ((0, 0), (1, 0)), (0, 1): ((1, 0), (1, 1)),
                             (1, 0): ((1, 1), (0, 1)), (0, -1): ((0, 1), (0, 0))}.items():
        ys, xs = np.nonzero(m[1:-1, 1:-1] & ~m[1 + dy:m.shape[0] - 1 + dy, 1 + dx:m.shape[1] - 1 + dx])
        for y, x in zip(ys.tolist(), xs.tolist()):
            edges.setdefault((x + p[0], y + p[1]), []).append((x + q[0], y + q[1]))
    out = []
    while edges:
        start = next(iter(edges))
        pts, cur = [start], start
        while True:
            nxt = edges[cur].pop()
            if not edges[cur]:
                del edges[cur]
            if nxt == start:
                break
            pts.append(nxt)
            cur = nxt
        pts = [pt for k, pt in enumerate(pts)
               if not (pts[k - 1][0] == pt[0] == pts[(k + 1) % len(pts)][0]
                       or pts[k - 1][1] == pt[1] == pts[(k + 1) % len(pts)][1])]
        out.append(pts)
    return out


def canvas_to_mm():
    """(mm per cell, x offset, y offset): page mm = cells * scale - offset (y down)."""
    c = MM_PER_CELL
    return c, (W * c - PAGE_MM[0]) / 2, (H * c - PAGE_MM[1]) / 2


def draw_layers(ax, layers):
    """Paint layers onto a page axes in mm (y up), the canvas centred."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    c, ox, oy = canvas_to_mm()
    for col, m in layers:
        verts, codes = [], []
        for pts in loops(m):
            v = [(x * c - ox, PAGE_MM[1] - (y * c - oy)) for x, y in pts]
            verts += v + [v[0]]
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(v) - 1) + [MPath.CLOSEPOLY]
        if verts:
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor=col, edgecolor="none", lw=0))


def new_page():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(PAGE_MM[0] / 25.4, PAGE_MM[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_MM[0])
    ax.set_ylim(0, PAGE_MM[1])
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save_png(layers, path):
    """The bleed page at PX pixels per cell (600 dpi)."""
    (pw, ph), (ox, oy) = page_px()
    img = Image.fromarray(flatten(layers)).resize((W * PX, H * PX), Image.NEAREST)
    img = img.crop((ox, oy, ox + pw, oy + ph))
    img.save(path, dpi=(PX * CPI, PX * CPI), optimize=True)
    return img


def preview(img, path, scale=0.25, marks=True):
    """A small preview with the trim drawn in."""
    from PIL import ImageDraw
    small = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    if marks:
        d = ImageDraw.Draw(small)
        b = BLEED / MM_PER_CELL * PX * scale
        d.rectangle([b, b, small.width - b, small.height - b], outline=(255, 0, 255))
    small.save(path)
    return small


# --- Fronts --------------------------------------------------------------------------

FRONTS = {}


def front(name):
    def wrap(fn):
        FRONTS[name] = fn
        return fn
    return wrap


def eca_layer(rule, cell, seed, colours):
    """An elementary cellular automaton over the canvas, `cell` cells per
    automaton cell: [(background hex, all), (pixel hex, on)]."""
    from gol_mosaics.eca import ECABackground
    np.random.seed(seed)
    on = ECABackground(rule).generate(W, H, cell).astype(bool)
    return [(colours[0], np.ones((H, W), bool)), (colours[1], on)]


# Conway: the original recipe (MosaicGenerator, black cells, UGent ECA).
UGENT = ("#FFD200", "#1E64C8")


def conway(level=3, grid=30, seed=5, rule=106, cutoff=0.65, contrast=5.0, drop=0):
    from gol_mosaics import MosaicGenerator
    img = Image.open(REPO / "input" / "images" / "john.png")
    gen = MosaicGenerator(level=level, grid_size=grid, eca_rule=rule)
    _, cells, transparent = gen.generate_from_pil(img, return_arrays=True, remove_background=False,
                                                  seed=seed, empty_tiles_cutoff=cutoff,
                                                  contrast=contrast)
    cells, transparent = np.asarray(cells) > 0, np.asarray(transparent) > 0
    h, w = cells.shape
    live, bg = np.zeros((H, W), bool), np.ones((H, W), bool)
    oy, ox = (H - h) // 2 + drop, (W - w) // 2          # the mosaic's corner on the canvas
    ys, xs = slice(max(0, oy), min(H, oy + h)), slice(max(0, ox), min(W, ox + w))
    src = (slice(ys.start - oy, ys.stop - oy), slice(xs.start - ox, xs.stop - ox))
    live[ys, xs] = cells[src]
    bg[ys, xs] = transparent[src]
    # the lattice's top and side edges end in blank half-tiles: let the
    # automaton show there, wherever no tile is near
    edge = np.zeros((h, w), bool)
    edge[:14], edge[:, :14], edge[:, -14:] = True, True, True
    band = np.zeros((H, W), bool)
    band[ys, xs] = edge[src]
    bg |= band & ~ndi.binary_dilation(live, iterations=3)
    log(f"conway L{level} grid {grid}: mosaic {w}x{h}, canvas still life {still_life(live)}")
    return live, bg


@front("conway")
def f_conway(level=3, grid=58, rule=106, eca_cell=12, seed=5, cpi=120, **kw):
    set_canvas(cpi)
    live, bg = conway(level, grid, seed, rule, **kw)
    ls = [("#FFFFFF", np.ones((H, W), bool))]
    ls += [(c, m & bg) for c, m in eca_layer(rule, eca_cell, seed, UGENT)]
    ls += [("#000000", live)]
    return ls, dict(live=live)


# Paraty: the cover's design c with tiles in the sea, denim on apricot.
PAL = C.load_palettes()


def paraty_sea(zoom=False):
    rgb = np.asarray(Image.open(C.SILHOUETTES / "paraty-c.png").convert("RGB"), float)
    g = rgb[..., 1] - rgb[..., 2]
    h0, w0 = g.shape
    if zoom:        # the cover's cell scale (360 across): a window of it
        big = np.asarray(Image.fromarray(g.astype(np.float32), "F").resize((360, 540), Image.BOX))
        top, left = (540 - H) // 2 - 20, (360 - W) // 2 + 10
        land = big[top:top + H, left:left + W] > 0
    else:           # the whole width, cropped top and bottom to the card's shape
        ch = round(w0 * H / W)
        t = (h0 - ch) // 2
        land = np.asarray(Image.fromarray(g[t:t + ch].astype(np.float32), "F").resize((W, H), Image.BOX)) > 0
    return ~land


@front("paraty")
def f_paraty(zoom=False, seed=96):
    p = Packed(paraty_sea(zoom), seed)
    return C.layers(p.labels, PAL, "denim", "apricot", C.COAST), dict(live=p.live, tile=p.tile)


# Stained glass: the whole card packed, each tile one pane, coloured from a
# liquid-light field (a lit pool of two immiscible dyes).
LEAD = "#1D1A2B"


def warp_noise(rng, sigma, swirl=2.2, centre=(0.47, 0.45)):
    """Smooth noise sampled along a vortex: coordinates turned by an angle
    that falls off with the distance from `centre`."""
    n = ndi.gaussian_filter(rng.normal(size=(H + 200, W + 200)), sigma, mode="wrap")
    n /= n.std()
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cy, cx = centre[1] * H, centre[0] * W
    dy, dx = yy - cy, xx - cx
    t = swirl * np.exp(-np.hypot(dx, dy) / H / 0.35)
    ry = cy + dx * np.sin(t) + dy * np.cos(t) + 100
    rx = cx + dx * np.cos(t) - dy * np.sin(t) + 100
    return ndi.map_coordinates(n, [ry, rx], order=1, mode="wrap")


def glass_field(seed, centre=(0.47, 0.45), sigma=26, drops=2.2):
    """The blue dye's pools (True), its droplets and the other dye's (True
    where the colour flips) and the radius from the light (0 at the light,
    ~1 in the corners)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    r = np.hypot((xx - centre[0] * W) / H, (yy - centre[1] * H) / H) / 0.62
    pools = warp_noise(rng, sigma, centre=centre) + 0.5 * warp_noise(rng, sigma / 2, 1.2, centre) > 0.3
    flips = warp_noise(rng, 4.5, 0.5, centre) > drops
    return pools, flips, r


def blob_noise(rng, sigma):
    n = ndi.gaussian_filter(rng.normal(size=(H, W)), sigma, mode="wrap")
    return n / n.std()


def dye_field(kind, k, seed, light=(0.47, 0.45)):
    """Which of `k` dyes fills each cell, and the distance from the light
    (0 at the light, about 1 in the far corners).

    vortex: pools swirled round the light (two dyes: the round-1 field);
    fingers: one dye spreading from the light in fingers (viscous
    fingering, like oil pushed into water), the others around it;
    bubbles: round drops of every size floating on the first dye;
    rays: bands fanning out from a light above the top edge."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    ly, lx = light[1] * H, light[0] * W
    r = np.hypot(xx - lx, yy - ly) / (0.62 * H)
    if kind == "vortex":
        if k == 2:
            pools = warp_noise(rng, 26, centre=light) + 0.5 * warp_noise(rng, 13, 1.2, light) > 0.3
            return pools.astype(int), r
        n = np.stack([warp_noise(rng, 26, centre=light) + 0.4 * warp_noise(rng, 11, 1.2, light)
                      for _ in range(k)])
        return n.argmax(0), r
    if kind == "fingers":
        th = np.arctan2(yy - ly, xx - lx)
        rho = np.hypot(yy - ly, xx - lx)
        big = ndi.gaussian_filter(rng.normal(size=(400, 400)), (2.5, 9), mode="wrap")
        big /= big.std()
        u = (th + np.pi) / (2 * np.pi) * 400                # around
        v = np.log1p(rho / 6) * 70 % 400                     # outwards, stretched
        n = ndi.map_coordinates(big, [u, v], order=1, mode="wrap")
        reach = 1.9 - 2.6 * np.clip(rho / (0.55 * H), 0, 1.4)
        lab = np.where(n + reach > 0.4, 0, 1)
        if k > 2:
            lab = np.where((lab == 1) & (blob_noise(rng, 30) > 0.35), 2, lab)
        return lab, r
    if kind == "bubbles":
        lab = np.zeros((H, W), int)
        n = int(0.0022 * H * W)
        radii = np.clip(34 * rng.random(n) ** 2.4, 2.5, None)
        order = np.argsort(-radii)
        for j in order:
            y, x = rng.random() * H, rng.random() * W
            lab[(yy - y) ** 2 + (xx - x) ** 2 <= radii[j] ** 2] = 1 + (j % (k - 1))
        return lab, r
    if kind == "rays":
        ly, lx = -0.35 * H, 0.5 * W
        th = np.arctan2(xx - lx, yy - ly)
        r = np.hypot(xx - lx, yy - ly) / (1.25 * H)
        t = np.linspace(-1, 1, 2048)
        band = ndi.gaussian_filter1d(rng.normal(size=2048), 22, mode="wrap")
        band = np.interp(th, np.linspace(-0.9, 0.9, 2048), band / band.std())
        band = band + 0.35 * blob_noise(rng, 7)
        qs = np.quantile(band, np.linspace(0, 1, k + 1)[1:-1])
        return np.digitize(band, qs), np.clip(r - 0.25, 0, None) / 0.8
    raise ValueError(kind)


def ramp(stops, t):
    """Interpolate OKLCH stops [(t, L, C, h)] at t."""
    ts = [s[0] for s in stops]
    return [float(np.interp(t, ts, [s[k] for s in stops])) for k in (1, 2, 3)]


DYES = {  # dye: stops from the light outwards (t, L, C, h)
    "liquid": {"warm": [(0, .97, .10, 105), (.25, .90, .17, 95), (.5, .78, .18, 70),
                        (.75, .66, .20, 45), (1.1, .52, .19, 25)],
               "cool": [(0, .86, .17, 150), (.3, .72, .15, 185), (.55, .56, .17, 245),
                        (.8, .45, .19, 262), (1.1, .34, .15, 268)]},
    "brasil": {"warm": [(0, .97, .09, 105), (.35, .91, .18, 98), (.7, .86, .18, 92), (1.1, .80, .17, 88)],
               "cool": [(0, .70, .17, 150), (.35, .58, .16, 152), (.6, .45, .14, 250),
                        (1.1, .33, .14, 262)]},
}
PALETTES = {  # name: dyes in the order dye_field numbers them
    "liquid": [DYES["liquid"]["warm"], DYES["liquid"]["cool"]],
    "liquid-blue": [DYES["liquid"]["cool"], DYES["liquid"]["warm"]],
    "brasil": [DYES["brasil"]["warm"], DYES["brasil"]["cool"]],
    "seventies": [[(0, .96, .11, 100), (.35, .86, .17, 78), (.7, .72, .19, 52), (1.1, .60, .20, 38)],
                  [(0, .82, .15, 350), (.4, .64, .22, 348), (.8, .50, .21, 332), (1.1, .40, .18, 318)],
                  [(0, .93, .18, 122), (.4, .82, .19, 138), (.8, .62, .14, 172), (1.1, .46, .11, 200)]],
    "cathedral": [[(0, .97, .10, 100), (.3, .89, .16, 86), (.7, .76, .17, 66), (1.1, .62, .16, 52)],
                  [(0, .76, .12, 232), (.4, .56, .17, 254), (.8, .42, .19, 264), (1.1, .32, .16, 270)],
                  [(0, .74, .16, 32), (.4, .60, .20, 26), (.8, .47, .18, 16), (1.1, .38, .15, 6)]],
}


def pane_colours(p, phase, r, dyes, seed, jitter=(0.03, 7)):
    """A colour per tile: its dye at its centre, lit by its distance from
    the light, with a little pane-to-pane variation. `phase` numbers the
    dye (a bool picks "cool" over "warm"); `dyes` is a dict or a list."""
    rng = np.random.default_rng(seed + 1)
    if isinstance(dyes, dict):
        dyes = [dyes["warm"], dyes["cool"]]
    cols = []
    for _, y, x in p.tiles:
        y, x = min(max(y, 0), H - 1), min(max(x, 0), W - 1)
        L, Cc, h = ramp(dyes[int(phase[y, x])], r[y, x])
        L += rng.normal(0, jitter[0])
        h += rng.normal(0, jitter[1])
        cols.append((round(float(np.clip(L, .2, .98)) * 50) / 50, round(Cc * 50) / 50,
                     3 * round(h / 3) % 360))
    uniq = sorted(set(cols))
    lut = dict(zip(uniq, safe([oklch_hex(*c) for c in uniq])))
    return [lut[c] for c in cols]


def pane_layers(p, colours, ink_step=0.11, lead=LEAD):
    """Lead in the seams, each tile's ground in its pane colour, its live
    cells a tone darker."""
    return [(lead, np.ones((H, W), bool))] + tile_layers(p.tile, p.live, colours, ink_step)


def tile_layers(tid, live, colours, ink_step):
    """Each tile's cells in its colour, then its live cells a tone darker.
    `colours[k]` belongs to tile k; `tid` is the tile under each cell (-1: none)."""
    uniq = sorted(set(colours))
    inks = safe([tone(c, ink_step) for c in uniq])
    idx = {c: j for j, c in enumerate(uniq)}
    cell = np.full(tid.shape, -1)
    has = tid >= 0
    cell[has] = np.array([idx[c] for c in colours])[tid[has]]
    ls = [(c, cell == j) for j, c in enumerate(uniq)]
    return ls + [(ic, (cell == j) & (live > 0)) for j, ic in enumerate(inks)]


@front("glass")
def f_glass(seed=7, dyes="liquid", scale=8.0, base=2.4, light=1.2, shore=9, **field):
    """Big panes in the middle of the pools, small ones along their shores
    and for the droplets."""
    pools, flips, r = glass_field(seed, **field)
    edge = ndi.distance_transform_edt(pools) + ndi.distance_transform_edt(~pools)   # to the shore
    flips &= edge < shore                   # droplets gather along the shores
    Z = base + edge / scale + light * np.clip(1 - r, 0, 1)
    Z[ndi.binary_dilation(flips, iterations=2)] = 1.6
    p = Packed(np.ones((H, W), bool), seed, Z=Z)
    cols = pane_colours(p, pools ^ flips, r, DYES[dyes], seed)
    return pane_layers(p, cols), dict(live=p.live, tile=p.tile)


@front("liquid")
def f_liquid(kind="vortex", palette="liquid", seed=7, scale=8.0, base=2.4, light=1.2,
             shore=9, drops=2.2, centre=(0.47, 0.45)):
    """Stained glass over any dye field (dye_field): big panes inside each
    dye, small ones along the shores, droplets of the next dye along them."""
    dyes = PALETTES[palette]
    lab, r = dye_field(kind, len(dyes), seed, centre)
    edge = sum(ndi.distance_transform_edt(lab == j) for j in range(len(dyes)))
    rng = np.random.default_rng(seed + 5)
    flips = (warp_noise(rng, 4.5, 0.5, centre) > drops) & (edge < shore)
    lab = np.where(flips, (lab + 1) % len(dyes), lab)
    Z = base + edge / scale + light * np.clip(1 - r, 0, 1)
    Z[ndi.binary_dilation(flips, iterations=2)] = 1.6
    p = Packed(np.ones((H, W), bool), seed, Z=Z)
    return pane_layers(p, pane_colours(p, lab, r, dyes, seed)), dict(live=p.live, tile=p.tile)


# The dragon: tiles in the silhouette, each coloured by the photo's
# luminance under it (gilded plates), over an ECA night sky.
GOLD = [(.40, .075, 70), (.52, .10, 76), (.63, .12, 82), (.73, .13, 88), (.83, .12, 94), (.92, .08, 98)]
NIGHT = ("#27477F", "#1B3363")


@front("dragon")
def f_dragon(seed=11, rule=106, eca_cell=4, halo=3, sky=NIGHT, halo_col="#10203F",
             shading="cells", levels=(7, 6, 5, 4, 3, 2), scale=5.0, blur=1.2, ink=0.12,
             angle=None, margin=6.0, sideways=False):
    """`shading`: "cells" paints the photo's gold cell by cell under the
    tiles (seams two steps darker, live cells one tone darker); "panes"
    gives every tile one gold, the mean under it. `sideways` makes the
    front for the card held in landscape."""
    if sideways:
        with turned():
            ls, info = f_dragon(seed, rule, eca_cell, halo, sky, halo_col, shading, levels,
                                scale, blur, ink, angle, margin)
        return turned.upright(ls, info)
    if angle is None:                   # round 1: the rising layout of dragon-layout
        d = np.load(HERE / "output" / "data" / "dragon_layout.npz")
        mask, lum = d["mask"], d["lum"]
    else:
        mask, lum = dragon_fit(angle, margin)
    p = Packed(mask, seed, levels=levels, scale=scale, base=2.2)
    lum_s = ndi.gaussian_filter(lum, blur)
    lo, hi = np.quantile(lum_s[mask], [0.03, 0.97])
    t = np.clip((lum_s - lo) / (hi - lo), 0, 1)
    step = np.minimum((t * len(GOLD)).astype(int), len(GOLD) - 1)
    golds = safe([oklch_hex(*g) for g in GOLD])
    tid = p.tile
    around = ndi.binary_dilation(mask, iterations=halo)
    ls = [(c, m & ~around) for c, m in eca_layer(rule, eca_cell, seed, sky)]
    ls.append((halo_col, around))
    if shading == "cells":
        inks = safe([tone(g, ink) for g in golds])
        for s_, g in enumerate(golds):
            ls.append((golds[max(s_ - 2, 0)], mask & (step == s_)))            # seams
        for s_, g in enumerate(golds):
            ls.append((g, mask & (step == s_) & ((tid >= 0) | ~ndi.binary_dilation(tid >= 0))))
        for s_, ic in enumerate(inks):
            ls.append((ic, mask & (step == s_) & (p.live > 0)))
        return ls, dict(live=p.live, tile=tid)
    ks = np.flatnonzero(np.bincount(tid[tid >= 0], minlength=len(p.tiles)))
    tile_col = {k: golds[int(round(s))] for k, s in zip(ks, ndi.mean(step, tid, ks))}
    ls.append((safe([oklch_hex(.33, .06, 68)])[0], mask))           # seams: dark bronze
    bare = mask & (tid < 0) & ~ndi.binary_dilation(tid >= 0, iterations=1)
    for s, g in enumerate(golds):                                    # thin parts no tile reached
        ls.append((g, bare & (step == s)))
    colours = [tile_col.get(k, golds[0]) for k in range(len(p.tiles))]
    ls.append((safe([oklch_hex(.26, .05, 65)])[0], mask & (tid < 0) & ~bare))   # lead between plates
    return ls + tile_layers(tid, p.live, colours, ink), dict(live=p.live, tile=tid)


@front("freeform")
def f_freeform(design="shaded", field=NIGHT[1], ink="#E0B04A"):
    """A free-form solve of the dragon (freeform_dragon.py): gold cells on
    the night sky."""
    live = np.load(HERE / "output" / "data" / f"freeform_{design}.npz")["live"]
    assert live.shape == (H, W)
    ls = [(field, np.ones((H, W), bool)), (safe([ink])[0], live > 0)]
    log(f"freeform {design}: {int(live.sum())} live cells, still life {still_life(live)}")
    return ls, dict(live=live)


PROFILE = None                  # the Coated FOGRA39 profile (--profile)

ROUND1 = {
    "conway": [("conway-l4", dict(level=4, grid=44)), ("conway-l3", dict(level=3, grid=58))],
    "paraty": [("paraty-whole", dict(zoom=False)), ("paraty-zoom", dict(zoom=True))],
    "glass": [("glass-liquid", dict(dyes="liquid")), ("glass-brasil", dict(dyes="brasil"))],
    "dragon": [("dragon-gilt", dict(levels=(5, 4, 3, 2), scale=6.0)), ("dragon-plates", {}),
               ("dragon-smooth", dict(levels=(5, 4, 3, 2), scale=6.0, blur=3.0, ink=0.17)),
               ("dragon-panes", dict(shading="panes", levels=(4, 3, 2), scale=6.0, ink=0.15))],
    "freeform": [("dragon-freeform-shaded", dict(design="shaded"))],
}


FINAL = {
    "conway": [("conway", dict(drop=70))],
    "dragon": [("dragon", dict(shading="panes", levels=(4, 3, 2), scale=6.0, ink=0.15, angle=0,
                               sideways=True))],
    "paraty": [("paraty", dict(zoom=False))],
    "liquid": [("liquid-vortex", dict()),
               ("liquid-cathedral", dict(kind="bubbles", palette="cathedral", seed=6, centre=(0.5, 0.3)))],
}

ROUND2 = {
    "conway": [("conway", dict(drop=70))],
    "dragon": [("dragon-level", dict(shading="panes", levels=(4, 3, 2), scale=6.0, ink=0.15, angle=0)),
               ("dragon-rising", dict(shading="panes", levels=(4, 3, 2), scale=6.0, ink=0.15, angle=-25))],
    "paraty": [("paraty", dict(zoom=False))],
    "liquid": [("glass-liquid", dict()),
               ("glass-fingers", dict(kind="fingers", palette="liquid-blue", seed=3, centre=(0.5, 0.52))),
               ("glass-bubbles", dict(kind="bubbles", palette="liquid", seed=4, centre=(0.4, 0.4))),
               ("glass-seventies", dict(kind="vortex", palette="seventies", seed=9, centre=(0.55, 0.42))),
               ("glass-cathedral", dict(kind="rays", palette="cathedral", seed=5, scale=5.0)),
               ("glass-fingers-70s", dict(kind="fingers", palette="seventies", seed=8, centre=(0.45, 0.48))),
               ("glass-bubbles-cathedral", dict(kind="bubbles", palette="cathedral", seed=6, centre=(0.5, 0.3))),
               ("glass-brasil", dict(palette="brasil", seed=12, centre=(0.4, 0.55)))],
}


def cmd_fronts(out: Path, names, variants):
    out.mkdir(parents=True, exist_ok=True)
    C.PROFILE = PROFILE
    for name in names:
        for vname, kw in variants.get(name, [(name, {})]):
            set_canvas(60)
            ls, info = FRONTS[name](**kw)
            ls = compress(ls)
            img = save_png(ls, out / f"{vname}.png")
            preview(img, out / f"{vname}_preview.png")
            np.savez_compressed(out / f"{vname}.npz", rgb=flatten(ls), cpi=CPI, **info)
            log(f"{vname}: {len(ls)} colours")


# --- Backs ---------------------------------------------------------------------------
# The front's field with a band of its pattern along the bottom, mirrored
# left to right so that, held against the light, it lines up with the
# front through the paper (the card turns over its long edge). Above it
# the text, in the thesis cover's inline-code look: black Courier Prime on
# light rounded boxes.

FONTS = ASSETS / "fonts"
TEXT = {
    "title": [("the rich landscape of iterated simplicity", "cellular automata and network automata"),
              ("het weelderige landschap van herhaalde eenvoud", "cellulaire automaten en netwerkautomaten"),
              ("a rica paisagem da simplicidade iterada", "autômatos celulares e autômatos de rede")],
    "thanks": ["thanks for being there!", "bedankt om erbij te zijn!", "valeu pela presença!"],
    "url": "michielrollier.be",
    "dateline": "Michiel Rollier’s PhD defence – 2 October 2026",
}
QR = ASSETS / "qr-michielrollier.png"   # https://michielrollier.be, one pixel per module, made with segno
BOX = "#F3F4F4"
BAND = 0.26                     # the band's share of the page height

BACKS = {  # front file stem: field, band kind, accent (the glider)
    "conway": dict(field=UGENT[0], band=("eca", 106, 12, UGENT), accent=UGENT[1]),
    "paraty": dict(field=PAL["fields"]["apricot"]["hex"], band="tiles",
                   accent=PAL["ramps"]["denim"]["grounds"][6]),
    "glass": dict(field=LEAD, band="tiles", accent="#FFD23F"),
    "dragon": dict(field=NIGHT[0], band=("eca", 106, 4, NIGHT), accent="#E0B04A"),
}
BACKS["liquid"] = BACKS["glass"]


def band_layers(front, spec):
    """The band's paint layers on the back's canvas (already set to the
    front's cell size)."""
    y0 = int(H * (1 - BAND))
    rows = np.arange(H)[:, None] * np.ones((1, W), int)
    if isinstance(spec, tuple):                 # a straight band of the automaton
        _, rule, cell, cols = spec
        on = eca_layer(rule, cell, 3, cols)[1][1]
        return [(cols[1], on & (rows >= y0))]
    rgb, tid = front["rgb"][:, ::-1], front["tile"][:, ::-1]
    n = tid.max() + 1
    top = ndi.minimum(rows, tid, np.arange(n))
    whole = (tid >= 0) & (np.asarray(top)[np.maximum(tid, 0)] >= y0)
    keep = whole | ((tid < 0) & ndi.binary_dilation(whole) & (rows >= y0 - 2))
    flat = rgb.reshape(-1, 3)
    cols, inv = np.unique(flat, axis=0, return_inverse=True)
    inv = inv.reshape(H, W)
    return [("#%02X%02X%02X" % tuple(int(v) for v in c), keep & (inv == j)) for j, c in enumerate(cols)]


def glider(ax, x, y, cell, colour):
    """A glider (heading down and to the right), top-left corner at (x, y) mm."""
    from matplotlib.patches import Rectangle
    for r, c in ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2)):
        ax.add_patch(Rectangle((x + c * cell, y - (r + 1) * cell), cell, cell, facecolor=colour,
                               edgecolor="none", lw=0))


def boxed(ax, x, y, s, size, bold=True, pad=0.55, gap=0.0):
    """One line of text in a light rounded box, the box's top-left at
    (x, y) mm. Returns the box's bottom and right edge."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyBboxPatch
    fp = FontProperties(fname=str(FONTS / ("CourierPrime-Bold.ttf" if bold else "CourierPrime-Regular.ttf")))
    t = ax.text(x, y, s, fontproperties=fp, fontsize=size, color="#000000", ha="left", va="top", zorder=3)
    fig = ax.figure
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    h = bb.height
    px, py = pad * h, 0.22 * h
    t.set_position((x + px, y - py))
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    box = FancyBboxPatch((x, bb.y0 - py), bb.width + 2 * px, bb.height + 2 * py,
                         boxstyle=f"round,pad=0,rounding_size={0.42 * h:.3f}",
                         facecolor=BOX, edgecolor="none", lw=0, zorder=2)
    ax.add_patch(box)
    return bb.y0 - py, x + bb.width + 2 * px


def draw_back(ax, spec, front):
    c, ox, oy = canvas_to_mm()
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (0, 0), PAGE_MM[0], PAGE_MM[1], facecolor=spec["field"], edgecolor="none", lw=0))
    draw_layers(ax, compress_sparse(band_layers(front, spec["band"])))
    from matplotlib.font_manager import FontProperties
    x = BLEED + 8.0
    top = PAGE_MM[1] - BLEED - 8.0
    quiet = tone(spec["field"], 0.40)
    ax.text(x + 0.3, top, TEXT["dateline"], fontsize=6.4, color=quiet, ha="left", va="top",
            fontproperties=FontProperties(fname=str(FONTS / "CourierPrime-Regular.ttf")))
    y = top - 6.0
    for k, (title, sub) in enumerate(TEXT["title"]):
        y, _ = boxed(ax, x, y, title, 8.4)
        y, _ = boxed(ax, x, y - 1.0, sub, 6.6, bold=False)
        y -= 3.4
    band_top = PAGE_MM[1] * BAND
    url_top = band_top + 12.0
    y_th = (y + url_top) / 2 + 9.0
    for s in TEXT["thanks"]:
        y_th, _ = boxed(ax, x, y_th, s, 9.0)
        y_th -= 1.2
    yb, right = boxed(ax, x, url_top, TEXT["url"], 10.0)
    glider(ax, right + 3.0, url_top + 0.1, 1.25, spec["accent"])
    qr_code(ax, PAGE_MM[0] - BLEED - 8.0, yb)


def qr_code(ax, right, bottom, module=0.55, quiet=3):
    """The QR code for the website, black on a light rounded box (its quiet
    zone), its bottom-right corner at (right, bottom) mm."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    q = np.asarray(Image.open(QR).convert("L")) < 128
    n = q.shape[0]
    size = (n + 2 * quiet) * module
    x0, y0 = right - size, bottom
    ax.add_patch(FancyBboxPatch((x0, y0), size, size, boxstyle=f"round,pad=0,rounding_size={1.4}",
                                facecolor=BOX, edgecolor="none", lw=0, zorder=2))
    for r_, c_ in zip(*np.nonzero(q)):
        ax.add_patch(Rectangle((x0 + (quiet + c_) * module, y0 + size - (quiet + r_ + 1) * module),
                               module, module, facecolor="#000000", edgecolor="none", lw=0, zorder=3))


def compress_sparse(layers):
    """Drop empty layers and merge repeats (the band's layers never overlap)."""
    merged = {}
    for col, m in layers:
        if m.any():
            merged[col] = merged[col] | m if col in merged else m
    return list(merged.items())


def front_layers(front):
    """Paint layers back from a front's cell colours."""
    rgb = front["rgb"]
    cols, inv = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    inv = inv.reshape(rgb.shape[:2])
    return [("#%02X%02X%02X" % tuple(int(v) for v in c), inv == j) for j, c in enumerate(cols)]


def cmd_cards(src: Path, out: Path, stems):
    """Front and back of each card: a two-page PDF (111 x 154 mm, 3 mm
    bleed, vector), the back as a 600 dpi PNG, and a preview of both."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        front = dict(np.load(src / f"{stem}.npz"))
        set_canvas(int(front["cpi"]))
        spec = BACKS[stem.split("-")[0]]
        with PdfPages(out / f"{stem}.pdf") as pdf:
            fig, ax = new_page()
            draw_layers(ax, front_layers(front))
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = new_page()
            draw_back(ax, spec, front)
            pdf.savefig(fig)
            fig.savefig(out / f"{stem}_back.png", dpi=600)
            plt.close(fig)
        log(f"{stem}: pdf {(out / f'{stem}.pdf').stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["dragon-layout", "fronts", "cards"])
    ap.add_argument("--src", type=Path, default=HERE / "output" / "round1")
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", type=Path, default=HERE / "output")
    ap.add_argument("--profile", type=Path, help="Coated FOGRA39 ICC profile")
    a = ap.parse_args()
    global PROFILE
    PROFILE = a.profile
    if a.cmd == "dragon-layout":
        cmd_dragon_layout(a.out)
    elif a.cmd == "fronts":
        rounds = {1: ROUND1, 2: ROUND2, 3: FINAL}[a.round]
        cmd_fronts(a.out, a.names or list(rounds), rounds)
    elif a.cmd == "cards":
        cmd_cards(a.src, a.out, a.names)


if __name__ == "__main__":
    main()
