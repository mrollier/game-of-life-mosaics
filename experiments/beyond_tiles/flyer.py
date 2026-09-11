"""PhD-defence flyer backgrounds: free-form still lifes on pond-tile frames.

Portrait, 1080x1536 px: the vignette at 360x512 cells and 3 px per cell, the
skyline at 540x768 cells and 2 px per cell. One size for WhatsApp, e-mail and
a Facebook event; the text goes on afterwards in Canva, so every design keeps
the middle of the canvas close to empty.

The palette is the one from `figures/linkedin_halo_filling.png`: a teal field
with cream pond tiles and a haze tone for the filler, a cream field with slate
cells for the free-form still life.

Each *base design* is a synthetic image, not a photo: a greyscale tone target
(black = dense, white = empty) plus a mask that says where the free-form still
life goes and where the tile mosaic goes. Dark = tiles, mid = free-form
graded by the tone, light = empty, so the flyer reads as one field thinning
out towards the text.

    python experiments/beyond_tiles/flyer.py preview          # base designs, no solver
    python experiments/beyond_tiles/flyer.py solve [name ...]  # CP-SAT, minutes each
    python experiments/beyond_tiles/flyer.py polish [name ...] # extra LNS rounds
    python experiments/beyond_tiles/flyer.py render            # the ten flyers + sheet

Solves land under `results/flyer/<design>/` (gitignored), renders under
`output/images/flyer/`. Every render is also merged with its tile field,
verified as a single still life, and exported to `output/golly/`.

Lessons from the first round (2026-09-09), all folded in below:

* **Seams.** `solve_strips` leaves two dead rows at every strip boundary and
  the generic polish repairs them only as far as its budget goes; at a peak
  density of 0.40 the dense band sits on the still-life packing ceiling, the
  polish stalls, and the seams read as faint horizontal stripes. The peak is
  now 0.32, the strips are 64 rows instead of 48 (seven seams, not ten), the
  polish gets three rounds, and `seam_report` measures what is left.
* **Dotted lines.** Rounding each window's target gives every window along a
  slow fade the same one-cell target, and the solver answers with one block
  per window, on the window lattice: a perfectly regular dotted line at the
  edge of the empty zone. Error-diffused targets (`--dither fs`) vary from
  window to window and the line dissolves.
* **Glow, not distance.** A distance transform throws star-like streaks off
  every thin spire; a Gaussian of the silhouette weights a spire by its area.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from contextlib import contextmanager

import numpy as np
from PIL import Image, ImageDraw

from beyond_tiles.lns import _MARGIN as LNS_MARGIN
from beyond_tiles.lns import _solve_patch_task as ORIGINAL_PATCH_TASK
from gol_mosaics import ColorScheme

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT, SCALE = 360, 512, 3
D_MAX = 0.32  # peak free-form density; 0.40 was heavier than wanted and unpolishable
STRIP_ROWS = 64  # strip height; the seam pass derives the separator rows from it

# The halo-figure palette (TEAL_HAZE in linkedin_banners.py), verbatim.
FLYER = ColorScheme(gol_background="#F4EDE2", gol_pixel="#2B3538",
                    eca_background="#1B5E5E", eca_pixel="#E8D9C0",
                    fill_pixel="#7FA8A0")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- Base designs -------------------------------------------------------------

@dataclass
class Design:
    name: str
    grey: np.ndarray   # uint8 (rows, cols); the solver's tone target
    free: np.ndarray   # bool, True on the free-form subject
    note: str = ""
    scale: int = SCALE  # px per cell; rows*scale x cols*scale is the flyer


def smoothstep(x, lo, hi):
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def to_grey(density: np.ndarray) -> np.ndarray:
    """Normalised density 0..1 -> grey; 1 is black and maps to D_MAX."""
    return np.round(255.0 * (1.0 - np.clip(density, 0.0, 1.0))).astype(np.uint8)


def grid():
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    return yy.astype(np.float64), xx.astype(np.float64)


def rounded_window(top, bottom, left, right, radius):
    """True inside a rounded rectangle inset by the given margins."""
    yy, xx = grid()
    y0, y1 = top + radius, HEIGHT - bottom - 1 - radius
    x0, x1 = left + radius, WIDTH - right - 1 - radius
    dy = np.maximum(np.maximum(y0 - yy, yy - y1), 0.0)
    dx = np.maximum(np.maximum(x0 - xx, xx - x1), 0.0)
    return np.hypot(dx, dy) <= radius


def distance_fade(free, fade, gamma=1.4):
    """Dense at the tile boundary, empty `fade` cells away from it."""
    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(free)
    return (1.0 - smoothstep(dist, 0.0, fade)) ** gamma


def glow(tiles, sigma, gamma=1.2):
    """A soft halo that follows the mass of the silhouette, not its spikes.

    A straight edge gives 0.5 at the boundary, which is normalised to full
    density; concave crevices between buildings saturate.
    """
    from scipy.ndimage import gaussian_filter

    soft = gaussian_filter(tiles.astype(np.float64), sigma, mode="nearest")
    return np.clip(soft / 0.5, 0.0, 1.0) ** gamma


def design_vignette() -> Design:
    """A rounded window in a tile frame; grain thickens towards the frame."""
    free = rounded_window(top=64, bottom=64, left=44, right=44, radius=48)
    yy, xx = grid()
    a, b = (WIDTH - 88) / 2, (HEIGHT - 128) / 2
    u, v = (xx - (WIDTH - 1) / 2) / a, (yy - (HEIGHT - 1) / 2) / b
    d = (np.abs(u) ** 3 + np.abs(v) ** 3) ** (1 / 3)
    density = smoothstep(d, 0.40, 1.0) ** 1.8
    return Design("vignette", to_grey(density), free,
                  "rounded window, superellipse vignette")


def design_vignette2() -> Design:
    """The vignette after the first review: sharper corners, narrower frame.

    Margins 50 (top/bottom) and 34 (sides) instead of 64 and 44, corner
    radius 22 instead of 48, and the grain starts fading further out, so the
    clean centre grows from roughly 136 to 180 cells wide. A level-4 diamond
    row (24-cell box plus the 2-cell gap on each side) still fits inside the
    34-cell side strips.
    """
    top, side, radius = 50, 34, 22
    free = rounded_window(top=top, bottom=top, left=side, right=side,
                          radius=radius)
    yy, xx = grid()
    a, b = (WIDTH - 2 * side) / 2, (HEIGHT - 2 * top) / 2
    u, v = (xx - (WIDTH - 1) / 2) / a, (yy - (HEIGHT - 1) / 2) / b
    d = (np.abs(u) ** 3 + np.abs(v) ** 3) ** (1 / 3)
    density = smoothstep(d, 0.52, 1.0) ** 1.8
    return Design("vignette2", to_grey(density), free,
                  "narrower frame, sharper corners")


def design_corners() -> Design:
    """The supplied gradient: its darkest fifth becomes tiles. (Dropped.)"""
    src = Image.open(REPO / "input/images/flyer/background_gradient.png")
    g = np.asarray(src.convert("L").resize((WIDTH, HEIGHT),
                                           Image.Resampling.LANCZOS),
                   dtype=np.float64) / 255.0
    t_low = float(np.quantile(g, 0.20))
    free = g > t_low
    density = (1.0 - smoothstep(g, t_low, 0.96)) ** 1.3
    return Design("corners", to_grey(density), free,
                  f"input gradient, tiles below grey {t_low:.2f}")


# Skylines are drawn on a PIL canvas in cell units. Heights are measured
# upwards from a ground line; the drawing helpers translate. Both cities get
# the same width, the same base band and six landmarks each.
GROUND = 30  # cells of base band under the buildings


class Sky:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT,
                 ground: int = GROUND):
        self.img = Image.new("L", (width, height), 0)
        self.d = ImageDraw.Draw(self.img)
        self.yg = height - ground

    def box(self, x0, x1, h0, h1):
        self.d.rectangle([x0, self.yg - h1, x1 - 1, self.yg - h0 - 1], fill=255)

    def spire(self, x0, x1, h0, h1):
        self.d.polygon([(x0, self.yg - h0), (x1 - 1, self.yg - h0),
                        ((x0 + x1 - 1) / 2, self.yg - h1)], fill=255)

    def crenels(self, x0, x1, h, merlon=3, gap=3, height=3):
        for x in range(x0, x1, merlon + gap):
            self.box(x, min(x + merlon, x1), h, h + height)

    def gable(self, x0, x1, h0, h1, n):
        dx, dh = (x1 - x0) / (2 * n), (h1 - h0) / n
        for i in range(n):
            self.box(round(x0 + i * dx), round(x1 - i * dx),
                     round(h0 + i * dh), round(h0 + (i + 1) * dh))

    def wave(self, x0, x1, h, amp, periods=1.0):
        """A slab whose roofline is a sine wave: the Copan's curve, in profile."""
        pts = [(x0, self.yg - 1)]
        for xx in range(x0, x1):
            t = (xx - x0) / (x1 - 1 - x0)
            pts.append((xx, self.yg - h - amp * np.sin(2 * np.pi * periods * t)))
        pts.append((x1 - 1, self.yg - 1))
        self.d.polygon(pts, fill=255)

    def line(self, x0, h0, x1, h1, width):
        self.d.line([(x0, self.yg - h0), (x1, self.yg - h1)], fill=255,
                    width=width)

    def roof(self, x0, x1, h0, h1, inset):
        """A pitched roof seen from the side: a flat ridge, sloping ends."""
        self.d.polygon([(x0, self.yg - h0), (x1 - 1, self.yg - h0),
                        (x1 - 1 - inset, self.yg - h1), (x0 + inset, self.yg - h1)],
                       fill=255)

    def cut(self, x0, x1, h0, h1):
        """An opening in the silhouette that stays empty: a slit, a span."""
        if not hasattr(self, "holes"):
            self.holes = Image.new("L", self.img.size, 0)
            self.hd = ImageDraw.Draw(self.holes)
        self.hd.rectangle([x0, self.yg - h1, x1 - 1, self.yg - h0 - 1], fill=255)

    def mask(self) -> np.ndarray:
        m = np.asarray(self.img) > 0
        m[self.yg:, :] = True
        if hasattr(self, "holes"):
            m &= ~(np.asarray(self.holes) > 0)
        return m

    def hole_mask(self) -> np.ndarray:
        if hasattr(self, "holes"):
            return np.asarray(self.holes) > 0
        return np.zeros((self.img.height, self.img.width), dtype=bool)


def ghent(s: Sky, x: int) -> None:
    """Six Ghent landmarks in 180 cells, west to east.

    Gravensteen, the Graslei's stepped gables, St Nicholas' Church, the
    Belfry, St Bavo's Cathedral and UGent's Boekentoren. Heights follow the
    real ones loosely (Belfry 91 m, St Bavo 89 m, St Nicholas 76 m,
    Boekentoren 64 m), so the three towers still read as the famous row.
    """
    # 1. Gravensteen: curtain wall, corner turret, keep, all crenellated.
    s.box(x + 2, x + 40, 0, 18); s.crenels(x + 2, x + 40, 18)
    s.box(x + 2, x + 9, 0, 30); s.crenels(x + 2, x + 9, 30, 2, 2, 2)
    s.box(x + 20, x + 38, 0, 42); s.crenels(x + 20, x + 38, 42)
    # 2. Graslei houses with stepped gables.
    for x0, x1, h, top in ((42, 54, 26, 36), (55, 66, 22, 32), (67, 78, 24, 31)):
        s.box(x + x0, x + x1, 0, h); s.gable(x + x0, x + x1, h, top, 4)
    # 3. St Nicholas' Church: nave, crossing tower, spire, corner turrets.
    s.box(x + 80, x + 112, 0, 36); s.gable(x + 80, x + 112, 36, 46, 3)
    s.box(x + 90, x + 102, 0, 88); s.spire(x + 90, x + 102, 88, 110)
    s.box(x + 88, x + 90, 0, 94); s.box(x + 102, x + 104, 0, 94)
    # 4. The Belfry: tower, gallery, spire, finial.
    s.box(x + 116, x + 132, 0, 96); s.box(x + 114, x + 134, 96, 104)
    s.spire(x + 116, x + 132, 104, 132); s.box(x + 123, x + 125, 130, 138)
    # 5. St Bavo's Cathedral: nave, flat-topped tower with four pinnacles.
    s.box(x + 136, x + 162, 0, 44); s.gable(x + 136, x + 162, 44, 56, 3)
    s.box(x + 142, x + 158, 0, 120)
    for px in (142, 147, 152, 156):
        s.box(x + px, x + px + 2, 120, 128)
    # 6. Boekentoren: the slender book tower with its belvedere, and the low
    #    reading-room wing at its foot.
    s.box(x + 163, x + 166, 0, 20)
    s.box(x + 166, x + 178, 0, 92); s.box(x + 168, x + 176, 92, 98)


def sao_paulo(s: Sky, x: int) -> None:
    """Six Sao Paulo landmarks in 180 cells, west to east.

    MASP, the Copan, Edificio Italia, the Banespa tower, USP's Torre do
    Relogio and the Ponte Estaiada. Heights follow the real ones loosely
    (Italia 165 m, Banespa 161 m, the bridge's pylon 138 m, Copan 115 m).
    """
    # 1. MASP: the slab hangs inside two portal frames over an open span.
    s.box(x + 0, x + 40, 16, 34); s.box(x + 0, x + 6, 0, 16); s.box(x + 34, x + 40, 0, 16)
    # 2. Copan: a wide slab whose roofline carries the building's curve.
    s.wave(x + 43, x + 77, 78, 5); s.box(x + 54, x + 66, 82, 88)
    # 3. Edificio Italia.
    s.box(x + 80, x + 94, 0, 116); s.box(x + 84, x + 90, 116, 122)
    # 4. Banespa: stepped tower with a spire.
    s.box(x + 97, x + 117, 0, 60); s.box(x + 100, x + 114, 60, 88)
    s.box(x + 103, x + 111, 88, 108); s.box(x + 106, x + 108, 108, 124)
    # 5. Torre do Relogio: USP's slender clock tower, the clock block near the top.
    s.box(x + 121, x + 129, 0, 62); s.box(x + 119, x + 131, 54, 68)
    # 6. Ponte Estaiada: X pylon, deck, a fan of cables from each arm to the
    #    deck on its own side.
    b = x + 133
    s.line(b + 1, 0, b + 45, 96, 6); s.line(b + 45, 0, b + 1, 96, 6)
    s.box(b - 3, b + 47, 16, 20)
    for k in range(4):
        s.line(b + 3, 94, b - 1 + 8 * k, 20, 2)
        s.line(b + 43, 94, b + 47 - 8 * k, 20, 2)


def houses(s: Sky, x0: int, x1: int, seed: int) -> None:
    """A filler row of gabled houses: Ghent's ordinary street."""
    rng = np.random.default_rng(seed)
    x = x0
    while x < x1 - 6:
        w = min(int(rng.integers(8, 14)), x1 - x)
        h = int(rng.integers(16, 28))
        s.box(x, x + w, 0, h)
        s.gable(x, x + w, h, h + int(rng.integers(6, 10)), 3)
        x += w + 1


def towers(s: Sky, x0: int, x1: int, seed: int) -> None:
    """A filler row of flat-roofed towers: Sao Paulo's ordinary street."""
    rng = np.random.default_rng(seed)
    x = x0
    while x < x1 - 6:
        w = min(int(rng.integers(8, 16)), x1 - x)
        h = int(rng.integers(30, 72))
        s.box(x, x + w, 0, h)
        if rng.random() < 0.4:
            s.box(x + w // 2, x + w // 2 + 2, h, h + 8)
        x += w + 2


def design_skyline() -> Design:
    """Ghent on the left half, Sao Paulo on the right, a straight band on top."""
    s = Sky()
    ghent(s, 0)
    sao_paulo(s, 180)
    tiles = s.mask()
    tiles[:40, :] = True
    free = ~tiles
    density = glow(tiles, sigma=30, gamma=1.5)
    return Design("skyline", to_grey(density), free,
                  "Ghent left, Sao Paulo right, band on top")


def design_antipodes() -> Design:
    """Ghent along the bottom; Sao Paulo hangs upside-down from the top.

    Each city gets its six landmarks centred plus 88 cells of its ordinary
    street on either side, so the two halves carry equal weight.
    """
    g = Sky()
    houses(g, 0, 88, seed=3); ghent(g, 90); houses(g, 272, 360, seed=4)
    p = Sky()
    towers(p, 0, 88, seed=5); sao_paulo(p, 90); towers(p, 272, 360, seed=6)
    tiles = g.mask() | np.flipud(p.mask())
    free = ~tiles
    density = glow(tiles, sigma=30, gamma=1.5)
    return Design("antipodes", to_grey(density), free,
                  "Ghent below, Sao Paulo inverted above")


# --- Skyline, second version ------------------------------------------------------
#
# After the review of the first: only landmarks anyone would recognise, at
# 540x768 cells (2 px per cell, so the same 1080x1536 flyer), and every
# tower's centre and width snapped to the level-4 diamond lattice so that a
# symmetric stack of whole diamonds sits inside it. The lattice at level L
# has box corners at rows 3L*s - py and columns 3L*m - px with s and m of
# equal parity, where (py, px) is the padding `symmetric_pad` chooses for
# this canvas; a tower centred on column 3L*m - px + (3L - 0.5) that is
# 3L*(2k+2) + 4 cells wide holds 2k+1 stacks with the 2-cell gap on each
# side, and a flat roof at row 3L*s - py - 2 (right parity) closes on a
# whole diamond. Spires, gables and crenellations sit above that.
SKY2 = dict(height=768, width=540, ground=40, ribbon=56, scale=2, level=4)


class Lattice:
    """Where the level-L diamonds fall on a canvas, after `symmetric_pad`."""

    def __init__(self, height, width, level=4):
        self.n = 3 * level
        self.py, self.px = symmetric_pad(level, "diamond", "auto", height, width)

    def centre(self, m):
        """Column centre of lattice column m (a half-integer)."""
        return self.n * m - self.px + self.n - 0.5

    def span(self, m, stacks):
        """x0, x1 of a building holding `stacks` (odd) diamond stacks on m.

        One stack is a 2n box plus the 2-cell gap each side; every further
        pair of stacks sits n cells out on either side.
        """
        w = self.n * (stacks + 1) + 4
        x0 = int(round(self.centre(m) - w / 2 + 0.5))
        return x0, x0 + w

    def m_near(self, x):
        return int(round((x - self.n + 0.5 + self.px) / self.n))

    def roof(self, yg, h, m):
        """Nearest height at which a flat roof closes on a whole diamond."""
        best = None
        for sgn in (1, 0):
            # box top row R = n*s - py with s == m (mod 2); roof at R - 2
            target = yg - h + 2 + self.py
            s0 = int(round(target / self.n))
            for s in (s0 - 1, s0, s0 + 1, s0 + 2):
                if (s - m) % 2:
                    continue
                cand = yg - (self.n * s - self.py - 2)
                if best is None or abs(cand - h) < abs(best - h):
                    best = cand
        return best


class Sky3(Sky):
    """`Sky` with centred, mirror-symmetric primitives and a `front()` scope.

    Every building sits on a lattice column, whose centre is a half-integer,
    so anything drawn with `cbox`/`cspire` and an even width is symmetric to
    the cell. Holes are kept as an array. A building drawn inside
    `with s.front():` stands in front of what was there: a two-cell gap is
    cut around its silhouette wherever it overlaps an earlier building, so
    the two read as separate bodies at different depths.
    """

    def __init__(self, width, height, ground):
        super().__init__(width, height, ground)
        self.holes = np.zeros((height, width), dtype=bool)

    # -- centred primitives; c is a lattice centre (a half-integer) --
    def cbox(self, c, w, h0, h1):
        x0 = int(round(c + 0.5 - w / 2))
        self.box(x0, x0 + w, h0, h1)

    def cspire(self, c, w, h0, h1, w_top=2):
        """A symmetric spire, one row at a time, so both flanks match."""
        for r in range(h0, h1):
            t = (r - h0) / max(1, h1 - h0)
            wr = max(w_top, 2 * int(round((w + (w_top - w) * t) / 2)))
            self.cbox(c, wr, r, r + 1)

    def ccrenels(self, c, w, h, merlon=3, gap=3, height=3):
        k = max(1, int((w + gap) // (merlon + gap)))
        total = k * merlon + (k - 1) * gap
        x0 = int(round(c + 0.5 - total / 2))
        for i in range(k):
            x = x0 + i * (merlon + gap)
            self.box(x, x + merlon, h, h + height)

    def flag(self, c, h0, h1, w=10, fh=7):
        """A two-cell pole centred on c, the flag flying to the right."""
        self.cbox(c, 2, h0, h1)
        x = int(round(c + 0.5 + 1))
        self.box(x, x + w, h1 - fh, h1)
        self.box(x + w, x + w + 1, h1 - fh + 2, h1 - 2)  # the free edge ripples

    def cut(self, x0, x1, h0, h1):
        self.holes[self.yg - h1:self.yg - h0, x0:x1] = True

    def mask(self) -> np.ndarray:
        m = np.asarray(self.img) > 0
        m[self.yg:, :] = True
        return m & ~self.holes

    def hole_mask(self) -> np.ndarray:
        return self.holes.copy()

    @contextmanager
    def front(self):
        from scipy.ndimage import binary_dilation

        back = np.asarray(self.img) > 0
        img, holes = self.img, self.holes
        self.img = Image.new("L", img.size, 0)
        self.d = ImageDraw.Draw(self.img)
        self.holes = np.zeros_like(holes)
        yield
        sil = np.asarray(self.img) > 0
        ring = binary_dilation(sil, iterations=2) & ~sil & back
        holes[sil] = False              # nothing behind shows through
        holes |= ring | self.holes
        merged = Image.fromarray(((back | sil) * 255).astype(np.uint8))
        self.img, self.d, self.holes = merged, ImageDraw.Draw(merged), holes


def ghent2(s: Sky3, L: Lattice) -> None:
    """Gravensteen, three Graslei gables, St Nicholas, the Belfry, St Bavo.

    Drawn against the photos. Heights within the city follow the real ones
    (Belfry 95 m with the dragon, St Bavo 89, St Nicholas 76, the keep of
    the Gravensteen and the guild houses far below), with the whole city
    raised so the Belfry's dragon reaches 171 cells against Edificio
    Italia's 224-cell roof.
    """
    yg = s.yg

    # Gravensteen: the keep at the left edge, crenellated, with a round
    # bartizan at each corner rising above the parapet, and the Flemish
    # lion on a pole in the middle. A stub of curtain wall to its right.
    c = L.centre(2)
    s.cbox(c, 36, 0, 44); s.ccrenels(c, 36, 44, 3, 3, 4)
    for tc in (c - 14, c + 14):
        s.cbox(tc, 8, 0, 52); s.ccrenels(tc, 8, 52, 2, 2, 3)
    s.flag(c, 44, 66, w=10, fh=7)
    s.box(36, 42, 0, 16); s.crenels(36, 42, 16, 2, 2, 2)
    # Graslei: three guild houses, stepped gables, the middle one taller
    # with a finial. About a quarter of the Belfry, as in life.
    for m, h, top, steps in ((5, 26, 12, 4), (7, 32, 12, 4), (9, 28, 12, 4)):
        c = L.centre(m)
        x0 = int(round(c + 0.5 - 12))
        s.box(x0, x0 + 24, 0, h); s.gable(x0, x0 + 24, h, h + top, steps)
        if m == 7:
            s.cbox(c, 2, h + top, h + top + 4)
    # St Nicholas' Church: symmetric about the crossing tower. The nave with
    # its pitched roof, a pointed turret at either end, the tower with four
    # pointed corner turrets (two seen), the pyramidal roof and the cross.
    c = L.centre(13)
    x0 = int(round(c + 0.5 - 30))
    s.box(x0, x0 + 60, 0, 40); s.roof(x0, x0 + 60, 40, 58, 9)
    for tc in (c - 26, c + 26):
        s.cbox(tc, 6, 0, 60); s.cspire(tc, 6, 60, 70)
    th = L.roof(yg, 100, 13)
    s.cbox(c, 28, 0, th)
    for tc in (c - 13, c + 13):
        s.cbox(tc, 4, th - 12, th + 8); s.cspire(tc, 4, th + 8, th + 16)
    s.cspire(c, 28, th, th + 26)
    s.cbox(c, 2, th + 26, th + 34); s.cbox(c, 6, th + 30, th + 32)
    # St Bavo's Cathedral: three stages widening a little towards the
    # ground, a flat top with the two big corner pinnacles and the slim
    # central flagpole, low aisles either side; the pinnacle on the east
    # aisle's corner (the west one stands behind the Belfry).
    c = L.centre(20)
    th = L.roof(yg, 130, 20)
    s.cbox(c, 36, 0, 52); s.cbox(c, 32, 0, 92); s.cbox(c, 28, 0, th)
    for tc in (c - 12, c + 12):
        s.cbox(tc, 6, th, th + 14); s.cspire(tc, 6, th + 14, th + 24)
    s.cbox(c, 2, th, th + 30)
    s.cbox(c - 22, 8, 0, 40); s.cbox(c + 22, 8, 0, 40)
    s.cbox(c + 24, 4, 40, 52); s.cspire(c + 24, 4, 52, 58)
    # The Belfry, its base touching the nave and the cathedral (the user's
    # call: no gap between the towers): a straight shaft, buttressed only
    # near the ground; the gallery block a little wider than the shaft and
    # about one shaft-width tall, four corner turrets with pointed caps,
    # the spire, and the dragon on its pole, a few cells of it, facing east.
    c = L.centre(17)
    th = L.roof(yg, 114, 17)
    s.cbox(c, 36, 0, 50); s.cbox(c, 28, 0, th)
    s.cbox(c, 34, th, th + 24)
    for tc in (c - 15, c + 15):
        s.cbox(tc, 4, th + 24, th + 32); s.cspire(tc, 4, th + 32, th + 38)
    s.cspire(c, 26, th + 24, th + 50)
    s.cbox(c, 2, th + 50, th + 54)
    x = int(round(c + 0.5 - 3))                       # the dragon
    s.box(x, x + 6, th + 54, th + 55)                 # body
    s.box(x, x + 1, th + 55, th + 56)                 # tail up
    s.box(x + 2, x + 4, th + 55, th + 56); s.box(x + 3, x + 4, th + 56, th + 57)  # wing
    s.box(x + 5, x + 7, th + 55, th + 56); s.box(x + 6, x + 7, th + 56, th + 57)  # head


def sao_paulo2(s: Sky3, L: Lattice) -> None:
    """The Copan, MASP in front of it, Edificio Italia, Banespa, the bridge.

    One metre is 1.36 cells: Edificio Italia's roof at 224, the Copan at
    about 156, Banespa's mast at 220, the pylon of the Ponte Estaiada at
    184, MASP's box hung seven cells above the ground.
    """
    yg = s.yg

    # Copan: the long curved slab. Seen face on, the curve shows in the
    # roofline, which rises and dips once across the width, and in the two
    # continuous bands that follow it. No rooftop cabin.
    c = L.centre(27)
    x0 = int(round(c + 0.5 - 40))
    x1 = x0 + 80

    def bend(x):
        t = (x - x0) / (x1 - 1 - x0)
        return 6.0 * np.sin(2 * np.pi * (t + 0.05)) + 4.0 * t

    for x in range(x0, x1):
        s.box(x, x + 1, 0, int(round(152 + bend(x))))
    for hb in (66, 108):
        for x in range(x0, 350):
            h = int(round(hb + bend(x)))
            s.cut(x, x + 1, h, h + 3)
    # MASP in front of the Copan: the glass box hung between its two portal
    # frames, the span underneath open, a gap cut around it where it
    # overlaps the Copan so it stands clear.
    with s.front():
        x0, x1 = 262, 338
        s.box(x0, x1, 7, 27)
        s.box(x0, x0 + 7, 0, 7); s.box(x1 - 7, x1, 0, 7)
        s.cut(x0 + 7, x1 - 7, 0, 7)
    # Edificio Italia, next to the Copan as in life: the broad slab with the
    # service core as a solid vertical stripe between two slits, the
    # machine room on the core, and a proper antenna with cross-arms.
    c = L.centre(33)
    x0 = int(round(c + 0.5 - 38))
    h = L.roof(yg, 224, 33)
    s.box(x0, x0 + 76, 0, h)
    xc = int(round(c + 0.5))
    s.cut(xc - 6, xc - 4, 0, h); s.cut(xc + 4, xc + 6, 0, h)
    s.cbox(c, 16, h, h + 8)
    s.cbox(c, 4, h + 8, h + 34)
    s.cbox(c, 10, h + 16, h + 18); s.cbox(c, 8, h + 24, h + 26)
    # Banespa (Farol Santander): broad base, the shaft, then the stepped
    # crown, the lantern, the mast and the flag.
    c = L.centre(38)
    s.cbox(c, 44, 0, L.roof(yg, 92, 38))
    s.cbox(c, 36, 0, L.roof(yg, 131, 38))
    s.cbox(c, 28, 0, 158); s.cbox(c, 20, 0, 176); s.cbox(c, 14, 0, 186)
    s.cbox(c, 8, 186, 200)
    s.flag(c, 200, 220, w=10, fh=7)
    # Ponte Estaiada at the right edge: the X pylon with the crossbeam that
    # joins the upper arms, the deck, a fan of stays from each arm to the
    # deck on its side, and the two stays that cross under the beam.
    b, w, H = 494, 42, 184
    s.line(b, 0, b + w, H, 8); s.line(b + w, 0, b, H, 8)
    yb = int(0.78 * H)
    s.box(b + 8, b + w - 8, yb, yb + 6)
    s.box(468, s.img.width, 28, 34)
    for k in range(4):
        y = H - 4 - 10 * k
        xl, xr = b + w * (1 - y / H), b + w * y / H
        land = b - 20 + 6 * k
        s.line(xl, y, land, 34, 3)
        s.line(xr, y, 2 * b + w - land, 34, 3)
    s.line(b + 3, H - 6, b + w + 6, 34, 3); s.line(b + w - 3, H - 6, b - 6, 34, 3)


def design_skyline2() -> Design:
    """Ghent left, Sao Paulo right, on the lattice; a clean ribbon on top."""
    H, W = SKY2["height"], SKY2["width"]
    L = Lattice(H, W, SKY2["level"])
    s = Sky3(W, H, SKY2["ground"])
    ghent2(s, L)
    sao_paulo2(s, L)
    tiles = s.mask()
    tiles[:SKY2["ribbon"], :] = True
    free = ~tiles
    density = glow(tiles, sigma=45, gamma=1.5)
    density[s.hole_mask()] = 0.0   # slits, spans and the gaps stay clean
    return Design("skyline2", to_grey(density), free,
                  "iconic buildings on the lattice, 540x768", scale=SKY2["scale"])


def design_ripples() -> Design:
    """Two cities as two wave sources; their ripples meet across the flyer. (Dropped.)"""
    yy, xx = grid()
    sources = [((HEIGHT - 32), 40, 64), (32, WIDTH - 40, 64)]  # (cy, cx, radius)
    tiles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    density = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    wavelength = 22.0
    for cy, cx, r in sources:
        dist = np.hypot(yy - cy, xx - cx)
        tiles |= dist <= r
        out = np.maximum(dist - r, 0.0)
        amp = (1.0 - smoothstep(out, 0.0, 170.0)) ** 1.3
        ring = 0.5 + 0.5 * np.cos(2 * np.pi * out / wavelength)
        density += amp * (0.25 + 0.75 * ring)
    free = ~tiles
    return Design("ripples", to_grey(np.clip(density, 0, 1)), free,
                  "two discs, interfering ripples")


def design_arch() -> Design:
    """A pointed arch in a tile frame, outlined in grain."""
    yy, xx = grid()
    left, right, bottom, apex = 44, WIDTH - 44, HEIGHT - 64, 64
    half = (right - left) / 2
    radius = 180.0
    rise = np.sqrt(radius ** 2 - (radius - half) ** 2)
    spring = apex + rise
    inside = (xx >= left) & (xx < right) & (yy >= spring) & (yy < bottom)
    cl, cr = right - radius, left + radius
    arc = ((np.hypot(xx - cl, yy - spring) <= radius) &
           (np.hypot(xx - cr, yy - spring) <= radius) & (yy < spring))
    free = inside | arc
    density = distance_fade(free, fade=60, gamma=1.5)
    return Design("arch", to_grey(density), free,
                  f"pointed arch, springline at row {spring:.0f}")


DESIGNS = {
    "vignette": design_vignette,
    "vignette2": design_vignette2,
    "skyline": design_skyline,
    "skyline2": design_skyline2,
    "antipodes": design_antipodes,
    "arch": design_arch,
    # Dropped after the first round; kept so their solves still render.
    "corners": design_corners,
    "ripples": design_ripples,
}
ACTIVE = ["vignette2", "skyline2"]


def preview(names) -> Path:
    """The base designs side by side: tone target with the tile region in teal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = REPO / "output/images/flyer"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 4.4))
    for ax, name in zip(np.atleast_1d(axes), names):
        d = DESIGNS[name]()
        rgb = np.repeat(d.grey[..., None], 3, axis=2).astype(np.float64)
        rgb[~d.free] = (27, 94, 94)
        ax.imshow(rgb.astype(np.uint8))
        ax.set_title(f"{d.name}\n{d.note}", fontsize=7)
        ax.axis("off")
        share = float(d.free.mean())
        h, w = d.grey.shape
        centre = d.grey[h // 2 - int(0.225 * h):h // 2 + int(0.225 * h),
                        w // 2 - int(0.3 * w):w // 2 + int(0.3 * w)]
        print(f"{name:10s} free {share:.1%}, mean density "
              f"{(1 - d.grey.mean() / 255) * D_MAX:.3f}, centre box mean "
              f"{(1 - centre.mean() / 255) * D_MAX:.4f}")
    fig.tight_layout(pad=0.4)
    path = out / "_preview_designs.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# --- Solve --------------------------------------------------------------------

def seam_report(pattern: np.ndarray, strip_rows: int,
                free: np.ndarray = None, verbose: bool = False) -> float:
    """Density of the strip separator rows relative to the rows around them.

    Densities are taken over free cells only, so a separator that falls inside
    the tile region (no free cells) is skipped rather than reported as empty,
    and a seam next to the frame is compared with the free part of its
    neighbourhood. 1.0 means the seams are indistinguishable from the rows
    around them; the first round's vignette scored 0.15, which is the faint
    stripe the eye picks up in a dense band. Seams whose neighbourhood holds
    fewer than 8 live cells per row are skipped: nothing to see either way.
    """
    if free is None:
        free = np.ones(pattern.shape, dtype=bool)
    worst = 1.0
    for s0 in seam_rows(strip_rows, pattern.shape[0]):
        sep_free = int(free[s0:s0 + 2].sum())
        nb = np.r_[s0 - 4:s0, s0 + 2:s0 + 6]
        nb_free = int(free[nb].sum())
        nb_live = int(pattern[nb].sum())
        if sep_free < 16 or nb_free < 64 or nb_live < 8 * len(nb):
            continue
        ratio = (int(pattern[s0:s0 + 2].sum()) / sep_free) / (nb_live / nb_free)
        if verbose:
            log(f"   seam rows {s0}-{s0 + 1}: {ratio:.2f} of neighbours "
                f"({int(pattern[s0:s0 + 2].sum())} cells in {sep_free} free)")
        worst = min(worst, ratio)
    return worst


def _targets(design: Design, dither: str):
    from beyond_tiles.targets import cell_targets, window_slices, window_targets

    cell_t = cell_targets(design.grey, D_MAX)
    # "partial": the 540-wide skyline is not a multiple of 8, and the LNS
    # needs disjoint windows (clamped edge windows overlap their neighbour).
    windows = window_slices(cell_t.shape, k=8, stride=8, edge="partial")
    targets, kept = window_targets(cell_t, design.free, windows, dither=dither)
    return cell_t, targets, kept


def _polish_rounds(pattern, free, kept, targets, args, seed0, obj, stop=0.98):
    from beyond_tiles.lns import LnsConfig, improve

    rounds = []
    for rnd in range(1, args.polish_rounds + 1):
        if obj == 0:
            break
        lcfg = LnsConfig(patch_windows=5, patch_time_s=args.patch_time,
                         budget_s=args.polish_budget, n_procs=args.polish_procs,
                         seed=seed0 + rnd, max_diag_run=5)
        t0 = time.perf_counter()
        res = improve(pattern, free, kept, targets, lcfg,
                      log=lambda *a, **k: None)
        wall = time.perf_counter() - t0
        log(f"polish {rnd} done in {wall:.0f}s: objective {obj:,} -> "
            f"{res.objective:,} ({res.patches_improved}/{res.patches_solved} "
            f"patches improved)")
        pattern, prev, obj = res.pattern, obj, int(res.objective)
        rounds.append(dict(objective=obj, wall_time_s=wall,
                           patches_improved=res.patches_improved,
                           patches_solved=res.patches_solved))
        if obj == 0 or obj > stop * prev:
            break
    return pattern, obj, rounds


def solve_strips_safe(grey, free, cfg, plan, n_procs: int, retries: int = 2):
    """`decompose.solve_strips`, one process per strip, with retries.

    The first skyline2 solve (2026-09-10) died in `solve_strips`: one CP-SAT
    worker aborted with a C-runtime fail-fast about two minutes in, and a
    `ProcessPoolExecutor` then discards every strip with it. Here each strip
    gets a pool of its own, so an abort costs one retry with another seed.
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from beyond_tiles.decompose import _check_cfg, _solve_strip_task

    cfg = _check_cfg(cfg)
    payloads = []
    for idx, (r0, r1) in enumerate(plan.spans):
        f = free[r0:r1].copy()
        if idx < len(plan.spans) - 1 and plan.gap:
            f[-plan.gap:] = False
        payloads.append(dict(grey=grey[r0:r1], free=f, cfg=cfg))

    def run(idx):
        payload = payloads[idx]
        for attempt in range(retries + 1):
            try:
                with ProcessPoolExecutor(max_workers=1) as pool:
                    return idx, pool.submit(_solve_strip_task, payload).result()
            except BrokenProcessPool:
                r0, r1 = plan.spans[idx]
                log(f"strip {idx} (rows {r0}-{r1}) aborted on attempt "
                    f"{attempt + 1}; retrying with another seed")
                payload = dict(payload, cfg=replace(cfg, seed=cfg.seed + 100 * (attempt + 1)))
        raise RuntimeError(f"strip {idx} aborted {retries + 1} times")

    with ThreadPoolExecutor(max_workers=n_procs) as tp:
        outs = [o for _, o in sorted(tp.map(run, range(len(payloads))))]
    return {
        "pattern": np.vstack([o["pattern"] for o in outs]).astype(np.uint8),
        "objective": int(sum(o["objective"] for o in outs)),
        "statuses": [o["status"] for o in outs],
        "wall_time_s": max(o["wall_time_s"] for o in outs),
    }


def solve(design: Design, args) -> Path:
    """poster.py's strips + LNS pipeline on a synthetic tone target."""
    from beyond_tiles.decompose import plan_strips, solve_strips
    from beyond_tiles.lns import window_devs
    from beyond_tiles.metrics import deviation_stats, max_diagonal_run
    from beyond_tiles.still_image import SpikeConfig, verify_still_life

    out = HERE / "results/flyer" / design.name
    out.mkdir(parents=True, exist_ok=True)
    grey, free = design.grey, design.free
    log(f"{design.name}: {grey.shape} grid, {int(free.sum()):,} free cells")

    cell_t, targets, kept = _targets(design, args.dither)
    log(f"{len(kept):,} windows, live-cell target {int(targets.sum()):,} "
        f"({args.dither} dither)")

    plan = plan_strips(grey.shape[0], 8, strip_rows=args.strip_rows, gap=2)
    cfg = SpikeConfig(k=8, stride=8, d_max=D_MAX, seed=args.seed,
                      time_limit_s=args.strip_time, workers=args.strip_workers,
                      max_diag_run=5, dither=args.dither, edge_windows="partial")
    if args.resume and (out / "strips_pattern.npy").exists():
        # A polish that failed after the strips (the 540-wide skyline hit
        # the LNS's disjoint-window check) picks up from the saved strips.
        pattern = np.load(out / "strips_pattern.npy")
        strips = dict(statuses=["resumed"] * len(plan.spans), wall_time_s=0.0)
        n_opt = 0
        obj = int(window_devs(pattern, free, kept, targets).sum())
        log(f"resumed from strips_pattern.npy: objective {obj:,}")
    else:
        t0 = time.perf_counter()
        strips = solve_strips_safe(grey, free, cfg, plan, args.strip_procs)
        wall = time.perf_counter() - t0
        n_opt = sum(s == "OPTIMAL" for s in strips["statuses"])
        pattern = strips["pattern"]
        obj = int(window_devs(pattern, free, kept, targets).sum())
        np.save(out / "strips_pattern.npy", pattern)
        log(f"strips done in {wall:.0f}s: {n_opt}/{len(plan.spans)} OPTIMAL, "
            f"objective {obj:,}")

    pattern, obj, rounds = _polish_rounds(pattern, free, kept, targets, args,
                                          args.seed, obj)

    ver = verify_still_life(pattern)
    assert ver["bounded"] and ver["toroidal"], f"verification failed: {ver}"
    assert int(pattern[~free].sum()) == 0, "live cells in the tile region"
    stats = deviation_stats(pattern, cell_t, free, kept)
    seams = seam_report(pattern, args.strip_rows, free, verbose=True)
    np.save(out / "pattern.npy", pattern)
    report = dict(design=design.name, note=design.note,
                  grid=[HEIGHT, WIDTH], d_max=D_MAX, config=vars(args),
                  strips=dict(n_optimal=n_opt, statuses=strips["statuses"],
                              wall_time_s=strips["wall_time_s"]),
                  polish_rounds=rounds, objective=obj, deviation=stats,
                  verify=ver, max_diagonal_run=max_diagonal_run(pattern),
                  seam_occupancy=seams, live_cells=int(pattern.sum()))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    log(f"{design.name} DONE: {int(pattern.sum()):,} live cells, objective "
        f"{obj:,}, MAD {stats['mad']:.4f}, Pearson {stats['pearson']:.4f}, "
        f"seam occupancy {seams:.2f}")
    return out


def seam_rows(strip_rows: int = STRIP_ROWS, height: int = None):
    """First row of every two-row strip separator, top to bottom.

    Pure arithmetic rather than `plan_strips`, because worker processes call
    this per patch; `polish` checks it against the plan once. The canvas
    height reaches the workers through the environment (`FLYER_HEIGHT`),
    which `polish` sets before the pool is spawned.
    """
    import os

    if height is None:
        height = int(os.environ.get("FLYER_HEIGHT", HEIGHT))
    return tuple(r1 - 2 for r1 in range(strip_rows, height, strip_rows))


def _seam_target(target: int, rows: int) -> int:
    """The share of a window's target that two of its rows should carry."""
    return int(round(target * 2 / rows))


# Weight of the seam sub-targets against the window targets, on both sides of
# the LNS (the patch solver's objective and the acceptance rule). Probed on
# the vignette: at weight 3 with the incumbent as hint, a seam patch solves
# to optimality in about ten seconds, leaves the window deviations untouched
# and fills the separator rows to their share. Blanking the hint instead
# filled the seams too, but the from-scratch re-solve left the windows
# hundreds of cells off target within the time limit.
SEAM_WEIGHT = 3


def patch_task_seam(payload: dict):
    """`lns._solve_patch_task` with a sub-window target on each seam.

    Runs in a worker process. For every window box that contains a strip
    separator, an extra two-row box with a proportional target is appended,
    so the patch solver has to move cells *into* the separator rows rather
    than merely meet the window total around them.
    """
    # `ORIGINAL_PATCH_TASK` is captured at import time: `polish` rebinds
    # `lns._solve_patch_task` to this function in the main process, so any
    # lookup through the module at call time would recurse.
    i0 = payload["box"][0]
    extra = []
    for s0 in seam_rows():
        sr = s0 - i0 + LNS_MARGIN
        for a0, a1, b0, b1, target in payload["windows"]:
            if a0 <= sr and sr + 2 <= a1:
                box = (sr, sr + 2, b0, b1, _seam_target(target, a1 - a0))
                extra += [box] * SEAM_WEIGHT
    payload = dict(payload, windows=list(payload["windows"]) + extra)
    return ORIGINAL_PATCH_TASK(payload)


def polish(design: Design, args) -> None:
    """Extra LNS rounds on a saved solve, optionally as a seam pass.

    Why a seam pass is needed at all: the objective counts live cells per 8x8
    window, and a window that contains the two dead separator rows of a
    strip still has six free rows to meet its target in. The strip solver
    therefore satisfies every seam window with the separator rows empty, the
    deviation there is zero, and the generic polish never selects a patch
    with nothing to improve. The seams survive as two empty rows across a
    dense band, which is the faint horizontal stripe the eye picks up.

    `--seams-only` therefore changes the objective, not just the patch
    selection: `window_devs` gains a one-sided deficit term for the separator
    rows of each seam window, the patch solver gets a matching sub-window
    target (`patch_task_seam`), and the candidate patches are the boxes
    centred on a seam. The incumbent then scores badly exactly at the seams,
    each re-solve has to weave cells across them, and the LNS acceptance rule
    (strictly lower deviation) sees the improvement. Everything else about
    the still-life guarantee is unchanged: patches are re-solved against a
    frozen, stable context, as always.

    With `--dither fs` on a solve made with rounded targets, the deviation
    lands on the fringe of the fade and the polish re-solves the dotted line
    there.
    """
    from beyond_tiles import lns
    from beyond_tiles.metrics import deviation_stats, max_diagonal_run
    from beyond_tiles.still_image import verify_still_life

    from beyond_tiles.decompose import plan_strips

    import os

    if args.strip_rows != STRIP_ROWS:
        raise SystemExit(f"the seam pass assumes --strip-rows {STRIP_ROWS}")
    height = design.grey.shape[0]
    os.environ["FLYER_HEIGHT"] = str(height)  # for the patch workers
    plan = plan_strips(height, 8, strip_rows=STRIP_ROWS, gap=2)
    assert seam_rows(STRIP_ROWS, height) == \
        tuple(r1 - 2 for _, r1 in plan.spans[:-1]), plan.spans
    out = HERE / "results/flyer" / design.name
    pattern = np.load(out / "pattern.npy")
    np.save(out / "pattern_prepolish.npy", pattern)
    free = design.free
    cell_t, targets, kept = _targets(design, args.dither)
    plain_devs = lns.window_devs
    obj = int(plain_devs(pattern, free, kept, targets).sum())
    seams_before = seam_report(pattern, args.strip_rows, free)

    original = (lns._patch_boxes, lns.window_devs, lns._solve_patch_task)
    if args.seams_only:
        rows = seam_rows(args.strip_rows, height)
        # The LNS window grid spans only the windows that hold free cells, so
        # its row index is offset from the canvas: map each seam through the
        # grid's row starts rather than dividing by 8.
        grid_rows = lns._window_grid(kept)[0]
        seam_grid_rows = sorted({grid_rows.index((r // 8) * 8) for r in rows
                                 if (r // 8) * 8 in grid_rows})

        def seam_boxes(n_rows, n_cols, patch):
            boxes = original[0](n_rows, n_cols, patch)
            # Every box that holds the seam with at least one window row of
            # variables on either side of it; a seam in the grid's first row
            # (the skyline's, right under the ribbon) has only the frozen
            # tiles above it, and a box starting on it will do.
            return [b for b in boxes for g in seam_grid_rows
                    if (b[0] + 1 <= g or (g == 0 and b[0] == 0)) and g <= b[1] - 2]

        def window_devs_seam(pattern, free_mask, windows, targets, slack=0):
            devs = np.array(plain_devs(pattern, free_mask, windows, targets,
                                       slack), copy=True)
            for m, (si, sj) in enumerate(windows):
                for s0 in rows:
                    if si.start <= s0 and s0 + 2 <= si.stop:
                        want = _seam_target(int(targets[m]), si.stop - si.start)
                        have = int(pattern[s0:s0 + 2, sj].sum())
                        devs[m] += SEAM_WEIGHT * max(0, want - have)
            return devs

        lns._patch_boxes = seam_boxes
        lns.window_devs = window_devs_seam
        lns._solve_patch_task = patch_task_seam
        obj_seam = int(window_devs_seam(pattern, free, kept, targets).sum())
        log(f"{design.name}: seam pass from objective {obj:,} "
            f"(+{obj_seam - obj:,} seam deficit), seam occupancy "
            f"{seams_before:.2f} ({args.dither} dither)")
        start = obj_seam
    else:
        log(f"{design.name}: polish from objective {obj:,}, seam occupancy "
            f"{seams_before:.2f} ({args.dither} dither)")
        start = obj
    try:
        pattern, _, rounds = _polish_rounds(pattern, free, kept, targets, args,
                                            args.seed + 100, start, stop=0.995)
    finally:
        lns._patch_boxes, lns.window_devs, lns._solve_patch_task = original

    obj = int(plain_devs(pattern, free, kept, targets).sum())
    ver = verify_still_life(pattern)
    assert ver["bounded"] and ver["toroidal"], f"verification failed: {ver}"
    assert int(pattern[~free].sum()) == 0, "live cells in the tile region"
    seams = seam_report(pattern, args.strip_rows, free, verbose=True)
    np.save(out / "pattern.npy", pattern)
    report = json.loads((out / "report.json").read_text())
    report.setdefault("extra_polish", []).append(dict(
        seams_only=args.seams_only, dither=args.dither, objective=obj,
        rounds=rounds, deviation=deviation_stats(pattern, cell_t, free, kept),
        max_diagonal_run=max_diagonal_run(pattern), seam_occupancy=seams))
    report["objective"] = obj
    report["seam_occupancy"] = seams
    (out / "report.json").write_text(json.dumps(report, indent=2))
    log(f"{design.name} polished: objective {obj:,}, seam occupancy "
        f"{seams_before:.2f} -> {seams:.2f}, {int(pattern.sum()):,} live cells")


# --- Palettes -----------------------------------------------------------------
#
# Slots: gol_background = the text zone, gol_pixel = the grain, eca_background
# = the frame field, eca_pixel = the tiles, fill_pixel = the filler haze.
# Two rules hold for every palette. The text zone keeps strong value contrast
# with its grain (the grain carries tone as density, and low contrast erases
# it), and the frame's field colour sits at least as dark or as light as the
# zone by a wide margin, so the flyer reads as a window in a frame at
# thumbnail size on a phone. Saturation stays moderate: dense pixel grain in
# a saturated hue vibrates.
PALETTES = {
    # The teal-and-cream of the halo figure, unchanged: the reference.
    "teal": FLYER,
    # UGent and USP both use blue and yellow. Complementary pair, both pulled
    # down in saturation: a deep ultramarine field with old-gold tiles, and a
    # navy grain on warm white. The one that reads as institutional.
    "blue-gold": ColorScheme(gol_background="#F6F2E8", gol_pixel="#1C2B4A",
                             eca_background="#17407E", eca_pixel="#E6B95A",
                             fill_pixel="#7F93B8"),
    # Analogous warm: terracotta, sand, umber. Earth colours, calm, with the
    # frame and the grain in the same family so nothing competes.
    "terracotta": ColorScheme(gol_background="#F8F1E6", gol_pixel="#4A3628",
                              eca_background="#9C4F35", eca_pixel="#F1DFC2",
                              fill_pixel="#D6A283"),
    # Split complementary of copper: indigo field, copper grain. The tiles in
    # parchment so the frame stays quiet and the grain carries the warmth.
    "indigo-copper": ColorScheme(gol_background="#F1EEE8", gol_pixel="#8C4A2F",
                                 eca_background="#2E3D5C", eca_pixel="#EADCC0",
                                 fill_pixel="#8E96B0"),
    # Muted complementary: plum against sage. The most unusual of the set.
    "plum-sage": ColorScheme(gol_background="#F4F0E8", gol_pixel="#3A2A3A",
                             eca_background="#4B2E4C", eca_pixel="#D6D2B4",
                             fill_pixel="#A08CA4"),
    # Monochrome navy at three values. One hue, so the structure does all the
    # work; the most formal option.
    "navy-mono": ColorScheme(gol_background="#F3F5F8", gol_pixel="#1B3A5C",
                             eca_background="#1B3A5C", eca_pixel="#B7CADF",
                             fill_pixel="#6F93B5"),
    # The reference palette with its values inverted: a dark slate text zone
    # with cream grain (light text in Canva) and a cream frame with teal
    # tiles. For a phone in dark mode.
    "teal-dark": ColorScheme(gol_background="#1F2A2E", gol_pixel="#E8DFCB",
                             eca_background="#EFE6D4", eca_pixel="#1B5E5E",
                             fill_pixel="#7FA8A0"),
    # Forest and gold: near-complementary, the gold kept dusty.
    "forest-gold": ColorScheme(gol_background="#F7F2E4", gol_pixel="#2B3A2E",
                               eca_background="#1F3D2B", eca_pixel="#D9B25C",
                               fill_pixel="#8AA48C"),
    # --- The quiet set (fourth round). The user found the first four heavy
    # on the eye. Three things carry that weight: tiles at high contrast
    # against their field (cream on teal is a 60-point lightness jump, so
    # every diamond shouts), a near-black grain, and fully saturated
    # fields. Here the tiles sit one step above their field (about 15
    # points), the filler halfway between, the grain a mid-tone of the
    # same hue family, and the fields are tints. The silhouette then reads
    # as a soft textured shape and the text zone as an unbroken sheet.
    # Monochrome cool grey-blue: the most restful, near-neutral.
    "mist": ColorScheme(gol_background="#F4F3EF", gol_pixel="#6E7D8A",
                        eca_background="#8497A8", eca_pixel="#B4C2CE",
                        fill_pixel="#9EAEBC"),
    # Warm neutral, sand and taupe: paper-like, no hue to tire of.
    "sand": ColorScheme(gol_background="#FAF7F1", gol_pixel="#8A7A64",
                        eca_background="#C6B394", eca_pixel="#E3D7C0",
                        fill_pixel="#D5C5A8"),
    # Sage: green-grey, the calmest of the chromatic hues.
    "sage": ColorScheme(gol_background="#F5F7F1", gol_pixel="#6C7D6A",
                        eca_background="#8FA48F", eca_pixel="#B7C8B4",
                        fill_pixel="#A4B7A1"),
    # Heather: greyed mauve, soft and a little unusual.
    "heather": ColorScheme(gol_background="#F7F5F8", gol_pixel="#75697F",
                           eca_background="#9A8FA8", eca_pixel="#C3BACD",
                           fill_pixel="#AFA5BA"),
    # Slate: a dark field kept, but the tiles only a step lighter, so the
    # skyline is one calm dark mass with a faint weave rather than a grid
    # of bright diamonds.
    "slate": ColorScheme(gol_background="#F2F1EC", gol_pixel="#5C6A73",
                         eca_background="#435663", eca_pixel="#60737F",
                         fill_pixel="#526572"),
    # The reference teal with its contrast halved: same identity, the
    # tiles a mid-teal instead of cream, the grain a grey-teal.
    "teal-soft": ColorScheme(gol_background="#F4EDE2", gol_pixel="#587274",
                             eca_background="#2A6666", eca_pixel="#4F8886",
                             fill_pixel="#3E7A78"),
    # Ochre: dusty gold and tan, warm without the terracotta's weight.
    "ochre": ColorScheme(gol_background="#FBF7EE", gol_pixel="#8C7448",
                         eca_background="#C9A86E", eca_pixel="#E4CF9E",
                         fill_pixel="#D7BC85"),
    # Denim: a mid blue, tiles a tint of it; institutional blue, softened.
    "denim": ColorScheme(gol_background="#F3F4F2", gol_pixel="#5E6C7B",
                         eca_background="#5F7A95", eca_pixel="#8AA1B8",
                         fill_pixel="#7590A8"),
}

# The quiet set, for the palette sheet, with the reference first.
QUIET = ["teal", "teal-soft", "slate", "mist", "denim", "sage", "heather",
         "sand", "ochre"]


# --- Symmetric tile placement ---------------------------------------------------

def symmetric_pad(level: int, shape: str = "diamond", fill="auto",
                  height: int = HEIGHT, width: int = WIDTH):
    """Top/left padding that centres every cascade lattice on the canvas.

    The diamond lattice at level L has sites at multiples of 3L on both axes
    (a checkerboard of them), so it is mirror-symmetric about every line
    through a row or column of site centres, i.e. rows 3L*k + (3L - 0.5).
    A plain call to `mosaic_background` starts the lattice at the canvas
    origin, and whether a symmetric window sees the same lattice at its top
    and bottom edges is then luck: on the 360x512 vignette the level-4 row
    touched the inner edge at the top and left a gap at the bottom.

    Padding the mask by (py, px) before placing the tiles shifts the lattice
    the other way. This searches the smallest padding that puts the canvas
    centre on a mirror line of the main level and of every cascade level at
    once, so the whole placement is symmetric top/bottom and left/right. Tile
    *choices* stay random; it is the placement the eye compares.
    """
    from gol_mosaics.compose import _resolve_fill_levels

    levels = [level] + list(_resolve_fill_levels(level, shape, fill) or ())
    if shape != "diamond":
        raise ValueError("symmetric placement is worked out for diamonds only")

    def aligned(centre, pad):
        return all((centre + pad - (3 * L - 0.5)) % (3 * L) == 0 for L in levels)

    py = next(p for p in range(0, 4 * 36) if aligned((height - 1) / 2, p))
    px = next(p for p in range(0, 4 * 36) if aligned((width - 1) / 2, p))
    return py, px


def symmetric_field(background: np.ndarray, level: int, shape: str = "diamond",
                    fill="auto", **kwargs) -> np.ndarray:
    """`filled_background` with the lattice centred on the canvas."""
    from gol_mosaics import filled_background

    py, px = symmetric_pad(level, shape, fill, *background.shape)
    padded = np.pad(background, ((py, 0), (px, 0)))  # padding is subject: no tiles
    field = filled_background(padded, level=level, shape=shape, fill=fill,
                              **kwargs)
    return np.ascontiguousarray(field[py:, px:])


def render_field(pattern, background, field, scheme, level, shape="diamond",
                 fill="auto", scale=SCALE) -> Image.Image:
    """`compose(style="mosaic")` with a field supplied instead of generated."""
    from gol_mosaics import MosaicRenderer
    from gol_mosaics.compose import fill_layer_count

    backdrop = background.astype(np.uint8)
    image = MosaicRenderer(scheme).render_full_mosaic(
        pattern, backdrop * (field + backdrop),
        layers=fill_layer_count(level, shape, fill))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale),
                             Image.Resampling.NEAREST)
    return image


# --- Render -------------------------------------------------------------------

HAZE_CREAM = replace(FLYER, fill_pixel="#D8CDB6")
HAZE_SAGE = replace(FLYER, fill_pixel="#A9BFB6")

# Level 4 throughout (the chosen tile size). Third round, after the review of
# the second: the vignette re-solved with the narrower frame and a centred
# lattice, in the reference palette and the three palettes the user picked
# (terracotta, indigo-copper, forest-gold; light text zones only), and the
# two skylines recoloured in the same three.
RENDERS = [
    # Fourth round: the skyline redrawn from the user's photos (skyline2,
    # 540x768 at 2 px), in the reference palette and the three chosen ones.
    ("skyline2-teal", "skyline2",
     dict(level=4, seed=4, scheme="teal", symmetric=True)),
    ("skyline2-terracotta", "skyline2",
     dict(level=4, seed=4, scheme="terracotta", symmetric=True)),
    ("skyline2-indigo-copper", "skyline2",
     dict(level=4, seed=4, scheme="indigo-copper", symmetric=True)),
    ("skyline2-forest-gold", "skyline2",
     dict(level=4, seed=4, scheme="forest-gold", symmetric=True)),
    ("vignette2-teal", "vignette2",
     dict(level=4, seed=1, scheme="teal", symmetric=True)),
    ("vignette2-terracotta", "vignette2",
     dict(level=4, seed=1, scheme="terracotta", symmetric=True)),
    ("vignette2-indigo-copper", "vignette2",
     dict(level=4, seed=1, scheme="indigo-copper", symmetric=True)),
    ("vignette2-forest-gold", "vignette2",
     dict(level=4, seed=1, scheme="forest-gold", symmetric=True)),
    ("skyline-terracotta", "skyline", dict(level=4, seed=4, scheme="terracotta")),
    ("skyline-indigo-copper", "skyline",
     dict(level=4, seed=4, scheme="indigo-copper")),
    ("skyline-forest-gold", "skyline", dict(level=4, seed=4, scheme="forest-gold")),
    ("antipodes-terracotta", "antipodes",
     dict(level=4, seed=6, scheme="terracotta")),
    ("antipodes-indigo-copper", "antipodes",
     dict(level=4, seed=6, scheme="indigo-copper")),
    ("antipodes-forest-gold", "antipodes",
     dict(level=4, seed=6, scheme="forest-gold")),
]

# Earlier rounds, kept so they can be re-rendered by label.
ROUND2 = [
    ("vignette-haze", "vignette", dict(level=4, seed=1)),
    ("vignette-cream", "vignette", dict(level=4, seed=1, scheme=HAZE_CREAM)),
    ("vignette-sage-dense", "vignette",
     dict(level=4, density=(0.6, 1.0), seed=2, scheme=HAZE_SAGE)),
    ("skyline-haze", "skyline", dict(level=4, seed=4)),
    ("skyline-cream", "skyline", dict(level=4, seed=4, scheme=HAZE_CREAM)),
    ("skyline-sage-dense", "skyline",
     dict(level=4, density=(0.6, 1.0), seed=5, scheme=HAZE_SAGE)),
    ("antipodes-haze", "antipodes", dict(level=4, seed=6)),
    ("antipodes-cream", "antipodes", dict(level=4, seed=6, scheme=HAZE_CREAM)),
    ("arch-haze", "arch", dict(level=4, seed=9)),
    ("arch-cream", "arch", dict(level=4, seed=9, scheme=HAZE_CREAM)),
]


def load_solve(name: str):
    path = HERE / "results/flyer" / name / "pattern.npy"
    if not path.exists():
        raise SystemExit(f"no solve for {name!r}: run `flyer.py solve {name}`")
    return np.load(path)


def render(labels=None) -> None:
    from beyond_tiles.linkedin_banners import contact_sheet
    from beyond_tiles.still_image import verify_still_life
    from gol_mosaics import compose, filled_background, life_safe_pattern
    from gol_mosaics.export import GollyExporter

    out = REPO / "output/images/flyer"
    out.mkdir(parents=True, exist_ok=True)
    golly = REPO / "output/golly"
    golly.mkdir(parents=True, exist_ok=True)
    designs, patterns, panels = {}, {}, []

    for label, name, kwargs in RENDERS + ROUND2:
        if labels and label not in labels:
            continue
        if not labels and (label, name, kwargs) not in RENDERS:
            continue
        if name not in designs:
            designs[name] = DESIGNS[name]()
            patterns[name] = load_solve(name)
        d, pattern = designs[name], patterns[name]
        assert pattern.shape == d.free.shape
        kwargs = dict(kwargs)
        scheme = kwargs.pop("scheme", FLYER)
        if isinstance(scheme, str):
            scheme = PALETTES[scheme]
        symmetric = kwargs.pop("symmetric", False)
        background = ~d.free

        if symmetric:
            field = symmetric_field(background, fill="auto", **kwargs)
            image = render_field(pattern, background, field, scheme,
                                 kwargs["level"], kwargs.get("shape", "diamond"),
                                 scale=d.scale)
        else:
            field = filled_background(background, fill="auto", **kwargs)
            image = compose(pattern, background, scheme, style="mosaic",
                            fill="auto", scale=d.scale, **kwargs)
        image.convert("RGB").save(out / f"{label}.png")

        # The same field the render used, merged and checked as one still life.
        whole = life_safe_pattern(pattern, background, field=field)
        ver = verify_still_life(whole)
        assert ver["bounded"] and ver["toroidal"], f"{label}: {ver}"
        GollyExporter.export_to_cells(whole, str(golly / f"flyer-{label}.cells"))
        print(f"{label:22s} {image.size[0]}x{image.size[1]} px, "
              f"subject {int(pattern.sum()):,} + field {int((field != 0).sum()):,} "
              f"= {int(whole.sum()):,} cells, still life {ver}")
        panels.append((label, image))

    if len(panels) > 1:
        half = (len(panels) + 1) // 2
        contact_sheet([panels[:half], panels[half:]], out / "_contact_sheet.png",
                      width=9, panel_h=5.6)
        print(f"contact sheet -> {out / '_contact_sheet.png'}")


def palette_sheet(name: str, level: int = 4, seed: int = 1,
                  symmetric: bool = True) -> Path:
    """One design, every palette; the geometry is identical across panels."""
    from beyond_tiles.linkedin_banners import contact_sheet
    from gol_mosaics import compose, filled_background

    out = REPO / "output/images/flyer"
    out.mkdir(parents=True, exist_ok=True)
    d, pattern = DESIGNS[name](), load_solve(name)
    background = ~d.free
    field = (symmetric_field(background, level=level, seed=seed) if symmetric
             else filled_background(background, level=level, seed=seed))
    panels = []
    for pname, scheme in PALETTES.items():
        image = render_field(pattern, background, field, scheme, level, scale=1)
        image.convert("RGB").save(out / f"_palette_{name}_{pname}.png")
        panels.append((pname, image))
    half = (len(panels) + 1) // 2
    path = out / f"_palettes_{name}.png"
    contact_sheet([panels[:half], panels[half:]], path, width=9, panel_h=5.6)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command",
                    choices=["preview", "solve", "polish", "render", "palettes"])
    ap.add_argument("names", nargs="*", help="designs (solve) or labels (render)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="solve: reuse a saved strips_pattern.npy")
    ap.add_argument("--strip-rows", type=int, default=STRIP_ROWS)
    ap.add_argument("--strip-time", type=float, default=240.0)
    ap.add_argument("--strip-procs", type=int, default=6)
    ap.add_argument("--strip-workers", type=int, default=2)
    ap.add_argument("--polish-rounds", type=int, default=3)
    ap.add_argument("--polish-budget", type=float, default=240.0)
    ap.add_argument("--patch-time", type=float, default=10.0)
    ap.add_argument("--polish-procs", type=int, default=6)
    ap.add_argument("--seams-only", action="store_true",
                    help="polish: only patches straddling the strip seams")
    ap.add_argument("--dither", choices=["round", "fs"], default="fs",
                    help="window-target quantisation (solve and polish)")
    args = ap.parse_args()

    if args.command == "preview":
        path = preview(args.names or ACTIVE)
        print(f"preview -> {path}")
    elif args.command == "solve":
        for name in args.names or ACTIVE:
            solve(DESIGNS[name](), args)
    elif args.command == "polish":
        for name in args.names or ACTIVE:
            polish(DESIGNS[name](), args)
    elif args.command == "palettes":
        for name in args.names or ["vignette"]:
            print(f"palette sheet -> {palette_sheet(name)}")
    else:
        render(set(args.names))


if __name__ == "__main__":
    main()
