"""Thesis cover (160 x 240 mm): gapless mosaics of tiles of levels 1-7.

The cover of a joint UGent / USP PhD thesis: the front is the drowned coast
of Paraty Mirim (Brazil) traced from satellite silhouettes, the back the
defence flyer's Ghent and Sao Paulo skyline over a band for the
bibliographic block. Only the background is made here; the text, its
see-through boxes and the logo strip are laid over it in Canva. The
design history is in README.md.

**The pond lattice.** Every diamond's frame is a ring of ponds, and all of
those ponds sit on one lattice: the pond in the 6 x 6 block at cell
(3a, 3b), for every a + b even. A level-L tile covers an L x L square of
pond vertices (in the lattice's own, 45-degree turned axes), and the
level-1 tile is a single pond. Tiles of any levels on vertex-disjoint
squares form a still life: where tiles meet, every cell sees only frame
ponds and forced-dead interlock cells, i.e. a subset of the all-pond
mosaic, which is a still life (the level-1 census). So tiles of different
levels may share a seam exactly as tiles of one lattice do, and a region
can be packed without a single gap. Every cover is still checked cell by
cell.

**Packing.** A design is a *pattern mask* (where tiles may go) and a
continuous *level field* Z (which level belongs where). Levels 7 down to 2
are packed greedily, one tile at a time:

- a level-L tile may sit where Z at its centre is at least L - 1/2, and its
  support must stay inside the mask where Z >= L - 1/2 - REACH (so it may
  lean over into the next level down, not further);
- of all the places it fits, it takes the one where its rim touches the
  most tiles already placed (or the pattern's edge), so tiles stack into
  corners instead of leaving holes; ties are broken towards higher Z, then
  at random.

Then every pond vertex still free inside the mask gets a pond (level 1).

**Pages.** The study pages (rounds 2-7) are a torus of 360 x 540 cells at
4/9 mm per cell, so every tile may cross any edge. The delivered covers
need not wrap: they are packed on the page padded by MARGIN cells (the
mask mirrored into it) and cut at the trim, so tiles run off the page and
none wraps back onto it. The back cover keeps the flyer's cells, 540 x 810
at 0.30 mm.

**Label maps** (one uint8 per cell): 0 field, L a live cell of a level-L
tile, 20 + L a dead cell inside it; a tile of level s = 1, 2 filling a hole
among bigger tiles, the biggest of level K, is 40 + 10 (K - 3) + s live and
140 + 10 (K - 3) + s dead, and the grout inside such a hole 200 + K, so the
hole can be closed in its host's colour (`layers` does that).

**Colour.** `palettes` designs the level ramps in OKLCH and moves them into
the Coated FOGRA39 gamut; its output for the delivered covers is versioned
as assets/palettes.json. The covers use Denim on apricot, with the COAST
steps of the ramp (the two palest dropped, so small tiles stand off the
field).

The delivered covers:

    python studies/cover/cover.py map --out studies/cover/renders/front --pairs denim-apricot
    python studies/cover/cover.py lifted levels-deep --out studies/cover/renders/back

The study rounds, into output/ (ignored by git):

    python studies/cover/cover.py build [name ...] --out studies/cover/output/data
    python studies/cover/cover.py print [name ...] --out studies/cover/output/round7
    python studies/cover/cover.py back --out studies/cover/output/back-round7
    python studies/cover/cover.py lifted --out studies/cover/output/lifted   # all six back variants
    python studies/cover/cover.py palettes --out DIR --profile Coated_Fogra39L_VIGC_300.icc

`map` writes its label maps to --data (default output/data), where `print`
reads them. The level-7 tiles come from the packed census at the
repository root (`solutions_pattern_level_7_orbits.npy`, 1.2 GB, not in
git, see search/tiles); only a random sample of it is read. The back cover
reads the flyer's solve from studies/flyer/solves.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ASSETS = HERE / "assets"
for p in (REPO / "src", HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from common import log
from gol_mosaics.tile_domain import build_domain
from gol_mosaics.tile_library import TileLibrary
from gol_mosaics.tile_scheme import diamond_scheme

W, H = 360, 540                 # cells; 160 x 240 mm at 4/9 mm per cell
A, B = H // 3, W // 3           # the pond lattice: vertex (a, b), a + b even
LEVEL7 = REPO / "solutions_pattern_level_7_orbits.npy"
LOGO_TOP = 478                  # the logo row sits below this row
REACH = 0.6                     # how far (in levels) a tile may lean into a lower zone


# --- Text layout (the user's draft: 1333 x 2000 px = the trim) -------------------

DRAFT_W = 1333
SPRITES = {  # name: (file, width, height) in draft px
    "the": ("box_0258_0101", 206, 124), "rich": ("box_0398_0101", 260, 123),
    "landscape": ("box_0537_0101", 505, 124), "of": ("box_0676_0101", 158, 124),
    "iterated": ("box_0816_0101", 464, 123), "simplicity": ("box_0955_0101", 558, 124),
    "sub1": ("box_1150_0101", 785, 84), "sub2": ("box_1248_0101", 610, 83),
    "name": ("box_1639_0721", 583, 81),
}
X0, GAP, STEP = 101, 30, 140
RIGHT = 1254                    # right edge of the logo row
THREE = [["the", "rich", "landscape"], ["of", "iterated"], ["simplicity"]]


def _lines(rows, y0, x0=X0):
    out = []
    for k, row in enumerate(rows):
        x = x0
        for w in row:
            out.append((w, x, y0 + k * STEP))
            x += SPRITES[w][1] + GAP
    return out


def _byline(left, y0=1442):
    """Subtitle and name stacked above the logos, flush left or right."""
    return [(w, X0 if left else RIGHT - SPRITES[w][1], y0 + 98 * k)
            for k, w in enumerate(["sub1", "sub2", "name"])]


LAYOUTS = {
    "top": _lines(THREE, 120) + [("sub1", X0, 580), ("sub2", X0, 678), ("name", 721, 1639)],
    "split-left": _lines(THREE, 120) + _byline(True),
    "split-right": _lines(THREE, 120) + _byline(False),
}
LAYOUT_LABELS = {"top": "Subtitle under the title", "split-left": "By the name, left",
                 "split-right": "By the name, right"}
LOGOS = (81, 1808, 1173, 140)   # x, y, w, h of the logo row in draft px


def text_mask(layout, margin=6):
    """Cells under the layout's boxes and the logo row, grown by `margin`."""
    f = W / DRAFT_W
    m = np.zeros((H, W), bool)
    boxes = [(x, y, SPRITES[w][1], SPRITES[w][2]) for w, x, y in LAYOUTS[layout]]
    for x, y, w, h in boxes + [LOGOS]:
        y0, x0 = int(y * f) - margin, int(x * f) - margin
        m[max(0, y0):int((y + h) * f) + margin, max(0, x0):int((x + w) * f) + margin] = True
    return m


# --- Tiles ---------------------------------------------------------------------

_POOLS = {}


def tile_pool(level, rng, size=48, window=(0.3, 0.8)):
    """A few dozen tiles of one level from the middle of its density range
    (all of them for the small levels, which have only a handful)."""
    if level in _POOLS:
        return _POOLS[level]
    if level <= 6:
        tiles = TileLibrary.load(level).tiles
    else:
        packed = np.load(LEVEL7, mmap_mode="r")
        idx = np.sort(rng.choice(len(packed), 20000, replace=False))
        tiles = build_domain(7).unpack(np.asarray(packed[idx]))
    if len(tiles) > 12:
        dens = tiles.reshape(len(tiles), -1).mean(1)
        lo, hi = np.quantile(dens, window)
        ok = np.flatnonzero((dens >= lo) & (dens <= hi))
        tiles = tiles[rng.choice(ok, size=min(size, len(ok)), replace=False)]
    _POOLS[level] = np.asarray(tiles, dtype=np.uint8)
    return _POOLS[level]


_SHAPES = {}


def shape(level):
    """Offsets (from the block corner) of the tile's support."""
    if level not in _SHAPES:
        sch = diamond_scheme(level)
        _SHAPES[level] = np.nonzero(sch.support | sch.frame)
    return _SHAPES[level]


_KERNELS = {}


def touches(mask, level):
    """True at p where the level's support, with its block corner at p,
    meets `mask` (a periodic correlation)."""
    if level not in _KERNELS:
        sy, sx = shape(level)
        k = np.zeros((H, W))
        k[sy % H, sx % W] = 1
        _KERNELS[level] = np.conj(np.fft.rfft2(k))
    return np.fft.irfft2(np.fft.rfft2(mask.astype(float)) * _KERNELS[level], s=(H, W)) > 0.5


AA, BB = np.mgrid[0:A, 0:B]
PONDS = (AA + BB) % 2 == 0


def block(L):
    """Pond vertices of a level-L tile, relative to its top vertex."""
    m, n = np.mgrid[0:L, 0:L]
    return (m + n).ravel(), (m - n).ravel()


def rim(L):
    """Pond vertices next to a level-L tile (its four sides and corners)."""
    inside = set(zip(*block(L)))
    out = {(a + da, b + db) for a, b in inside for da in (-1, 1) for db in (-1, 1)} - inside
    out |= {(a + da, b + db) for a, b in inside for da, db in ((2, 0), (-2, 0), (0, 2), (0, -2))} - inside
    return tuple(np.array(v) for v in zip(*sorted(out)))


_VKERNELS = {}


def vcorr(mask, L, kind):
    """Periodic correlation on the pond lattice: at each top vertex, how
    many vertices of `mask` lie under the tile's block (or its rim)."""
    key = (L, kind)
    if key not in _VKERNELS:
        da, db = block(L) if kind == "block" else rim(L)
        k = np.zeros((A, B))
        np.add.at(k, (da % A, db % B), 1)
        _VKERNELS[key] = np.conj(np.fft.rfft2(k))
    return np.fft.irfft2(np.fft.rfft2(mask.astype(float)) * _VKERNELS[key], s=(A, B))


def corner(L):
    """Block corner and centre cell of a level-L tile at every top vertex."""
    return ((3 * AA) % H, (3 * BB - 3 * L + 3) % W), ((3 * AA + 3 * L) % H, (3 * BB + 3) % W)


def _wrap_labels(lab, n):
    """Join the components of `ndi.label` (8-connected) that meet across
    the edges of the torus."""
    parent = np.arange(n + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for u, v in ((lab[-1], lab[0]), (lab[:, -1], lab[:, 0])):
        for d in (-1, 0, 1):
            w = np.roll(v, d)
            for p, q in zip(u[(u > 0) & (w > 0)], w[(u > 0) & (w > 0)]):
                parent[find(p)] = find(q)
    roots = np.array([find(i) for i in range(n + 1)])
    return roots[lab]


def periodic_still_life(g):
    g = g.astype(np.int16)
    n = sum(np.roll(np.roll(g, dy, 0), dx, 1)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    nxt = ((g == 1) & ((n == 2) | (n == 3))) | ((g == 0) & (n == 3))
    return bool(np.array_equal(nxt, g == 1))


# --- A cover -------------------------------------------------------------------

class Cover:
    """Whole tiles on vertex-disjoint squares of the pond lattice."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.live = np.zeros((H, W), np.uint8)
        self.ground = np.zeros((H, W), np.uint8)
        self.vlevel = np.zeros((A, B), np.uint8)
        self.mask = np.zeros((H, W), bool)
        self.placed = Counter()
        self.tiles = []

    def put(self, L, a, b):
        """Seat a random level-L tile (turned or mirrored at random: the
        support and frame are symmetric) with its top pond at vertex (a, b)."""
        sy, sx = shape(L)
        pool = tile_pool(L, self.rng)
        t = np.rot90(pool[self.rng.integers(len(pool))], self.rng.integers(4))
        if self.rng.random() < .5:
            t = t.T
        rows, cols = (3 * a + sy) % H, (3 * b - 3 * L + 3 + sx) % W
        self.live[rows, cols] = t[sy, sx]
        self.ground[rows, cols] = L
        da, db = block(L)
        self.vlevel[(a + da) % A, (b + db) % B] = L
        self.placed[L] += 1
        self.tiles.append((L, a, b))

    def sites(self, L, allowed, centre_ok):
        """Top vertices where a level-L tile's support lies in `allowed`
        and its centre in `centre_ok`."""
        (ky, kx), (cy, cx) = corner(L)
        return PONDS & ~touches(~allowed, L)[ky, kx] & centre_ok[cy, cx]

    def pack(self, L, sites, zpref, wall):
        """Seat level-L tiles one at a time until none fits, each where its
        rim touches the most tiles or `wall` (then highest `zpref`)."""
        nr = len(rim(L)[0])
        while True:
            occ = self.vlevel > 0
            cand = sites & (vcorr(occ, L, "block") < 0.5)
            if not cand.any():
                break
            score = np.round(vcorr(occ | wall, L, "rim") / nr, 6) + 1e-3 * zpref
            score = np.where(cand, score, -np.inf)
            ties = np.argwhere(score >= score.max() - 1e-9)
            self.put(L, *ties[self.rng.integers(len(ties))])

    def fill(self, mask, Z, levels=(7, 6, 5, 4, 3, 2), reach=REACH, ponds=True):
        """Pack levels big to small into `mask` following the level field Z,
        then put a pond on every vertex left free."""
        self.mask |= mask
        vin = self.sites(1, mask, mask)
        wall = ~vin
        for L in levels:
            s = self.sites(L, mask & (Z >= L - 0.5 - reach), mask & (Z >= L - 0.5))
            _, (cy, cx) = corner(L)
            self.pack(L, s, np.clip(Z[cy, cx] - (L - 0.5), 0, 2), wall)
        if ponds:
            for a, b in np.argwhere(vin & (self.vlevel == 0)):
                self.put(1, a, b)

    # -- output --
    def labels(self):
        """The label map. A clump of small tiles (levels 1 and 2) that does
        not reach the pattern's edge fills a hole among bigger tiles; its
        tiles are written with the biggest level K around the clump, so
        they can take either colour."""
        g, v = self.ground, self.vlevel
        lab = np.where(g > 0, 20 + g, 0).astype(np.uint8)
        lab[self.live > 0] = g[self.live > 0]
        diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        near = lambda m: np.any([np.roll(m, d, (0, 1)) for d in diag], axis=0)
        small = (v == 1) | (v == 2)
        clumps, n = ndi.label(small, np.ones((3, 3), bool))
        clumps = _wrap_labels(clumps, n)
        open_ = np.unique(clumps[small & near(PONDS & (v == 0))])
        host = {}
        for k in range(1, n + 1):
            if k in open_:
                continue
            ring = near(clumps == k) & (v >= 3)
            if ring.any():
                host[k] = int(v[ring].max())
        owner = np.zeros((H, W), np.int32)        # the hole clump under each cell
        for L, a, b in self.tiles:
            k = clumps[a, b]
            if L > 2 or k not in host:
                continue
            sy, sx = shape(L)
            rows, cols = (3 * a + sy) % H, (3 * b - 3 * L + 3 + sx) % W
            alive = self.live[rows, cols] > 0
            lab[rows, cols] = np.where(alive, 40, 140) + 10 * (host[k] - 3) + L
            owner[rows, cols] = k
        # The grout between the small tiles of one hole, written 200 + K, so
        # a renderer can close it in the host's ground and the hole reads as
        # one more patch of the host level. Grout that also touches a
        # bigger tile stays field, like every other seam.
        nbrs = [np.roll(np.roll(g > 0, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        owners = [np.roll(np.roll(owner, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        k = np.max(owners, axis=0)
        inner = (g == 0) & (k > 0)
        for n, o in zip(nbrs, owners):
            inner &= ~n | (o == k)
        hosts = np.zeros(max(host, default=0) + 1, np.uint8)
        for c, K in host.items():
            hosts[c] = K
        lab[inner] = 200 + hosts[k[inner]]
        return lab

    def verify(self):
        return periodic_still_life(self.live)


# --- Field helpers -------------------------------------------------------------------

YY, XX = np.mgrid[0:H, 0:W].astype(float)
X = XX[0]


def noise(rng, sigma):
    """Smooth periodic noise with unit standard deviation."""
    z = ndi.gaussian_filter(rng.normal(size=(H, W)), sigma, mode="wrap")
    return z / z.std()


def wave(rng, a1=16, a2=7):
    """A slow wave across the page width that repeats exactly across the seam."""
    p = rng.random(2) * 2 * np.pi
    return a1 * np.sin(2 * np.pi * X / W + p[0]) + a2 * np.sin(4 * np.pi * X / W + p[1])


def strata_field(bounds, levels):
    """Z from row boundaries: `bounds[k]` (per column) is where Z passes
    `levels[k]`, linear in between; bounds may be arrays of length W."""
    Z = np.full((H, W), np.nan)
    b = [np.broadcast_to(np.asarray(v, float), (W,)) for v in bounds]
    for col in range(W):
        xs = [v[col] for v in b]
        order = np.argsort(xs)
        Z[:, col] = np.interp(YY[:, col], np.asarray(xs)[order], np.asarray(levels, float)[order])
    return Z


def periodic_edt(mask):
    """Distance to the nearest cell outside `mask`, across the seams."""
    big = np.tile(mask, (3, 3))
    return ndi.distance_transform_edt(big)[H:2 * H, W:2 * W]


def top_seam(a, rows=6):
    """Copy the first rows into the last `rows` (under the logos), so tiles
    touching the top edge can cross the seam."""
    a = a.copy()
    a[H - rows:] = a[:rows]
    return a


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def islands(n, layout, target, margin=7, smooth=6):
    """Land where the noise `n` is high: about `target` of the page, kept
    clear of the text and the logos, with slivers opened away."""
    room = ~text_mask(layout, margin) & (YY < LOGO_TOP - 6)
    thr = np.quantile(n[room], max(0.0, 1 - target * H * W / room.sum()))
    land = (n > thr) & room
    return ndi.binary_opening(np.tile(land, (3, 3)), disk(smooth))[H:2 * H, W:2 * W]


def shore_field(rng, land, scale=10, jitter=0.2, base=2.4):
    """Z rising with the distance from the shore: level 3 right from the
    beach (level 2 and ponds fill what it leaves), level 7 in the cores of
    the biggest islands."""
    return base + periodic_edt(land) / scale + jitter * noise(rng, 8)


# --- Designs -------------------------------------------------------------------

DESIGNS = {}


def design(name, title, text):
    def wrap(fn):
        DESIGNS[name] = dict(fn=fn, title=title, text=text)
        return fn
    return wrap


@design("hills", "Rolling hills",
        "Seven layers bent into slow waves that repeat exactly across the page width: "
        "level 7 on the floor above the logos, ponds and level 2 along the crest. Every "
        "tile is seated where it touches the most, so the smaller ones close up every "
        "notch left by the bigger ones.")
def d_hills(layout, seed=22):
    c = Cover(seed)
    r = c.rng
    lift = 0 if layout == "top" else 26
    base = [472, 410, 356, 308, 266, 226, 214]
    bounds = [base[0]] + [b - lift * k / 6 + wave(r, 26, 9) for k, b in enumerate(base[1:], 1)]
    Z = strata_field(bounds, [7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5]) + 0.25 * noise(r, 14)
    mask = (YY >= bounds[-1][None, :]) & (YY < 472) & ~text_mask(layout, 5)
    c.fill(mask, Z)
    return c


@design("archipelago", "Archipelago",
        "Islands over about 40% of the page, thinning out under the title. Every island "
        "is a small census: level 7 in the cores of the big ones, stepping down to "
        "ponds on the beach. The islands wrap round all four edges.")
def d_archipelago(layout, seed=25):
    c = Cover(seed)
    r = c.rng
    land = top_seam(islands(noise(r, 40) + 0.8 * (YY / H - 0.4), layout, 0.46))
    c.fill(land, shore_field(r, land))
    return c


@design("continents", "Continents",
        "Fewer, bigger land masses over nearly half the page, so the level-7 cores grow into "
        "plains and the coastlines get long and ragged.")
def d_continents(layout, seed=31):
    c = Cover(seed)
    r = c.rng
    land = top_seam(islands(noise(r, 62) + 0.5 * (YY / H - 0.4), layout, 0.52, smooth=8))
    c.fill(land, shore_field(r, land, scale=11))
    return c


@design("atolls", "Atolls",
        "Many small islands, a third of the page, spread evenly. Few are big enough "
        "for level 6 or 7, so the page is mostly levels 2 to 5: finer grain, more rhythm.")
def d_atolls(layout, seed=32):
    c = Cover(seed)
    r = c.rng
    land = top_seam(islands(noise(r, 19), layout, 0.38, smooth=4))
    c.fill(land, shore_field(r, land, scale=6, base=2.2))
    return c


@design("shoals", "Shoals",
        "An archipelago that rises out of the bottom of the page: a coast above the "
        "logos breaking up into ever smaller islands towards the title.")
def d_shoals(layout, seed=33):
    c = Cover(seed)
    r = c.rng
    land = top_seam(islands(noise(r, 30) + 2.6 * (YY / H - 0.55), layout, 0.48))
    c.fill(land, shore_field(r, land))
    return c


@design("lakes", "Lakes",
        "The archipelago turned inside out: a mainland from just under the title to "
        "the logos, with lakes of field in it. The mainland's top coast is ragged like "
        "the islands' beaches.")
def d_lakes(layout, seed=34):
    c = Cover(seed)
    r = c.rng
    top = (215 if layout == "top" else 175) + 14 * noise(r, 22)
    lakes = noise(r, 24)
    land = (YY > top) & (lakes < np.quantile(lakes, 0.8))
    land &= ~text_mask(layout, 7) & (YY < LOGO_TOP - 6)
    land = ndi.binary_opening(np.tile(land, (3, 3)), disk(6))[H:2 * H, W:2 * W]
    c.fill(land, shore_field(r, land))
    return c


# --- Open designs (round 6): the pattern runs under the text -------------------------
# The text boxes and the logo strip are laid over the pattern with some
# transparency afterwards, so these designs ignore the layout and build one
# map each ("open").

def aniso_noise(rng, s_along, s_across, angle):
    """Smooth periodic noise stretched along `angle` (degrees from the
    horizontal, clockwise on the page): a Gaussian filter in Fourier space."""
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    t = np.radians(angle)
    u = kx * np.cos(t) + ky * np.sin(t)
    v = -kx * np.sin(t) + ky * np.cos(t)
    f = np.exp(-2 * np.pi ** 2 * ((s_along * u) ** 2 + (s_across * v) ** 2))
    z = np.fft.ifft2(np.fft.fft2(rng.normal(size=(H, W))) * f).real
    return z / z.std()


def open_land(n, target, smooth=6):
    """Land where `n` is high over about `target` of the page, slivers opened away."""
    land = n > np.quantile(n, 1 - target)
    return ndi.binary_opening(np.tile(land, (3, 3)), disk(smooth))[H:2 * H, W:2 * W]


def open_cover(seed, land, **shore):
    c = Cover(seed)
    c.fill(land, shore_field(c.rng, land, **shore))
    return c


def graded_noise(rng, w, big=40, small=16):
    """Noise whose island size shrinks from `big` where w = 1 to `small`
    where w = 0, so a coast breaks up into ever smaller islands."""
    w = np.clip(w, 0, 1)
    return (w * noise(rng, big) + (1 - w) * noise(rng, small)) / np.hypot(w, 1 - w)


def shoal(seed, w, gain=1.6, target=0.52, **kw):
    r = np.random.default_rng(seed)
    return open_cover(seed, open_land(graded_noise(r, w, **kw) + gain * (np.clip(w, 0, 1) - 0.5),
                                      target, smooth=6))


@design("open-sea", "Open sea",
        "The archipelago over the whole page, text and logos included: islands of "
        "all sizes, evenly spread, on about half the page.")
def d_open_sea(layout, seed=41):
    r = np.random.default_rng(seed)
    return open_cover(seed, open_land(0.9 * noise(r, 30) + 0.35 * noise(r, 13), 0.5, smooth=6))


@design("islets", "Islands and islets",
        "Two scales at once: a few big islands with level-7 cores, and a scatter "
        "of islets (levels 2 to 4) in the channels between them.")
def d_islets(layout, seed=42):
    r = np.random.default_rng(seed)
    big = open_land(noise(r, 52), 0.32, smooth=8)
    near = ndi.binary_dilation(np.tile(big, (3, 3)), disk(8))[H:2 * H, W:2 * W]
    small = open_land(noise(r, 12), 0.34, smooth=5) & ~near
    return open_cover(seed, big | small)


@design("lagoons", "Lagoons",
        "Ring islands: the bigger islands hold a lagoon of field in their middle, "
        "so the deepest levels form rings instead of solid cores.")
def d_lagoons(layout, seed=43):
    r = np.random.default_rng(seed)
    n = noise(r, 34)
    lagoon = open_land(n + 0.35 * noise(r, 10), 0.07, smooth=4)
    land = open_land(n, 0.52, smooth=6) & ~ndi.binary_dilation(
        np.tile(lagoon, (3, 3)), disk(3))[H:2 * H, W:2 * W]
    return open_cover(seed, ndi.binary_opening(np.tile(land, (3, 3)), disk(4))[H:2 * H, W:2 * W])


@design("sandbanks", "Sandbanks",
        "Islands drawn out sideways into long banks that run across the page and "
        "off both edges, like sandbanks at low tide.")
def d_sandbanks(layout, seed=44):
    r = np.random.default_rng(seed)
    return open_cover(seed, open_land(aniso_noise(r, 60, 13, -8), 0.47, smooth=5), scale=8)


@design("drift", "Drift",
        "Long islands tilted on a slant, as if blown by one wind, denser towards the "
        "bottom right.")
def d_drift(layout, seed=45):
    r = np.random.default_rng(seed)
    bias = 0.9 * (0.4 * XX / W + YY / H - 0.7)
    return open_cover(seed, open_land(aniso_noise(r, 48, 14, -32) + bias, 0.47, smooth=5),
                      scale=8)


@design("rising", "Rising shoals",
        "Shoals over the whole page: a mainland under the logos breaks up into ever "
        "smaller islands towards the top, and the smallest islets scatter round the "
        "title.")
def d_rising(layout, seed=46):
    return shoal(seed, (YY / H - 0.15) / 0.8)


@design("falling", "Falling shoals",
        "Shoals upside down: the mainland hangs from the top edge behind the title "
        "and breaks up downwards into islets round the name and the logos.")
def d_falling(layout, seed=47):
    return shoal(seed, (0.75 - YY / H) / 0.75)


@design("corner", "Corner shoals",
        "The coast in the bottom-left corner, breaking up diagonally into ever smaller "
        "islands towards the top right. One diagonal gesture across the page.")
def d_corner(layout, seed=48):
    return shoal(seed, (0.65 * (1 - XX / W) + YY / H - 0.35) / 1.1)


@design("arc", "Island arc",
        "A chain of islands sweeping up from the bottom left and curling over to the "
        "right edge: big islands along the spine, islets scattered off it.")
def d_arc(layout, seed=49):
    t = XX / W
    y = H * (0.86 - 0.95 * t + 0.55 * t * t)          # the arc's spine
    return shoal(seed, 1 - np.abs(YY - y) / (0.3 * H), gain=2.0, target=0.48)


@design("strait", "Strait",
        "Two unequal shores, a big one bottom right and a smaller one top left, "
        "breaking up into a strait of islets between them.")
def d_strait(layout, seed=50):
    s = XX / W + YY / H                               # 0 top left .. 2 bottom right
    return shoal(seed, np.maximum((s - 0.95) / 0.75, (0.6 - s) / 0.5))


OPEN = ["open-sea", "islets", "lagoons", "sandbanks", "drift",
        "rising", "falling", "corner", "arc", "strait"]


# --- Round 7: the favourites again, and the coast of Paraty ------------------------
# Paraty Mirim and the Saco do Mamanguá are a drowned coast (a ria): the
# Serra do Mar runs straight into the sea, and the sea has flooded its
# valleys, so the coast is a comb of long, narrow inlets between steep
# fingers of land, with islands off the headlands. The designs below
# build that the way it formed: a terrain (a coast gradient plus graded
# noise for the hills), valleys carved into it along the zero lines of
# noise stretched in the inland direction, and the sea let in to a level.
# A valley floor rises slowly inland, so the flooded part is long and
# narrows to a point, as a drowned valley does.

def periodic(fn, m, *a, **kw):
    """A scipy.ndimage morphology op on the torus."""
    return fn(np.tile(m, (3, 3)), *a, **kw)[H:2 * H, W:2 * W]


def terrain(rng, w, angle, gain=2.0, rough=0.6, vdepth=1.2, vwidth=0.35, along=110, across=28):
    """Height: rising with w (0 open sea, 1 mainland), hills from graded
    noise, and valleys running along `angle` (see `aniso_noise`)."""
    T = gain * (np.clip(w, 0, 1) - 0.5) + rough * graded_noise(rng, w)
    return T - vdepth * np.exp(-(aniso_noise(rng, along, across, angle) / vwidth) ** 2)


def flood(T, target, smooth=5):
    """Land where the terrain stands above the sea, over about `target` of the page."""
    return periodic(ndi.binary_opening, open_land(T, target, smooth=4), disk(smooth))


def path_distance(path, width, taper):
    """Distance to a polyline (in page fractions), in units of a half-width
    that narrows along it from `width` to (1 - taper) * `width`."""
    p = np.asarray(path, float) * [W, H]
    d, t = np.full((H, W), np.inf), np.zeros((H, W))
    L = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    for (x0, y0), (x1, y1), l0, l1 in zip(p[:-1], p[1:], L[:-1], L[1:]):
        dx, dy = x1 - x0, y1 - y0
        s = np.clip(((XX - x0) * dx + (YY - y0) * dy) / (dx * dx + dy * dy), 0, 1)
        dd = np.hypot(XX - x0 - s * dx, YY - y0 - s * dy)
        closer = dd < d
        d, t = np.where(closer, dd, d), np.where(closer, (l0 + s * (l1 - l0)) / L[-1], t)
    return d / (width * (1 - taper * t))


def lagoon_land(rng, sigma=34, lagoons=0.07, target=0.52):
    n = noise(rng, sigma)
    lagoon = open_land(n + 0.35 * noise(rng, 10), lagoons, smooth=4)
    land = open_land(n, target, smooth=6) & ~periodic(ndi.binary_dilation, lagoon, disk(3))
    return periodic(ndi.binary_opening, land, disk(4))


CORNER = (0.65 * (1 - XX / W) + YY / H - 0.35) / 1.1     # the corner shoals' coast


@design("open-sea-2", "Open sea II",
        "Open sea again, on a new draw and with a wider spread of island sizes: "
        "a few large ones with level-7 cores among many small ones.")
def d_open_sea_2(layout, seed=61):
    r = np.random.default_rng(seed)
    return open_cover(seed, open_land(0.85 * noise(r, 38) + 0.5 * noise(r, 14), 0.5, smooth=6))


@design("lagoons-2", "Lagoons II",
        "Lagoons on a new draw: ring islands round a lagoon of field.")
def d_lagoons_2(layout, seed=73):
    return open_cover(seed, lagoon_land(np.random.default_rng(seed)))


@design("lagoons-3", "Lagoons III",
        "Lagoons with slightly bigger islands and more lagoons, some islands "
        "holding several.")
def d_lagoons_3(layout, seed=62):
    return open_cover(seed, lagoon_land(np.random.default_rng(seed), 36, 0.08, 0.53))


@design("corner-2", "Corner shoals II",
        "Corner shoals on a new draw: the coast in the bottom-left corner, breaking "
        "up diagonally towards the top right.")
def d_corner_2(layout, seed=63):
    return shoal(seed, CORNER)


@design("corner-3", "Corner shoals III",
        "The mirror gesture: the coast in the bottom-right corner, the islets "
        "scattering towards the top left, round the title.")
def d_corner_3(layout, seed=64):
    return shoal(seed, (0.65 * XX / W + YY / H - 0.35) / 1.1)


@design("arc-2", "Island arc II",
        "A flatter arc on a new draw, sweeping from the left edge low under the "
        "middle and up to the right edge.")
def d_arc_2(layout, seed=65):
    y = H * (0.42 + 0.9 * (XX / W - 0.45) ** 2)       # the arc's spine, a smile
    return shoal(seed, 1 - np.abs(YY - y) / (0.28 * H), gain=2.0, target=0.48)


@design("paraty-mirim", "Paraty Mirim",
        "The drowned coast of Paraty Mirim on the corner shoals: mountains in the "
        "bottom-left corner flooded by the sea, so long, narrow inlets run deep into "
        "the land between fingers of it, and islands lie off the headlands towards "
        "the top right.")
def d_paraty_mirim(layout, seed=71):
    r = np.random.default_rng(seed)
    return open_cover(seed, flood(terrain(r, CORNER, -40, vdepth=1.5, vwidth=0.3), 0.55))


@design("paraty-mirim-2", "Paraty Mirim II",
        "The same coast on another draw, the mainland a little higher, so there "
        "is more land and fewer islands.")
def d_paraty_mirim_2(layout, seed=77):
    r = np.random.default_rng(seed)
    return open_cover(seed, flood(terrain(r, CORNER, -40, gain=2.4, vdepth=1.6, vwidth=0.3), 0.55))


@design("mamangua", "Saco do Mamanguá",
        "The Saco do Mamanguá: one long, narrow drowned valley that runs from the "
        "sea at the top right deep into the mountains, between the mainland and the "
        "long ridge beside it, with small inlets off its shores.")
def d_mamangua(layout, seed=83):
    r = np.random.default_rng(seed)
    w = np.maximum((0.45 * XX / W + YY / H + 0.1), 0.9 * (1 - XX / W) + 0.5 * YY / H - 0.25)
    T = 2.2 * (np.clip(w, 0, 1) - 0.5) + 0.5 * graded_noise(r, w)
    T -= 0.5 * np.exp(-(aniso_noise(r, 110, 26, -55) / 0.3) ** 2)
    T -= 4 * np.exp(-path_distance([(0.8, 0.22), (0.66, 0.42), (0.52, 0.64), (0.34, 0.9)],
                                   22, 0.5) ** 4)
    return open_cover(seed, flood(T, 0.6))


@design("paraty-bay", "Baía de Paraty",
        "The bay of Paraty: the mainland wraps round the left, the bottom and the "
        "right of the page, its drowned coast full of inlets, and the bay in the "
        "middle is strewn with islands, opening to the sea at the top.")
def d_paraty_bay(layout, seed=68):
    r = np.random.default_rng(seed)
    x = XX / W
    w = 0.3 + 0.7 * np.maximum.reduce([(0.3 - x) / 0.3, (x - 0.7) / 0.3, (YY / H - 0.6) / 0.32])
    return open_cover(seed, flood(terrain(r, w, -90, gain=2.2), 0.55))


ROUND7 = ["open-sea-2", "lagoons-2", "lagoons-3", "corner-2", "corner-3", "arc-2",
          "paraty-mirim", "paraty-mirim-2", "mamangua", "paraty-bay"]


# --- Round 8: the real coast of Paraty Mirim ------------------------------------------
# The land is traced from the user's silhouettes of satellite images
# (studies/cover/silhouettes, 2:3, land green, sea blue), not drawn from
# noise. The cover need not wrap nor be a still life at the trim, so the
# tiles are packed on a page with a MARGIN of cells all round (the coast
# mirrored into it) and cut at the trim: a tile may run off the page, and
# no tile wraps back onto it. The padded page is itself a periodic still
# life, so everything inside the trim is one.

SILHOUETTES = ASSETS / "silhouettes"
MARGIN = 48                     # > the 40-cell span of a level-7 tile, a multiple of 6
PAGE = (W, H)


def set_page(w, h):
    """Rebind the page size (cells, multiples of 6) that the helpers read."""
    global W, H, A, B, AA, BB, PONDS, YY, XX, X
    W, H = w, h
    A, B = H // 3, W // 3
    AA, BB = np.mgrid[0:A, 0:B]
    PONDS = (AA + BB) % 2 == 0
    YY, XX = np.mgrid[0:H, 0:W].astype(float)
    X = XX[0]
    _KERNELS.clear()
    _VKERNELS.clear()


def traced(path, w=PAGE[0], h=PAGE[1]):
    """Land on a w x h page from a silhouette: where the green outweighs the
    blue, averaged over each cell."""
    rgb = np.asarray(Image.open(path).convert("RGB"), float)
    g = Image.fromarray((rgb[..., 1] - rgb[..., 2]).astype(np.float32), "F")
    return np.asarray(g.resize((w, h), Image.BOX)) > 0


def map_cover(path, seed, sea=False, **shore):
    """The label map of a traced coast, packed with a margin and cut at the
    trim. With `sea`, the tiles fill the water instead of the land, their
    levels rising with the distance from the shore like a depth chart."""
    land = np.pad(traced(path) ^ sea, MARGIN, mode="symmetric")
    set_page(PAGE[0] + 2 * MARGIN, PAGE[1] + 2 * MARGIN)
    try:
        c = open_cover(seed, land, **shore)
        lab, ok = c.labels(), c.verify()
    finally:
        set_page(*PAGE)
    log(f"{path.stem}: tiles {dict(sorted(c.placed.items()))}, padded still life {ok}")
    return lab[MARGIN:MARGIN + PAGE[1], MARGIN:MARGIN + PAGE[0]]


MAPS = {  # name: (silhouette, seed, options)
    "paraty-a-land": ("paraty-a", 91, {}),
    "paraty-b-land": ("paraty-b", 92, {}),
    "paraty-c-land": ("paraty-c", 93, {}),
    "paraty-a-sea": ("paraty-a", 94, {"sea": True}),
    "paraty-b-sea": ("paraty-b", 95, {"sea": True}),
    "paraty-c-sea": ("paraty-c", 96, {"sea": True}),
}
# The ramp steps of levels 1-7 on a map: the palest steps are dropped, so the
# ponds and small tiles along the shore still stand off the field and the
# coast reads from a distance.
COAST = (3, 4, 4, 5, 5, 6, 7)


def build_maps(names, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name in names or list(MAPS):
        sil, seed, shore = MAPS[name]
        lab = map_cover(SILHOUETTES / f"{sil}.png", seed, **shore)
        Image.fromarray(lab, "L").save(out / f"{name}_open.png", optimize=True)
OPEN += ROUND7


# --- Palettes --------------------------------------------------------------------
# A palette is a ramp of seven tile grounds, levels 1..7, designed in OKLCH:
# lightness carries the order (level 1 lightest), the hue walks a short way
# round the wheel, and the chroma stays moderate. The field is chosen
# separately. Every colour, and every derived ink, is moved into the Coated
# FOGRA39 gamut. Text boxes stay #F3F4F4 with black text (the inline-code look).


def _ramp(Ls, Cs, hs):
    return list(zip(Ls, Cs, hs))


RAMPS = {  # key: (label, [(L, C, h) for levels 1..7])
    "denim": ("Denim", _ramp([.95, .905, .85, .74, .63, .51, .39], [.02, .03, .045, .07, .09, .10, .09],
                             [238, 241, 245, 248, 252, 256, 260])),
    "ember": ("Ember", _ramp([.93, .87, .80, .71, .62, .52, .41], [.05, .09, .13, .15, .16, .15, .12],
                             [80, 70, 58, 40, 22, 5, 345])),
    "indigo": ("Indigo", _ramp([.95, .90, .84, .73, .62, .50, .38], [.025, .04, .06, .085, .11, .12, .11],
                               [255, 260, 266, 272, 278, 284, 290])),
    "terracotta": ("Terracotta", _ramp([.93, .88, .81, .71, .61, .51, .40], [.03, .05, .08, .10, .11, .10, .08],
                                       [78, 70, 60, 50, 40, 32, 25])),
    "tide": ("Tide", _ramp([.94, .89, .84, .74, .63, .52, .40], [.04, .07, .085, .095, .10, .10, .09],
                           [160, 168, 175, 198, 222, 248, 272])),
    "orchid": ("Orchid", _ramp([.94, .89, .83, .73, .62, .51, .40], [.03, .06, .08, .11, .12, .12, .10],
                               [40, 32, 25, 0, 340, 318, 295])),
}
FIELDS = {
    "apricot": ("Apricot", "#FFD9A8"), "peach": ("Peach", "#FFCBA4"), "cream": ("Cream", "#FFF0D6"),
    "sky": ("Sky", "#CFE3EF"), "ice": ("Ice", "#E4EFF5"), "butter": ("Butter", "#FFD166"),
}
TONES = {"subtle": 0.09, "strong": 0.16}   # lightness step of the tone-on-tone ink

# The press profile: Coated FOGRA39 from the ICC registry
# (color.org/registry/profiles/Coated_Fogra39L_VIGC_300.icc, 8.6 MB, not in
# git), passed with --profile. The Agfa SWOP profile that ships with Windows
# is no substitute: it does not even round-trip muted colours.
PROFILE = None


def _to_lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _from_lin(c):
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)
    return c * 255


def oklch(rgb):
    """sRGB 0-255 -> (L, C, h in degrees)."""
    r, g, b = _to_lin(rgb)
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return L, float(np.hypot(a, bb)), float(np.degrees(np.arctan2(bb, a)) % 360)


def _oklch_lin(L, C, h):
    a, b = C * np.cos(np.radians(h)), C * np.sin(np.radians(h))
    l, m, s = [(L + p * a + q * b) ** 3 for p, q in
               ((0.3963377774, 0.2158037573), (-0.1055613458, -0.0638541728),
                (-0.0894841775, -1.2914855480))]
    return np.array([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                     -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                     -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])


def from_oklch(L, C, h):
    """OKLCH -> sRGB hex, chroma reduced until the colour is in sRGB."""
    lo, hi = 0.0, C
    if np.all((_oklch_lin(L, C, h) >= 0) & (_oklch_lin(L, C, h) <= 1)):
        lo = C
    for _ in range(30):
        if hi - lo < 1e-4:
            break
        mid = (lo + hi) / 2
        c = _oklch_lin(L, mid, h)
        lo, hi = (mid, hi) if np.all((c >= 0) & (c <= 1)) else (lo, mid)
    rgb = np.clip(np.round(_from_lin(_oklch_lin(L, lo, h))), 0, 255).astype(int)
    return "#%02X%02X%02X" % tuple(rgb)


def _hex(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def _lab(rgb):
    """sRGB 0-255 -> CIELAB (D65)."""
    c = _to_lin(rgb)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def gamut_check(rgb):
    """True where LittleCMS says the sRGB colour is inside the press gamut
    (its gamut check paints out-of-gamut colours in the alarm grey)."""
    from PIL import ImageCms
    F = ImageCms.Flags
    srgb = ImageCms.createProfile("sRGB")
    t = ImageCms.buildProofTransform(srgb, srgb, ImageCms.getOpenProfile(str(PROFILE)),
                                     "RGB", "RGB", renderingIntent=0, proofRenderingIntent=1,
                                     flags=F.SOFTPROOFING | F.GAMUTCHECK)
    rgb = np.asarray(rgb, np.uint8).reshape(1, -1, 3)
    out = np.asarray(ImageCms.applyTransform(Image.fromarray(rgb), t))[0]
    return ~np.all(out == 0x7F, axis=-1)


_GAMUT = None


def print_safe(hexes):
    """Each colour, or the nearest printable one when it is outside the
    press gamut (CIE76, lightness weighted a little more so ramps keep
    their order)."""
    global _GAMUT
    if _GAMUT is None:
        v = np.arange(0, 256, 4)
        grid = np.stack(np.meshgrid(v, v, v, indexing="ij"), -1).reshape(-1, 3)
        _GAMUT = grid[gamut_check(grid)]
        _GAMUT = (_GAMUT, _lab(_GAMUT))
    rgb = np.array([_hex(h) for h in hexes], np.uint8)
    inside = gamut_check(rgb)
    out = []
    for c, ok, h in zip(rgb, inside, hexes):
        if ok:
            out.append(h)
            continue
        d = (_GAMUT[1] - _lab(c)) * np.array([1.3, 1.0, 1.0])
        out.append("#%02X%02X%02X" % tuple(_GAMUT[0][np.argmin((d ** 2).sum(-1))]))
    return out


def _tone(hexcol, step):
    """The same hue a lightness step darker (lighter on very dark grounds)."""
    L, C, h = oklch(_hex(hexcol))
    L2 = L - step if L - step > 0.24 else L + step
    return from_oklch(L2, C * 1.05, h)


def palettes(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    res = {"fields": {}, "ramps": {}}
    for key, (label, hexcol) in FIELDS.items():
        res["fields"][key] = dict(label=label, hex=print_safe([hexcol])[0], screen=hexcol)
    for key, (label, spec) in RAMPS.items():
        grounds = print_safe([from_oklch(*c) for c in spec])
        last = oklch(_hex(grounds[-1]))
        dark = from_oklch(min(last[0] - 0.14, 0.26), last[1] * 0.8, last[2])
        entry = dict(label=label, grounds=grounds, dark=print_safe([dark])[0], light="#FBF8F1")
        for tk, step in TONES.items():
            entry[f"tone_{tk}"] = print_safe([_tone(g, step) for g in grounds])
        res["ramps"][key] = entry
        log(f"{key:12s} {' '.join(grounds)}")
    (out / "palettes.json").write_text(json.dumps(res, indent=1))


# --- Commands --------------------------------------------------------------------

def build(names, out: Path, layouts=tuple(LAYOUTS)):
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "designs.json"
    meta = {d["name"]: d for d in json.loads(meta_path.read_text())} if meta_path.exists() else {}
    for name in names or list(DESIGNS):
        d = DESIGNS[name]
        maps = {}
        for layout in (["open"] if name in OPEN else layouts):
            c = d["fn"](layout)
            ok = c.verify()
            Image.fromarray(c.labels(), "L").save(out / f"{name}_{layout}.png", optimize=True)
            cover = float((c.ground > 0).mean())
            free = int((c.sites(1, c.mask, c.mask) & (c.vlevel == 0)).sum())
            maps[layout] = dict(tiles={str(L): c.placed[L] for L in range(1, 8)},
                                still_life=ok, tile_cover=round(cover, 3), free_ponds=free)
            log(f"{name} / {layout}: tiles {dict(sorted(c.placed.items()))}, periodic still "
                f"life {ok}, page {cover:.0%}, free pond sites {free}")
        meta[name] = dict(name=name, title=d["title"], text=d["text"], maps=maps)
    order = [n for n in DESIGNS if n in meta]
    meta_path.write_text(json.dumps([meta[n] for n in order], indent=1))
    (out / "layouts.json").write_text(json.dumps(
        {"sprites": SPRITES, "layouts": LAYOUTS, "labels": LAYOUT_LABELS, "logos": LOGOS,
         "draft_w": DRAFT_W, "W": W, "H": H}))


# --- Print files -----------------------------------------------------------------
# The background alone, text and logos left to Canva. A small tile in a
# hole takes its host's colours and the grout inside the hole is closed
# (labels 40+, 140+, 200+), so the hole reads as part of the host. PNG at
# a whole number of pixels per cell; SVG with one path per colour, the
# outline of each colour's cells traced into polygons, so it is exact at any
# size and has no seams between neighbouring cells.

PAIRS = {"denim-apricot": ("denim", "apricot"), "ember-sky": ("ember", "sky")}


def load_palettes(data=None):
    """The ramps and fields: `data`/palettes.json when a `palettes` run left
    one there, else the versioned assets/palettes.json (the print-safe
    colours of the delivered covers)."""
    p = data / "palettes.json" if data else None
    return json.loads((p if p and p.exists() else ASSETS / "palettes.json").read_text())


def layers(lab, pal, ramp, field, shade=range(1, 8)):
    """Paint order: the field, each level's tile ground (ink cells included),
    each level's ink. `shade[L - 1]` is the step of the ramp that level L
    takes. [(hex, mask)]"""
    v, lab = pal["ramps"][ramp], lab.astype(int)
    v = {k: [v[k][s - 1] for s in shade] for k in ("grounds", "tone_subtle")}
    level = np.where(lab < 20, lab, np.where(lab < 40, lab - 20, 0))
    host = np.where(lab >= 200, lab - 200, 0)
    small = np.where((lab >= 40) & (lab < 100), lab - 40, np.where((lab >= 140) & (lab < 200), lab - 140, -1))
    host = np.where(small >= 0, small // 10 + 3, host)
    ground = np.where(host > 0, host, level)
    ink = np.where((lab >= 1) & (lab <= 7), lab, np.where((lab >= 40) & (lab < 100), host, 0))
    out = [(pal["fields"][field]["hex"], np.ones(lab.shape, bool))]
    out += [(v["grounds"][L - 1], ground == L) for L in range(1, 8)]
    out += [(v["tone_subtle"][L - 1], ink == L) for L in range(1, 8)]
    merged = {}                                         # one path per colour
    for c, m in out:
        if m.any():
            merged[c] = merged[c] | m if c in merged else m
    return list(merged.items())


def outline(mask):
    """SVG path data tracing the cells of `mask` (nonzero fill): every
    boundary edge, directed with the region on its right, chained into
    loops, collinear steps merged."""
    m = np.pad(mask, 1)
    edges = {}
    for (dy, dx), (p, q) in {(-1, 0): ((0, 0), (1, 0)), (0, 1): ((1, 0), (1, 1)),
                             (1, 0): ((1, 1), (0, 1)), (0, -1): ((0, 1), (0, 0))}.items():
        ys, xs = np.nonzero(m[1:-1, 1:-1] & ~m[1 + dy:m.shape[0] - 1 + dy, 1 + dx:m.shape[1] - 1 + dx])
        for y, x in zip(ys.tolist(), xs.tolist()):
            edges.setdefault((x + p[0], y + p[1]), []).append((x + q[0], y + q[1]))
    parts = []
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
        pts = [pt for k, pt in enumerate(pts)            # drop the collinear corners
               if not (pts[k - 1][0] == pt[0] == pts[(k + 1) % len(pts)][0]
                       or pts[k - 1][1] == pt[1] == pts[(k + 1) % len(pts)][1])]
        d = [f"M{pts[0][0]} {pts[0][1]}"]
        for a, b in zip(pts, pts[1:]):
            d.append(f"h{b[0] - a[0]}" if b[1] == a[1] else f"v{b[1] - a[1]}")
        parts.append("".join(d) + "z")
    return "".join(parts)


def write_page(ls, stem: Path, scale):
    """`stem`.png at `scale` px per cell and `stem`.svg, both 160 mm wide,
    from paint layers [(hex, mask)] (the first one the field)."""
    h, w = ls[0][1].shape
    rgb = np.zeros((h, w, 3), np.uint8)
    for c, m in ls:
        rgb[m] = _hex(c)
    dpi = scale * 25.4 * w / 160
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    Image.fromarray(rgb).resize((w * scale, h * scale), Image.NEAREST).save(png, dpi=(dpi, dpi), optimize=True)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="160mm" height="{160 * h / w:g}mm" '
             f'viewBox="0 0 {w} {h}" shape-rendering="crispEdges">',
             f'<rect width="{w}" height="{h}" fill="{ls[0][0]}"/>']
    lines += [f'<path fill="{c}" d="{outline(m)}"/>' for c, m in ls[1:]]
    svg.write_text("\n".join(lines + ["</svg>"]))
    log(f"{stem.name}: png {png.stat().st_size / 1e6:.1f} MB, svg {svg.stat().st_size / 1e6:.2f} MB")


def print_files(names, data: Path, out: Path, pairs=tuple(PAIRS), scale=10, shade=range(1, 8)):
    out.mkdir(parents=True, exist_ok=True)
    pal = load_palettes(data)
    for name in names:
        lab = np.asarray(Image.open(data / f"{name}_open.png"))
        assert lab.shape == (H, W)
        for pair in pairs:
            write_page(layers(lab, pal, *PAIRS[pair], shade), out / f"{name}_{pair}", scale)


# --- Back cover ------------------------------------------------------------------
# The defence flyer's skyline (studies/flyer, design skyline2, the approved
# solve and its seed-4 tile field), made 160 x 240 mm: the flyer is 540 x
# 768 cells, and 42 dead rows let into the middle of its empty band (rows
# 138-429, where no live cell is within reach) give 540 x 810, exactly 2:3,
# without touching a single neighbourhood. Cells are 160/540 = 0.30 mm, and
# 7 px per cell is 600 dpi. Coloured from the front cover's ramps: the
# buildings, the ground and the ribbon in one tile ground with its
# tone-on-tone ink, the grain in the same colour on the field.

BACK_ROWS, BACK_AT = 810, 284
BACK = {  # name: (ramp, field, level of the building colour)
    "denim-apricot-deep": ("denim", "apricot", 6), "denim-apricot-mid": ("denim", "apricot", 5),
    "ember-sky-deep": ("ember", "sky", 6), "ember-sky-mid": ("ember", "sky", 5),
}


def back_cover(data: Path, out: Path, scale=7):
    sys.path.insert(0, str(HERE.parent / "flyer"))
    import flyer as F
    from gol_mosaics import filled_background

    d, grain = F.DESIGNS["skyline2"](), F.load_solve("skyline2").astype(bool)
    tiles = ~d.free
    field = filled_background(tiles, fill="auto", centred=True, level=4, seed=4)
    ink = field > 0
    assert not (grain | ink)[BACK_AT - 3:BACK_AT + 3].any()
    grow = lambda a, v: np.insert(a, BACK_AT, np.full((BACK_ROWS - a.shape[0], a.shape[1]), v), 0)
    grain, ink, tiles = grow(grain, False), grow(ink, False), grow(tiles, False)
    whole = grain | ink
    bounded = periodic_still_life(np.pad(whole, 2))
    log(f"back cover {whole.shape}: still life {bounded}")
    assert bounded
    pal = load_palettes(data)
    out.mkdir(parents=True, exist_ok=True)
    for name, (ramp, fld, L) in BACK.items():
        v = pal["ramps"][ramp]
        write_page([(pal["fields"][fld]["hex"], np.ones(whole.shape, bool)),
                    (v["grounds"][L - 1], tiles), (v["tone_subtle"][L - 1], ink),
                    (v["grounds"][L - 1], grain)], out / f"back_{name}", scale)


# --- Round 9: the back cover lifted, room for the colophon ---------------------------
# The city moves up LIFT cells (25 mm) and a band of tiles fills the room
# it leaves under the ground, for the bibliographic block laid over it.
# The rows come out of the empty band above the city, so the page stays
# 540 x 810 and no neighbourhood changes. Tiles are packed as on the front
# (levels 7 to 2 biggest first, then ponds), on the page padded by MARGIN
# and cut at the trim, so they may run off any edge. No tile cell comes
# within two cells of the grain: the two are still lifes each, and no cell
# sees both. Every tile stays whole on its own colour; only the trim cuts
# tiles.

LIFT = 84                       # 24.9 mm at 0.30 mm per cell (round 9 had 102, 30 mm)
BACK_PAGE = (540, 810)


def lifted_skyline(lift=LIFT):
    """The flyer's grain, level-4 ink and tile region on the back page with
    the city `lift` rows higher and tile region under it."""
    sys.path.insert(0, str(HERE.parent / "flyer"))
    import flyer as F
    from gol_mosaics import filled_background

    d, grain = F.DESIGNS["skyline2"](), F.load_solve("skyline2").astype(bool)
    tiles = ~d.free
    ink = filled_background(tiles, fill="auto", centred=True, level=4, seed=4) > 0
    cut = grain.shape[0] + lift - BACK_PAGE[1]
    lo = BACK_AT - cut // 2
    assert not (grain | ink)[lo - 3:lo + cut + 3].any()
    lifted = lambda a, v: np.vstack([np.delete(a, np.s_[lo:lo + cut], 0),
                                      np.full((lift, a.shape[1]), v)])
    return lifted(grain, False), lifted(ink, False), lifted(tiles, True)


def back_pack(region, grain, seed, cap=None, shift=(0, 0), **shore):
    """Front-cover tiles packed into `region` of the back page, no tile
    cell within two cells of the `grain` (so no dead cell sees both), levels
    rising with the distance from the region's edge (`shore_field`), held
    under about `cap` (give or take a level, at random) if given. The page
    sits `shift` cells (each 0-5) into the pond lattice. (labels, padded
    live cells, the page's slice of them, tile counts)"""
    (h, w), (dy, dx) = region.shape, shift
    pad = ((MARGIN + dy, MARGIN + 6 - dy), (MARGIN + dx, MARGIN + 6 - dx))
    clear = region & ~ndi.binary_dilation(grain, np.ones((5, 5), bool))
    land = np.pad(clear, pad, mode="symmetric")
    set_page(*land.shape[::-1])
    try:
        c = Cover(seed)
        Z = shore_field(c.rng, land, **shore)
        if cap is not None:
            Z = np.minimum(Z, cap + 0.8 * noise(c.rng, 16))
        c.fill(land, Z)
        lab = c.labels()
    finally:
        set_page(*PAGE)
    page = np.s_[pad[0][0]:pad[0][0] + h, pad[1][0]:pad[1][0] + w]
    return lab[page], c.live, page, c.placed


def back_still_life(live, page, *masks):
    """The padded tiles with the page's other live `masks` let in."""
    g = live.astype(bool)
    for m in masks:
        g[page] |= m
    return periodic_still_life(g)


# A city is "flyer" (the flyer's own level-4 mosaic on one colour) or
# "levels" (repacked like the front and coloured by level with its ramp);
# the band under it likewise. A region coloured by level is first filled
# in its ponds' colour, so the buildings keep their outline where no tile
# reaches (spires, the edge along the grain) and the grout inside closes.
LIFTED = {  # name: (city, band, cap on the levels, ramp level of the one colour, of the grain)
    "classic-deep": ("flyer", "flyer", 6.0, 6, 6),
    "classic-mid": ("flyer", "flyer", 6.0, 5, 5),
    "levels-band": ("flyer", "levels", 5.5, 6, 6),
    "levels": ("levels", "levels", 5.5, None, 5),
    "levels-deep": ("levels", "levels", 6.5, None, 6),
    "levels-light": ("levels", "levels", 4.5, None, 4),
}


def lifted_back(data: Path, out: Path, names=(), scale=7, seed=101):
    pal = load_palettes(data)
    v, fld = pal["ramps"]["denim"], pal["fields"]["apricot"]["hex"]
    grounds, inks = ([v[k][s - 1] for s in COAST] for k in ("grounds", "tone_subtle"))
    grain, ink, tiles = lifted_skyline()
    band = np.zeros_like(tiles)
    band[BACK_PAGE[1] - LIFT:] = True
    out.mkdir(parents=True, exist_ok=True)
    for name in names or list(LIFTED):
        city, under, cap, L, G = LIFTED[name]
        if city == "levels":
            lab, live, page, placed = back_pack(tiles, grain, seed, cap)
            ok, region = back_still_life(live, page, grain), tiles
            ls = [(fld, np.ones(tiles.shape, bool))]
        else:
            lab, live, page, placed = back_pack(band, grain | ink, seed, cap)
            ok, region = back_still_life(live, page, grain, ink), band
            ls = [(fld, np.ones(tiles.shape, bool)), (v["grounds"][L - 1], tiles),
                  (v["tone_subtle"][L - 1], ink)]
        if under == "levels":
            ls += [(grounds[0], region)] + layers(lab, pal, "denim", "apricot", COAST)[1:]
        else:
            live_cells = ((lab >= 1) & (lab <= 7)) | ((lab >= 40) & (lab < 100))
            ls += [(v["tone_subtle"][L - 1], live_cells)]
        log(f"back {name}: tiles {dict(sorted(placed.items()))}, padded still life {ok}")
        assert ok
        write_page(ls + [(v["grounds"][G - 1], grain)], out / f"back_{name}", scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "palettes", "print", "back", "map", "lifted"])
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS), help="print, map: the palette pairs")
    ap.add_argument("--data", type=Path, default=HERE / "output" / "data",
                    help="the label maps (and a palettes.json that overrides assets/)")
    ap.add_argument("--scale", type=int, default=10, help="print, map: PNG pixels per cell")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", type=Path, default=HERE / "output")
    ap.add_argument("--profile", type=Path)
    ap.add_argument("--layouts", nargs="*", default=list(LAYOUTS))
    a = ap.parse_args()
    globals()["PROFILE"] = a.profile
    if a.cmd == "build":
        build(a.names, a.out, a.layouts)
    elif a.cmd == "print":
        print_files(a.names or ROUND7, a.data, a.out, a.pairs, scale=a.scale)
    elif a.cmd == "map":
        build_maps(a.names, a.data)
        print_files(a.names or list(MAPS), a.data, a.out, a.pairs, scale=a.scale, shade=COAST)
    elif a.cmd == "back":
        back_cover(a.data, a.out)
    elif a.cmd == "lifted":
        lifted_back(a.data, a.out, a.names)
    else:
        palettes(a.out)


if __name__ == "__main__":
    main()
