# Studies

Finished pieces made with the library, each with the script that made it,
the solved patterns it needs (bit-packed, small enough to version) and its
deliverables. They double as worked examples of `gol_mosaics.freeform` and
`gol_mosaics.compose` on real commissions; the research behind them is in
[`experiments/beyond_tiles/REPORT.md`](../experiments/beyond_tiles/REPORT.md).

| Study | What | Re-render | Re-solve |
|---|---|---|---|
| [`cover/`](cover/README.md) | PhD-thesis cover, 160 x 240 mm: the coast of Paraty Mirim on the front, the flyer skyline over a tile band on the back, gapless mosaics of levels 1-7 as print PNG and SVG | about 1 min, no solver; needs the level-7 census | – |
| [`postcards/`](postcards/README.md) | PhD-defence handouts: five A6 postcards (Conway, the Belfry dragon, Paraty Mirim, two liquid-light windows) and an A3 jury print of Paraty Mirim with land and sea packed with tiles by altitude and depth, vector PDFs | seconds to 2 min, no solver; needs the level-7 census and the FOGRA39 profile | – |
| [`sacred/`](sacred/README.md) | Five A2 posters after photographs of sacred geometry (three mandala thangkas, a Persian star dome, a tile panel), each one symmetric still life of gapless tiles of levels 1-7 at the source's proportions, vector PDFs | about 3 min per sheet, no solver; needs the level-7 census, the FOGRA39 profile and the source photos (not in git) | – |
| [`flyer/`](flyer/README.md) | PhD-defence flyer, Ghent and São Paulo skylines and a vignette, 1080x1536 px, one verified still life each | seconds, from `solves/` | about 30 min per design |
| [`linkedin/`](linkedin/README.md) | 1584x396 banners from two mountain photos, and the halo-filling and fit-rule figures of REPORT.md sections 6-8 | about 40 s, no solver | `poster.py`, 800x200 |
| [`lam_gods/`](lam_gods/README.md) | The Adoration of the Mystic Lamb as one 2480x1656 still life, and the seam study of REPORT.md C9 | seconds | about 3.5 h on 36 threads |

Conventions: every script runs from the repository root
(`python studies/<name>/<script>.py`), shares `common.py`, writes renders to
its own `output/` and solver runs to its own `results/` (both ignored by
git), and keeps what is versioned in `assets/`, `solves/`, `figures/`,
`renders/` and `golly/`.
