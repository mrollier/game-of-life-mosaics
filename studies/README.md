# Studies

Finished pieces made with the library, each with the script that made it,
the solved patterns it needs (bit-packed, small enough to version) and its
deliverables. They double as worked examples of `gol_mosaics.freeform` and
`gol_mosaics.compose` on real commissions; the research behind them is in
[`experiments/beyond_tiles/REPORT.md`](../experiments/beyond_tiles/REPORT.md).

| Study | What | Re-render | Re-solve |
|---|---|---|---|
| [`flyer/`](flyer/README.md) | PhD-defence flyer, Ghent and São Paulo skylines and a vignette, 1080x1536 px, one verified still life each | seconds, from `solves/` | about 30 min per design |
| [`linkedin/`](linkedin/README.md) | 1584x396 banners from two mountain photos, and the halo-filling and fit-rule figures of REPORT.md sections 6-8 | about 40 s, no solver | `poster.py`, 800x200 |
| [`lam_gods/`](lam_gods/README.md) | The Adoration of the Mystic Lamb as one 2480x1656 still life, and the seam study of REPORT.md C9 | seconds | about 3.5 h on 36 threads |

Conventions: every script runs from the repository root
(`python studies/<name>/<script>.py`), shares `common.py`, writes renders to
its own `output/` and solver runs to its own `results/` (both ignored by
git), and keeps what is versioned in `assets/`, `solves/`, `figures/`,
`renders/` and `golly/`.
