# The Adoration of the Mystic Lamb

The Ghent Altarpiece (`input/images/lam-gods-classic.png`) as a single
2480x1656 Game of Life still life, one cell per pixel: 808,796 live cells,
verified bounded and toroidal. Solved on a 36-thread workstation in 2 h 50 min
of blocks and polish plus 41 min of seam rounds (REPORT.md C9):

```bash
python experiments/beyond_tiles/poster.py input/images/lam-gods-classic.png \
    --width 2480 --height 1656 --out ../../studies/lam_gods/results \
    --keep-background --dither fs --strip-rows 64 --block-cols 416 \
    --strip-procs 14 --polish-procs 28 --seam-rounds 3
python studies/lam_gods/lam_gods_seams.py     # the before/after seam study
```

| Path | Contents |
|---|---|
| `assets/lam_gods_2480x1656_pipeline.npz` | v2, the delivered pattern (bit-packed, 369 kB) |
| `assets/lam_gods_2480x1656_v1_before_seams.npz` | v1, before the seam rounds |
| `golly/lam-gods-2480x1656-v2.cells` | v2 for Golly (4.1 MB) |
| `figures/` | the full render and the seam study figure cited in REPORT.md |
| `results/`, `output/` | a local re-solve and the study's JSON table (not versioned) |

The seam study compares v1 and v2: the separators of the block
decomposition go from 0.65 of their neighbours' density (worst 0.53) to 1.00
(worst 0.93) for two cells of objective (469 to 471).
