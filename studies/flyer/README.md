# PhD-defence flyer

Backgrounds for the flyer of a joint UGent / Universidade de São Paulo PhD
defence (September 2026): portrait, 1080x1536 px, a free-form still life on
a frame of pond tiles, with the middle kept light for the text (added
afterwards in Canva). Two designs were delivered, each in four palettes, and
every render is one verified still life:

| Design | Canvas | What |
|---|---|---|
| `vignette2` | 360x512 cells, 3 px/cell | a rounded window in a tile frame |
| `skyline2` | 540x768 cells, 2 px/cell | Ghent (left) and São Paulo (right) under a tile ribbon |

```bash
python studies/flyer/flyer.py render                  # renders/ and golly/, seconds
python studies/flyer/flyer.py palettes skyline2       # every palette, into output/
python studies/flyer/flyer.py preview                 # the base designs, no solver
python studies/flyer/flyer.py solve skyline2 --strip-procs 2 --polish-procs 3   # ~30 min
python studies/flyer/flyer.py polish skyline2 --seams-only --polish-procs 3
```

| Folder | Versioned | Contents |
|---|---|---|
| `solves/` | yes | the delivered patterns, bit-packed (`gol_mosaics.freeform.io`) |
| `renders/` | yes | the eight delivered flyers; `palettes/` has the quiet palette set on the skyline |
| `golly/` | yes | each flyer merged with its tile field, as a Golly `.cells` file |
| `results/` | no | solver runs; when present, `render` uses them instead of `solves/` |
| `output/` | no | previews, contact and palette sheets, earlier rounds, dropped designs |

The design history (four review rounds, the seam and dotted-line fixes, the
palettes) is in [REPORT.md section 9](../../experiments/beyond_tiles/REPORT.md).
On a 16 GB Windows machine run the solve with `--strip-procs 2 --polish-procs 3`:
six parallel strips exhaust the commit charge and crash CP-SAT workers.

`solve` now runs the library pipeline (`gol_mosaics.freeform.solve_poster`,
seam rounds included) and `polish --seams-only` the library's seam
sub-targets. The delivered solves predate both, so a fresh solve is a new
still life, not the same one.
