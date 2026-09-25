# Changelog

## 3.0.0 (2026-09)

A reorganisation for newcomers, a clean API, and the free-form solver in the
package. 3.0 renames the public API without compatibility aliases; the full
2.x to 3.0 table is in [docs/api.md](docs/api.md#from-2x-to-30).

### Changed (breaking)

- One vocabulary: pond, tile, mosaic, layout. `PatternLibrary` is
  `TileLibrary` (`load(level, layout=)`, `.tiles`, `tiles_for_values`, ...),
  `MosaicGenerator(tile_shape=, random_patterns=)` is
  `MosaicGenerator(layout=, random_tiles=)`, `shape=` is `layout=` throughout
  `compose`, `life_safe_pattern` is `merge_background`, and the renderer's
  `render_full_mosaic` / `render_eca_overlay` are `render` / `render_backdrop`.
- British spelling in identifiers: `gol_mosaics.colours`, `ColourScheme`,
  `colours=`, `alpha_colour=`, `fill_colour=`, `neighbours`.
- The pond geometry moves to `tile_domain` (`POND`, `pond_lattice`,
  `pond_frame`, `free_octant`); the three pack/unpack function pairs are
  `Domain.pack` / `Domain.unpack`.
- Tile data files are all packed orbit bits named
  `tiles_{diamond,square}_level_N_orbits.npy` (the package data drops from
  21.4 MB to 2.7 MB); tiles load as uint8 at every level.
- The Gurobi generator is `gol_mosaics.legacy_ilp.generate_tiles`, imported
  only when used; `import gol_mosaics` no longer touches gurobipy.
- Python 3.10 or newer. cellpylib and matplotlib are no longer dependencies.
- A seeded mosaic with `random_tiles=True` differs from 2.x (see Fixed).
  Deterministic selection and every other seeded output are unchanged.
- `.cells` and RLE files are written with `\n` line endings on every platform.

### Added

- `gol_mosaics.freeform`: the free-form CP-SAT still-life solver, promoted
  from `experiments/beyond_tiles` (behind the `[beyond]` extra), with
  `solve_poster()`, the strips / polish / seams / diagonal-repair pipeline.
- `compose`: `centred=` lattice placement, `compose(field=)` for rendering a
  precomputed field, `centring_pad()`.
- `TileLibrary.from_tiles()`, `TileLibrary.pond`, and `scheme` for both
  layouts.
- `tools/pack_tiles.py` to write and check the shipped databases.
- Documentation in `docs/` (usage, API, glossary, tiles, free-form, pipeline,
  reproduction, deployment, roadmap, pixel-perfect viewing), a README for
  newcomers, `CONTRIBUTING.md`, and two new tutorials (`04_golly_export`
  rewritten, `05_how_tiles_work`).
- Tests: golden digests of deterministic output, the public API, notebook
  imports, repository hygiene, documentation links.

### Fixed

- Tile selection at level 6 built a matrix of every tile against every
  library entry (about 14 GB for a 60-tile grid); it now searches the sorted
  densities.
- The bottom-left glider of `GollyExporter` flew out of the grid instead of
  into it.
- The web app's Warhol palette ignored the session seed.
- `run_experiment.py lns` imported the Unix-only `resource` module.
- The no-symmetry search renamed its checkpoints with `Path.rename`, which
  fails on Windows when the target exists; all search runners now write
  through one atomic helper that retries while another process holds the
  file.
- `requires-python` claimed 3.8 support that the code did not have.

### Faster

- The ECA background is stepped with numpy (10x), `.cells` export is
  vectorised (8x), tile assembly uses a reshape instead of `np.block`, and
  the renderer uses a colour lookup table.
- The LNS polisher rescores only a patch's own windows (1.4 s to 0.65 ms per
  candidate on a 2480x1656 canvas) and keeps one process pool per call.
- The web app no longer loads the tile libraries at import, and the
  `.cells` download reuses the last render.

### Moved

- `app.py`, its requirements, the Space README and `deploy.sh` to `app/`;
  `deploy.sh` refuses uncommitted changes, runs the tests and has `--dry-run`.
- `workstation/` to `search/tiles` and `search/nosym`; the DRAT certificates
  (161 MB) are no longer versioned (hashes in `search/tiles/README.md`).
- The flyer, LinkedIn and Lam Gods studies to `studies/`, each with its
  assets; the flyer keeps its delivered solves bit-packed and drops its
  runtime patching of the LNS module.
- Notebooks into `notebooks/tutorials` and `notebooks/research`; research
  notebooks are versioned without outputs.
- The parallel-tempering annealer to `experiments/archive`.

## 2.3.0 and earlier

See the git history.
