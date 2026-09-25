# Contributing

Contributions are welcome: bug reports, ideas (see the [roadmap](docs/roadmap.md)),
code and documentation. For anything larger than a fix, open an issue first so
we can agree on the approach.

## Setting up

```bash
git clone https://github.com/mrollier/game-of-life-mosaics.git
cd game-of-life-mosaics
pip install -e ".[dev,beyond]"        # tests, notebooks, the free-form solver
nbstripout --install --attributes .gitattributes
```

Python 3.10 or newer. The last line makes git strip the outputs of the
research notebooks on commit (see below). Where things live:
[docs/repo_map.md](docs/repo_map.md).

## Tests

```bash
pytest                                  # everything, about 4 minutes
pytest --ignore=tests/freeform --ignore=tests/experiments   # the tile library, about 1 minute
```

The suite includes checks that keep the repository honest:

- `tests/test_goldens.py` pins digests of deterministic output: the CNF
  fingerprints, every shipped tile database, all 256 ECA rules, seeded
  mosaics, rendering, export and the free-form objective. A change that is
  meant to alter output re-pins the digest there
  (`python -m tests.test_goldens` prints the current values) and says why in
  the commit message and CHANGELOG.
- `tests/test_public_api.py` pins the exports and fails on any 2.x name.
- `tests/test_notebook_imports.py` checks that every name a notebook imports
  exists; `tests/test_repo_hygiene.py` that research notebooks have no saved
  outputs and no notebook holds a machine-specific path; `tests/test_docs.py`
  that every relative link in the Markdown files resolves.

Checks for the search pipeline, which need `[sat]` and a minute or two:

```bash
python -m gol_mosaics.sat_search bruteforce --level 4
(cd search/tiles && python search.py validate)
python tools/pack_tiles.py check
```

## Conventions

- **Vocabulary.** Pond, tile, mosaic, layout, level, as in the
  [glossary](docs/glossary.md). A 0/1 array of Game of Life cells is `cells`.
- **British English** in prose, comments and identifiers (`colour`,
  `neighbour`, `normalise`); third-party names keep their own spelling
  (`gr.ColorPicker`, PIL's `color=`).
- **Style.** Match the surrounding code: docstrings with Args/Returns/Raises,
  comments that say *why*. Keep the core dependencies at numpy, scipy and
  Pillow; anything else is an optional extra, imported where it is used.
- **Commits.** One logical change per commit, with a message that says what
  changed and why. Moves are separate commits from edits, so history follows
  the files.
- **Notebooks.** Tutorials (`notebooks/tutorials/`) keep their outputs, so
  GitHub shows them; re-run them top to bottom before committing. Research
  notebooks (`notebooks/research/`) are committed without outputs. Every
  notebook finds the repository root itself; never use `../` paths.
- **Data files.** Tile databases are written with `tools/pack_tiles.py`,
  never by hand, and their hashes go into [docs/reproduce.md](docs/reproduce.md).
- **Large results** (solver runs, level-7 data, certificates) stay out of git:
  `results/`, `output/` folders of the studies and `search/*/work*` are
  ignored.

## Releasing

1. Update `__version__` in `src/gol_mosaics/__init__.py` (the single source)
   and add a section to [CHANGELOG.md](CHANGELOG.md).
2. `pytest`, then `python -m build` and check the wheel contains
   `gol_mosaics/data/*.npy` and `gol_mosaics/freeform/`.
3. Deploy the web app with `app/deploy.sh` ([docs/deploy.md](docs/deploy.md)).
