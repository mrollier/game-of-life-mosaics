# The web app and its deployment

The Gradio app in `app/` turns an uploaded portrait into a mosaic in the
browser. It runs publicly as a Hugging Face Space:
[huggingface.co/spaces/mrollier/game-of-life-mosaics](https://huggingface.co/spaces/mrollier/game-of-life-mosaics).

## Running it locally

```bash
pip install -e ".[app]"
python app/app.py          # then open http://127.0.0.1:7860
```

The first background removal downloads the 176 MB u2net model; later ones are
fast. `gradio app/app.py` runs it with hot reload.

## What is in `app/`

| File | |
|---|---|
| `app.py` | the interface and its handlers; importing it does no work, `main()` warms the tile libraries in a thread and launches |
| `requirements.txt` | the Space's runtime dependencies (the library's own plus gradio and rembg) |
| `README_space.md` | the Space's landing page; its YAML front matter configures the Space (SDK version, entry file) |
| `deploy.sh` | publishes the app |

How it behaves: background removal runs once per upload, in the upload
handler, and both the original and the background-free copy are kept in the
session so the "Remove background" toggle and every slider re-render without
running rembg again. Every render is seeded by a per-session seed ("New
variation" rerolls it), so a look is stable while you adjust other settings.
The last render's cells are kept, so "Download .cells" does not render again.

## Deploying

```bash
hf auth login                     # once: a token with write permission
app/deploy.sh --dry-run           # stage and list what would be uploaded
app/deploy.sh "Describe the change"
```

`deploy.sh`

1. refuses to run when `src/` or `app/` have uncommitted changes, so what goes
   live is always a commit;
2. runs the library and app tests (`--skip-tests` to skip);
3. stages `app.py`, `requirements.txt`, `README_space.md` (as `README.md`) and
   the `gol_mosaics` package in a temporary folder, leaving out the
   free-form solver, which needs OR-Tools the Space does not install;
4. uploads the folder with `hf upload`; `.npy` files go up through Git LFS
   automatically.

The Space rebuilds on every upload. Files deleted locally are not deleted on
the Space; remove them in the Space's file browser if needed.

## Keeping the Space in step

- The Space imports the package from the uploaded copy, not from an install,
  so `gol_mosaics.__version__` is a literal in `__init__.py`.
- When the library's dependencies change, update `app/requirements.txt` to
  match the `[project.dependencies]` and `[app]` extra in `pyproject.toml`.
- The front matter's `sdk_version` pins Gradio on the Space; keep it within
  the range in `requirements.txt`.
