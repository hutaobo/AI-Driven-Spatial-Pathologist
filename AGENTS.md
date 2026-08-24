# Agentic Spatial Pathologist (`spatho`)

## Cursor Cloud specific instructions

This section captures durable, non-obvious context for working in this repo inside a
Cloud Agent VM. The startup update script already installs all Python dependencies, so
you should not need to reinstall anything by hand.

### What lives here

- `src/spatho`: the maintained, public-facing Python package + `spatho` CLI. This is the
  primary product surface.
- `main.py`: a legacy Gradio/SciLifeLab-Serve web app ("Agentic Spatial Pathologist")
  kept for compatibility. It turns Xenium `cells.parquet` + `clusters.csv` into a
  cophenetic dendrogram, candidate spatial structures, and HistoSeg contour maps.
- `src/pathology_ai_service`: an optional self-hosted HTTP service. Its full runtime
  depends on heavy external services (vLLM, TEI embed/rerank, Qdrant) that are **not**
  provisioned here; only its unit tests run out of the box.

### Critical dependency gotcha: pin `histoseg==0.1.9.2`

`main.py` imports `from histoseg import Pattern1IsolineConfig` and from
`histoseg.contours.*` / `histoseg.sfplot.*`. The latest published `histoseg` (0.1.9.3)
**removed the top-level export and renamed `histoseg.contours` -> `histoseg.contour`**,
which breaks the Gradio app (it degrades to a "HistoSeg could not be imported" error at
run time). `pyproject.toml` only requires `histoseg>=0.1.9.2`, so a plain install resolves
to the broken 0.1.9.3.

The update script therefore force-pins `histoseg==0.1.9.2`, which is compatible with
**both** the `spatho` test suite and the legacy Gradio app. If you ever reinstall deps
manually, re-pin it: `python3 -m pip install --break-system-packages "histoseg==0.1.9.2"`.

### Running things

- Python is the system `python3` (3.12). Packages install with `--break-system-packages`
  (PEP 668 externally-managed); there is intentionally no virtualenv.
- Console scripts (`spatho`, `histoseg*`, `gradio`, `pytest`, ...) install to
  `~/.local/bin`, which is **not** on `PATH` by default. Either run `python3 -m pytest`
  etc., or prepend it: `export PATH="$HOME/.local/bin:$PATH"`.
- Tests (mirrors CI): `python3 -m pytest` (83 tests, ~3s).
- Build check (mirrors CI): `python3 -m build`. Note `python3 -m twine check dist/*`
  currently fails on a Metadata-2.4 `license-expression`/`license-file` field mismatch
  between setuptools and twine — this is a packaging/tooling version issue, not a repo
  bug, and only affects the publish step.
- CLI hello-world: `spatho list-organ-packs`, `spatho config-schema --output out.json`,
  `spatho init-workflow --organ breast --case-name c1 --dataset-root <dir> \
  --base-pipeline-config <cfg.json> --output wf.json`, then `spatho doctor --config wf.json`.
- Gradio app: `python3 main.py` serves on port `7860` (set `GRADIO_SERVER_NAME=0.0.0.0`
  to expose it). It needs `cells.parquet` (with `x_centroid`/`y_centroid` + a barcode/id
  column) and `clusters.csv` (columns `Barcode`,`Cluster`) as inputs. Runtime output is
  written under `./project-vol/` (git-ignored). The full flow is: upload files -> "Build
  dendrogram" -> select structures -> "Run multi-structure contour analysis".
