# Report

- `report.pdf`: compiled English report, with LaTeX-style typography and vector scientific figures.
- `report.typ`: editable Typst source.
- `presentation.pdf`: supplied presentation used to align the report's narrative.
- `build_report.py`: extracts saved numerical notebook outputs, checks the comparison signs and binary loss, draws figures, and compiles the report without retraining models.
- `assets/`: generated SVG figures and `report_data.json`, including source notebook hashes and the numerical statistics used in the report.

From the repository root:

```powershell
uv sync --locked
uv run --locked python docs/build_report.py
```

The six notebooks must contain their executed Plotly outputs. Optional `results/` caches are not required to build the report. To regenerate training results, follow the notebook execution commands in the root README; then rebuild the report. The prose describes the current experiment snapshot, so review it if the experimental settings or conclusions change.

Only rebuild figures and the data snapshot:

```powershell
uv run --locked python docs/build_report.py --figures-only
```
