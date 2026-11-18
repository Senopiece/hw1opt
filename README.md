# RobustCE experiments

Python 3.12, PyTorch on CPU, and dependencies managed with uv.

Install the locked environment from the project root:

```powershell
uv sync --locked
```

Open a notebook and select the project's `.venv` Python interpreter as its kernel. All notebook text is in English; all figures use Plotly.

- `notebooks/01_loss_check.ipynb`: compare analytic binary RobustCE against numerical maximization over feasible targets.
- `notebooks/02_experiments.ipynb`: compare five fixed variants on both datasets: CE, RobustCE(0.05), RobustCE(0.1), RobustCE(0.2), and Smoothing(0.2). Smoothing uses `epsilon=0.2` to match the positive-margin penalty of RobustCE with `delta=0.1` for critical analysis (d); the losses differ on negative margins. Tune only learning rate at training-label corruption `eta=0.1`, separately for each variant, then freeze it and compare `eta=0` against `eta=0.1` on 5 new evaluation seeds. Uses 3 learning-rate candidates and 3 tuning seeds; validation and test labels stay clean. All three RobustCE radii appear in the same metric, learning-curve, confidence, and probability-map figures. The common robust evaluation radius stays `delta_eval=0.1` for all variants.

Execute the notebook and save its outputs from the command line:

```powershell
uv run --locked jupyter execute notebooks/01_loss_check.ipynb --inplace
uv run --locked jupyter execute notebooks/02_experiments.ipynb --inplace --timeout=1800
```

The second notebook runs independently and performs 90 tuning runs and 100 evaluation runs on CPU. It saves metrics, selected settings, the protocol, representative models, and paired confidence changes for every pair of RobustCE radii under `results/02_fixed_losses/`. Detailed records are saved as CSV/JSON files; no tables are displayed in the notebooks.
