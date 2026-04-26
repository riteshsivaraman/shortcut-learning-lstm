# Notebooks

Exploratory and figure-generation notebooks. Run from the repo root with the venv active:

```bash
jupyter notebook notebooks/plots.ipynb
```

- `plots.ipynb` — generates the three core figures from `results/all_runs.csv`. Falls back to synthetic data if no real results exist yet, so it can be run end-to-end before any experiments complete.
