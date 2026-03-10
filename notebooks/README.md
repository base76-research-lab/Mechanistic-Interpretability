# Notebooks

Notebooks are used for exploration and GPU-backed runs (Colab/local CUDA). The repo keeps notebooks
*reviewable* and *re-runnable*:

- Prefer saving conclusions in [`reports/`](../reports/) (not inside cell outputs).
- Keep outputs minimal; strip heavy outputs before committing.
- Treat large tensors/weights as build artifacts (ignored by `.gitignore`).

## `#research` role

- notebooks are exploratory and support the `ai_microscopy` track
- notebooks do not satisfy package-readiness or external-claim requirements on their own
- claims should be promoted into [`reports/`](../reports/) with explicit evidence level and claim boundary
- use [`../research_index.md`](../research_index.md) as the first orientation file for current state

## Index

- `phi2_sae_fieldview.ipynb` — Phi-2 SAE training + Field View runs (GPU recommended)

## Running locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
jupyter lab
```

## Stripping outputs before commit

```bash
python3 scripts/strip_ipynb_outputs.py notebooks/phi2_sae_fieldview.ipynb
```
