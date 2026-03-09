# Mechanistic Interpretability

This repository contains Base76 Research Lab's mechanistic interpretability work on sparse autoencoders, residual-state analysis, subspace probing, and intervention-based runtime observability.

Its current scientific focus is twofold:

1. to study internal representations and control-relevant structure in small and medium-sized language models
2. to develop a geometry-based reliability signal, `Field View`, for distinguishing reasoning-like and hallucination-prone regimes before output collapse

This repository should be read as part of the Base76 `#research` system and, more specifically, as a working repository within the global `ai_microscopy` track.

## Research context

Within the Base76 research portfolio:

- `#research` is the explicit activation convention for research mode
- this repository belongs to the `ai_microscopy` track
- `research_index.md` is the primary orientation file
- substantive claims should be labeled with explicit evidence levels: `Exploratory`, `Supported`, or `Replicated`
- external communication should not bypass state tracking or claim boundaries
- `hallucinations` currently remains a sub-track within this research line, not a separate repository

At the repository layer:

- GitHub Issues and the GitHub Project act as the operational lab surface
- GitHub Project: `https://github.com/users/base76-research-lab/projects/1/views/1`
- `GITHUB_PROJECT_LAB_SPEC.md` defines the repository's GitHub operating model
- `.github/ISSUE_TEMPLATE/` contains the standard forms for research questions, runs, analyses, and package work

## What this repository studies

The main research objectives are:

1. extracting sparse, interpretable features from residual and MLP activations
2. identifying circuits and control-relevant subspaces through patching and ablation
3. building feature dictionaries with interpretable labels and example behaviors
4. measuring state-candidate misalignment as a precursor to reliability failures

## Start here

If you read only one file first, read `research_index.md`.

Recommended reading order:

- `research_index.md`
- `https://github.com/users/base76-research-lab/projects/1/views/1`
- `reports/MODEL_MICROSCOPY_PLAN_2026-03-07.md`
- `reports/summary_findings_2026-03-06.md`
- `reports/exp_001_sae.md`
- `reports/feature_dict.md`
- `reports/figures/field_view_triage.png`
- `experiments/exp_001_sae_v3/`
- `notebooks/README.md`

## Quickstart

Set up the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a representative SAE experiment:

```bash
python3 scripts/run_sae.py --model gpt2 --layer 5 --prompts data/prompts.txt --out experiments/exp_001_sae_local
```

Run a representative Field View probe:

```bash
python3 scripts/field_view.py --prompt "the opposite of hot is" --model gpt2 --layer 5 --units 472 468 57 156 346 --mode pc2 --topk 8
```

Generate report figures:

```bash
python3 scripts/make_figures.py
```

## Reproducibility and claims

- Large tensors such as `activations.pt` and `sae_weights.pt` are treated as build artifacts and are ignored by git.
- Reviewable outputs such as metrics, JSON artifacts, figures, and findings notes are retained.
- `research_index.md` tracks current state, latest runs, evidence level, claim boundary, and next transition.
- Notebooks are exploratory surfaces; stable conclusions should be promoted into `reports/`.
- Notebooks alone do not satisfy package-readiness or external-claim requirements.

For more detail, see `experiments/README.md`, `reports/README.md`, and `notebooks/README.md`.

## Repository structure

```text
Mechanistic-Interpretability/
├── data/         # prompt sets and small datasets
├── experiments/  # experiment artifacts, JSON outputs, and run summaries
├── notebooks/    # exploratory GPU-backed work (Colab/local)
├── paper/        # internal writing area
├── reports/      # findings, plans, logs, and figures
└── scripts/      # executable tools for SAE, Field View, patching, and analysis
```
