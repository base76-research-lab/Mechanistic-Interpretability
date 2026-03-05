# Mechanistic Interpretability (Base76)

English TL;DR: We build reviewable mechanistic interpretability experiments (SAEs + subspace probes) and a
geometry-based reliability signal ("Field View") that separates reasoning vs hallucination-like regimes.

Language: English (repo-wide).

Goal: map internal circuits, representations, and *subspaces* in small/medium language models.
Focus: polysemanticity, superposition, circuit discovery, and feature dictionaries via Sparse Autoencoders (SAE).

This track also develops a reliability signal ("Field View"): a geometry-based risk score that compares the
residual state (projected into a chosen subspace) against the top-k candidate token directions *before* collapse
via unembedding.

## Read this first

Start here:

- Findings report: `reports/exp_001_sae.md`
- Feature dictionary: `reports/feature_dict.md`
- Run log: `reports/logs/2026-03-04.md`
- Run artifacts (JSON): `experiments/exp_001_sae_v3/`
- Figure: `reports/figures/field_view_triage.png`
- Notebooks: `notebooks/README.md`
- Lab notes: `notes/README.md`

## Goals

1. Train SAEs on residual/MLP latents to extract sparse, interpretable features.
2. Circuit discovery via activation patching / ablation on known phenomena (e.g., induction heads).
3. Feature dictionaries: a catalog of discovered features with examples, labels, and patch effects.
4. Subspace-based risk/triage: separate reasoning-like vs hallucination-like regimes via state–candidate misalignment.

## Quickstart

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run SAE (example):
```bash
python3 scripts/run_sae.py --model gpt2 --layer 5 --prompts data/prompts.txt --out experiments/exp_001_sae_local
```

Run Field View (risk signal):
```bash
python3 scripts/field_view.py --prompt "the opposite of hot is" --model gpt2 --layer 5 --units 472 468 57 156 346 --mode pc2 --topk 8
```

Generate figures used in reports:
```bash
python3 scripts/make_figures.py
```

## Artifacts and reproducibility

- Large tensors (e.g., `activations.pt`, `sae_weights.pt`) are treated as build artifacts and ignored by git.
- Reports and small JSON artifacts (metrics, top_features, field_view runs) are committed to keep findings reviewable.

More details: `experiments/README.md` and `reports/README.md`.

## Layout

```
Mechanistic Interpretability/
├── data/                 # prompts, small datasets
├── experiments/          # experiment runs + JSON artifacts
├── notebooks/            # explorations (Colab/GPU when relevant)
├── reports/              # findings, logs, feature dictionary
└── scripts/              # runnable tools (SAE, field_view, patching)
```
