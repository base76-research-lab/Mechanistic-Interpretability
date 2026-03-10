# Experiments

This directory contains experiment-specific run artifacts, primarily JSON and Markdown outputs, and occasionally small derived files.

Within the Base76 `#research` system, this is the experimental evidence layer. [`research_index.md`](../research_index.md) indicates the current state of the track and which runs matter most at the moment.

## Conventions

- Each experiment should live in its own directory: `exp_###_*`
- Always preserve:
  - `metrics.json` where relevant
  - `top_features.json`, `field_view*.json`, or `runs/*.json` where relevant
  - `runs/*.md` as a short human-readable run log
- Large tensors such as `activations.pt` and `sae_weights.pt` are treated as build artifacts and ignored by git.

## State and routing

- `#research` routes this repo into the global `ai_microscopy` track
- experiment artifacts support state transitions such as `PROTOCOL -> RUN -> ANALYSIS`
- findings and external claims should not originate here directly; they should be promoted to [`reports/`](../reports/)

## Current experiment groups

- `exp_001_sae/`, `exp_001_sae_v2/`, `exp_001_sae_v3/` — SAE + Field View on GPT-2 layer 5
- `exp_002_persona/` — persona and traits artifacts
- `exp_003_compression_vectorized/` — comparison of raw, compressed, and vectorized proxy variants (`mean`, `attn_weighted`, `pca1`), including delta metrics against raw and robustness guards such as `--require-compressor` and `--exclude-invalid-compression`

## Orientation

Start with [`../research_index.md`](../research_index.md) to see:

- current state
- latest runs
- latest findings
- claim boundary
