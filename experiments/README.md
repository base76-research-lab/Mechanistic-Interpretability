# Experiments

This folder contains run artifacts per experiment (JSON/MD) and sometimes small outputs.

## Conventions

 - Keep each experiment in its own folder: `exp_###_*`
 - Always save:
   - `metrics.json` (when relevant)
   - `top_features.json` / `field_view*.json` / `runs/*.json` (when relevant)
   - `runs/*.md` (short human-readable run note)
 - Large tensors (e.g., `activations.pt`, `sae_weights.pt`) are build artifacts and ignored by git.

## Current experiment folders (high-level)

- `exp_001_sae/`, `exp_001_sae_v2/`, `exp_001_sae_v3/` — SAE + Field View signal (GPT-2)
- `exp_002_persona/` — persona/traits artifacts (see the folder outputs/logs)
