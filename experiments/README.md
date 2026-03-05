# Experiments

Den här mappen innehåller körningsartefakter per experiment (JSON/MD) och *ibland* små outputs.

## Konvention

- Varje experiment ligger i en egen mapp: `exp_###_*`
- Spara alltid:
  - `metrics.json` (om relevant)
  - `top_features.json` / `field_view*.json` / `runs/*.json` (om relevant)
  - `runs/*.md` (kort human-readable log per körning)
- Stora tensorfiler (t.ex. `activations.pt`, `sae_weights.pt`) är build artifacts och ignoreras av git.

## Nuvarande experimentmappar (high-level)

- `exp_001_sae/`, `exp_001_sae_v2/`, `exp_001_sae_v3/` — SAE + Field View signal (GPT-2)
- `exp_002_persona/` — persona/traits-artefakter (se respektive README/log)

