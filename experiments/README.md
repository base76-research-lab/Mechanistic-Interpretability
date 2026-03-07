# Experiments

Den här mappen innehåller körningsartefakter per experiment (JSON/MD) och *ibland* små outputs.

I Base76 `#research`-systemet är detta experimentlagret. `research_index.md` visar vilket state spåret är i och vilka runs som just nu är viktigast.

## Konvention

- Varje experiment ligger i en egen mapp: `exp_###_*`
- Spara alltid:
  - `metrics.json` (om relevant)
  - `top_features.json` / `field_view*.json` / `runs/*.json` (om relevant)
  - `runs/*.md` (kort human-readable log per körning)
- Stora tensorfiler (t.ex. `activations.pt`, `sae_weights.pt`) är build artifacts och ignoreras av git.

## State and routing

- `#research` routes this repo into the global `ai_microscopy` track
- experiment artifacts support state transitions such as `PROTOCOL -> RUN -> ANALYSIS`
- findings and external claims should not originate here directly; they should be promoted to `reports/`

## Nuvarande experimentmappar (high-level)

- `exp_001_sae/`, `exp_001_sae_v2/`, `exp_001_sae_v3/` — SAE + Field View signal (GPT-2 layer 5)
- `exp_002_persona/` — persona/traits-artefakter (se respektive README/log)
- `exp_003_compression_vectorized/` — jämförelse raw vs compressed vs vectorized proxy (mean/attn_weighted/pca1); delta-metrics mot raw; robusthet via `--require-compressor`/`--exclude-invalid-compression`

## Orientation

Börja i `../research_index.md` för att se:

- current state
- latest runs
- latest findings
- claim boundary
