# Trajectory Block Protocol

Date: 2026-03-10
Scope: `ESA/research/mechanistic-interpretability/`

## Purpose

Define the standard next-run block for:

1. pre-hallucination detection
2. layer bifurcation
3. regime stability

The block is designed to produce `Supported`-level conclusions inside the current GPT-2 Small setup
without crossing into cross-model claims.

## Observer rule

The block uses a hard observer/intervention split:

- `read-only oscilloscope` = primary observer surface
- `unified baseline without write-back` = compatible control surface
- `reconstruction/write-back` = intervention only

No write-back trace may be used as observational evidence.

## Canonical block defaults

- model: `gpt2`
- panel: `data/prompts_observability_panel_2026-03-07.jsonl`
- layers: `3 5 6 9 12`
- SAE state: `experiments/exp_001_sae_v3/sae_weights.pt`
- basis: `pc2`
- units: `472 468 57 156 346`

## Required run modes

Each evidence-bearing block must include:

- `readonly_observer`
- `unified_baseline`
- `writeback_intervention`

## Required artifacts

Each block must write:

- `trace.jsonl`
- `metadata.json` where supported
- `summary.json` and/or `per_prompt_summary.csv`
- at least one reviewer-readable figure per experiment family
- one findings-note per family
- one synthesis note for the full block

## Execution scripts

- block runner: `scripts/run_trajectory_block.py`
- block analysis: `scripts/analyze_trajectory_block.py`
- read-only trace: `transformer_oscilloscope/`
- unified baseline/intervention: `scripts/run_unified_observability_stack.py`

## Acceptance criteria

### Detection

- compare geometry-driven score against entropy alone on same material
- report regime-wise medians and IQR
- classify result as `useful`, `uncertain`, or `not better than entropy`

### Bifurcation

- compute layer-wise divergence summary
- identify strongest divergence layer or divergence band
- report whether `Layer 6` remains strongest or should be reframed as `L6-L9`

### Regime stability

- report `path length`, `trajectory curvature`, `endpoint variance`, and `within-regime dispersion`
- state whether fingerprint is regime-level or still prompt-sensitive

## Reporting rule

Every findings-note in this block must include:

- Goal
- Setup
- Results
- Interpretation
- Threats to validity
- Claim boundary
- Next step
