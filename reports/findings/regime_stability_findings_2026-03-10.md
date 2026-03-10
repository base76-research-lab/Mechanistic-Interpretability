# Regime Stability Findings — 2026-03-10

Date: 2026-03-10
Owner: Bjorn / Base76

## Goal

Describe each regime as a trajectory form rather than only as an output class, using path and
endpoint stability metrics in shared subspace.

## Setup

- Model: `gpt2`
- Layer(s): `3 5 6 9 12`
- Dataset / prompts: `data/prompts_observability_panel_2026-03-07.jsonl`
- Method: read-only regime stability analysis over shared SAE subspace trajectories
- Params: SAE `exp_001_sae_v3`, basis `pc2`, units `472 468 57 156 346`
- Artifacts written to:
  - `experiments/exp_005_trajectory_block/analysis_2026-03-10/stability/`

## Results

- Fingerprint status: `regime_level_signal`
- Minimum pairwise fingerprint distance: `1.357`
- Regime summaries:
  - anchored: path median `13.207`, curvature median `3.761`, endpoint variance `0.844`
  - reasoning: path median `12.237`, curvature median `2.388`, endpoint variance `0.775`
  - transition: path median `11.263`, curvature median `2.657`, endpoint variance `1.119`
  - hallucination-prone: path median `14.159`, curvature median `4.135`, endpoint variance `1.584`
  - control: path median `13.395`, curvature median `4.181`, endpoint variance `2.698`

## Interpretation

- The current panel supports regime-level trajectory fingerprints rather than only prompt-specific
  behavior.
- Hallucination-prone prompts are characterized by longer paths and higher curvature than anchored,
  reasoning, and transition prompts.
- `Control` prompts remain geometrically noisy, which is useful because it shows that the stability
  space is not simply tracking correctness.

## Threats to validity

- Each regime is represented by a very small number of prompts
- `Control` fingerprints may partly reflect prompt corruption rather than a stable semantic regime
- Fingerprint distinctness is still internal to the current GPT-2 panel

## Claim boundary

Allowed:

- regime stability fingerprints are distinct at regime level in the current GPT-2 setup

Not yet allowed:

- attractor-level claims
- cross-model regime fingerprint transfer
- production-ready runtime classification from these fingerprints alone

## Next step

Test regime fingerprints on a slightly expanded panel before deciding which metrics should become
canonical for runtime monitoring.
