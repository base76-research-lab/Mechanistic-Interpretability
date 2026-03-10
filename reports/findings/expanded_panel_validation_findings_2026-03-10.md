# Expanded Panel Validation Findings — 2026-03-10

Date: 2026-03-10
Owner: Bjorn / Base76

## Goal

Test whether the current GPT-2 trajectory picture survives a harder observability panel with more
anchored, reasoning, transition, hallucination-prone, and matched control prompts.

## Setup

- Model: `gpt2`
- Layer(s): `5 6 7 8 9 10 11 12`
- Dataset / prompts: `data/prompts_observability_panel_expanded_2026-03-10.jsonl`
- Method: read-only oscilloscope dense trajectory block with baseline and write-back control
- Params: SAE `exp_001_sae_v3`, basis `pc2`, units `472 468 57 156 346`
- Artifacts written to:
  - `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/block_manifest.json`
  - `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/detection/summary.json`
  - `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/bifurcation/summary.json`
  - `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/stability/summary.json`
  - `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/synthesis/summary.json`

## Results

- Detection:
  - geometry AUC: `0.603`
  - entropy AUC: `0.423`
  - verdict: `useful`
  - pairwise AUC:
    - hallucination vs anchored: geometry `0.889`, entropy `0.111`
    - hallucination vs reasoning: geometry `0.528`, entropy `0.528`
    - hallucination vs transition: geometry `0.479`, entropy `0.771`
- Bifurcation:
  - strongest local divergence layer: `6`
  - top layers: `6`, `8`, `12`, `9`
  - largest hallucination expansion step: `10 -> 11` with mean step distance `4.366`
- Stability:
  - fingerprint status: `regime_level_signal`
  - minimum pairwise fingerprint distance: `2.594`

## Interpretation

- The expanded panel preserves the main onset picture:
  - `Layer 6` remains the strongest current local onset candidate
  - `10 -> 11` remains the strongest later expansion step
- Geometry still beats entropy at full-panel level, but less strongly than in the smaller panels.
- The weak boundary is now clearer:
  - geometry does not beat entropy against `transition`
  - geometry only ties entropy against `reasoning`
- This means the current method is strongest as onset-localization and regime-structure work, not
  yet as a broadly clean detector.

## Threats to validity

- The panel is larger, but still small in absolute model-evaluation terms
- Transition prompts remain mixed and partially regime-adjacent by design
- The current signal remains internal to GPT-2 Small and one metric bundle

## Claim boundary

Allowed:

- the expanded GPT-2 panel preserves the current onset picture: `Layer 6` onset and `10 -> 11`
  later expansion
- geometry remains more informative than entropy at full-panel level
- the current ambiguity boundary is concentrated around transition and regime-adjacent prompts

Not yet allowed:

- production-ready pre-hallucination detection
- a claim that geometry cleanly dominates entropy across all regime boundaries
- cross-model onset localization claims

## Next step

Use the completed lead-time slice to drive an explicit transition counter-case pass before deciding
whether GPT-2 validation is strong enough to justify one control-model block.
