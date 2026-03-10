# Trajectory Detection Findings — 2026-03-10

Date: 2026-03-10
Owner: Bjorn / Base76

## Goal

Test whether trajectory geometry separates `hallucination_prone` prompts from anchored, reasoning,
and transition prompts better than entropy alone in the current GPT-2 panel.

## Setup

- Model: `gpt2`
- Layer(s): `3 5 6 9 12`
- Dataset / prompts: `data/prompts_observability_panel_2026-03-07.jsonl`
- Method: read-only oscilloscope with shared SAE subspace plus block-level detection analysis
- Params: SAE `exp_001_sae_v3`, basis `pc2`, units `472 468 57 156 346`
- Artifacts written to:
  - `experiments/exp_005_trajectory_block/readonly_full_metrics_2026-03-10/trace.jsonl`
  - `experiments/exp_005_trajectory_block/analysis_2026-03-10/detection/`
  - control traces: `experiments/exp_004_unified_observability_stack/hallu_panel_baseline_2026-03-10/trace.jsonl`
    and `.../hallu_panel_baseline_recon_2026-03-10/trace.jsonl`

## Results

- Overall geometry detection AUC: `0.821`
- Overall entropy baseline AUC: `0.513`
- Pairwise AUC:
  - hallucination vs anchored: geometry `0.889`, entropy `0.222`
  - hallucination vs reasoning: geometry `0.778`, entropy `0.444`
  - hallucination vs transition: geometry `0.750`, entropy `1.000`
- Hallucination-prone regime medians:
  - geometry score: `0.648`
  - mean gap: `6.148`
  - mean coherence: `0.649`
  - mean degeneracy: `0.920`
  - path length: `14.159`
- Detection verdict from current block: `useful`

## Interpretation

- In the current GPT-2 panel, a geometry-driven detection score separates hallucination-prone
  prompts more strongly than entropy alone at the full-panel level.
- The signal is strongest against anchored and reasoning prompts.
- Transition prompts remain the hardest boundary, which is consistent with the idea that transition
  prompts partially inhabit the same geometric neighborhood as hallucination-prone prompts.

## Threats to validity

- Small panel size: `16` prompts total, `3` hallucination-prone prompts
- The detection score is composite and should be treated as a block-specific scoring rule, not yet
  as a stable detector design
- Transition comparisons remain fragile because entropy still dominates one pairwise boundary

## Claim boundary

Allowed:

- a geometry-driven score is more informative than entropy alone on the current panel-level
  hallucination separation task in GPT-2 Small

Not yet allowed:

- production-ready pre-hallucination detection
- cross-model generalization
- a fixed detector threshold that should transfer outside the current panel

## Next step

Run the same detection analysis on a larger or slightly perturbed prompt panel before promoting the
score from `useful in current panel` to a stronger detector claim.
