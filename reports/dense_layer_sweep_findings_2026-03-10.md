# Dense Layer Sweep Findings — 2026-03-10

Date: 2026-03-10
Owner: Bjorn / Base76

## Goal

Resolve whether the current GPT-2 trajectory picture is best described as:

- an early local bifurcation around `Layer 6`
- a broader `L6-L9` transition zone
- or a later expansion event that was previously undersampled

The dense sweep also tests whether `phase velocity` adds useful observer-side signal without
collapsing the observer/intervention boundary.

## Setup

- Model: `gpt2`
- Layer(s): `5 6 7 8 9 10 11 12`
- Dataset / prompts: `data/prompts_observability_panel_2026-03-07.jsonl`
- Method: read-only oscilloscope trace plus trajectory-block analysis
- Params: SAE `exp_001_sae_v3`, basis `pc2`, units `472 468 57 156 346`
- Artifacts written to:
  - `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/dense_layers_5_12_2026-03-10_readonly/trace.jsonl`
  - `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/detection/`
  - `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/bifurcation/`
  - `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/stability/`
  - control traces:
    - `experiments/exp_004_unified_observability_stack/dense_layers_5_12_2026-03-10_baseline/trace.jsonl`
    - `experiments/exp_004_unified_observability_stack/dense_layers_5_12_2026-03-10_writeback/trace.jsonl`

## Results

- Detection:
  - geometry AUC: `0.692`
  - entropy AUC: `0.462`
  - verdict: `useful`
  - pairwise AUC:
    - hallucination vs anchored: geometry `1.000`, entropy `0.222`
    - hallucination vs reasoning: geometry `0.667`, entropy `0.444`
    - hallucination vs transition: geometry `0.500`, entropy `0.750`
- Bifurcation:
  - strongest local divergence layer: `6`
  - top composite divergence layers: `6`, `8`, `12`
  - largest hallucination expansion step: `10 -> 11` with mean step distance `4.597`
- Stability:
  - fingerprint status: `regime_level_signal`
  - minimum pairwise fingerprint distance: `2.032`
  - hallucination-prone medians:
    - path length: `15.243`
    - mean phase velocity: `2.178`
    - step-distance std: `1.383`
    - curvature: `5.524`

## Interpretation

- The dense sweep sharpens the earlier picture rather than overturning it.
- `Layer 6` still looks like the strongest current local onset candidate for divergence.
- The most forceful expansion in hallucination-prone trajectories occurs later, at `10 -> 11`.
  This matters because it separates `onset` from `late expansion`; they should not be treated as
  the same event.
- Geometry still beats entropy at panel level, but less decisively than in the sparser first block.
  The transition boundary remains the main weakness.
- `Phase velocity` is useful as a descriptive observer metric, but the dense sweep does not support
  treating it as a standalone early-warning signal.

## Threats to validity

- Panel size is still small: `16` prompts total, `3` hallucination-prone prompts
- The transition stratum remains geometrically mixed and partially overlaps the hallucination-prone
  region
- The current sweep is denser in layer-space but still not a cross-model test
- Some high-score non-hallucination prompts remain regime-adjacent rather than clean negatives

## Claim boundary

Allowed:

- in the current GPT-2 panel, the best current picture is `early local bifurcation at Layer 6`
  plus `later expansion around 10 -> 11`
- geometry remains more informative than entropy at panel level in the dense sweep
- regime fingerprints remain distinct at regime level after the denser sweep

Not yet allowed:

- a stable production detector
- a model-general claim that `10 -> 11` is the universal commit point
- a claim that `phase velocity` alone reliably predicts hallucination onset

## Next step

Expand the prompt panel and rerun the dense sweep so that:

1. the `Layer 6 onset` claim is tested against more prompt variety
2. the `10 -> 11 expansion` finding is checked for stability
3. token-level lead time can be measured before any early-warning claim is strengthened
