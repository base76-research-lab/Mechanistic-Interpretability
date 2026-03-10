# Current Trajectory Findings — 2026-03-10

Date: 2026-03-10
Evidence level: `Supported` in the active GPT-2 Small setup
Scope: current synthesis across the first trajectory block and the dense `5-12` follow-up sweep

## What is currently supported

- A geometry-driven detection score separates `hallucination_prone` prompts better than entropy at
  panel level in the current GPT-2 setup.
- `Layer 6` remains the strongest current local onset candidate for hallucination-prone
  divergence.
- A denser `5-12` sweep separates onset from later acceleration: earliest strongest divergence at
  `Layer 6`, largest later hallucination expansion at `10 -> 11`.
- Regime stability fingerprints remain distinct at regime level after the dense sweep.
- Read-only oscilloscope traces and write-back runs must be treated as different evidence classes;
  write-back is intervention, not neutral observation.

## Why this matters

The current findings shift the work from token-error description toward runtime geometry and regime
dynamics:

- hallucination-prone behavior is not best described only as high entropy
- the trajectory appears to change form before output collapse
- onset and later expansion are not the same event and should not be collapsed into one layer claim

This makes the work more useful as observability research than as a narrow output-analysis study.

## Claim boundary

Allowed now:

- geometry is more informative than entropy on the current panel-level separation task
- `Layer 6` is the strongest current local divergence onset candidate in GPT-2 Small
- `10 -> 11` is the strongest current later expansion step in the dense sweep
- regime fingerprints are stable enough to count as regime-level signal in the current setup

Not yet allowed:

- cross-model generalization
- production-ready early warning
- a universal `commit layer`
- attractor-level claims

## Reading map

Read these together:

1. `reports/trajectory_detection_findings_2026-03-10.md`
2. `reports/layer_bifurcation_findings_2026-03-10.md`
3. `reports/regime_stability_findings_2026-03-10.md`
4. `reports/dense_layer_sweep_findings_2026-03-10.md`

Primary synthesis artifacts:

- `reports/trajectory_block_synthesis_2026-03-10.md`
- `reports/current_trajectory_findings_2026-03-10.md`

## Next validation block

1. expand the prompt panel
2. rerun the dense `5-12` trajectory block
3. add token-level lead-time analysis before strengthening any early-warning framing
