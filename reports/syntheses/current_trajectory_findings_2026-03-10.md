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
- The expanded panel rerun preserved the `Layer 6` onset plus `10 -> 11` expansion picture, while
  making the transition boundary look more ambiguous than the smaller panel suggested.
- Regime stability fingerprints remain distinct at regime level after the dense sweep.
- Token-level lead-time is now measurable in a conservative slice before prompt end, but broad
  thresholding still leaks into transition-adjacent prompts.
- The transition counter-case pass suggests that the current transition detections differ from the
  hallucination slice mainly by lower post-onset persistence, and disappear under conservative
  thresholding.
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
- transition and regime-adjacent prompts remain the main ambiguity boundary in the expanded panel
- a conservative token-level lead-time slice exists for some hallucination-prone prompts before
  prompt end, but coverage remains low
- current transition counter-cases differ mainly by lower post-onset persistence rather than a
  cleaner onset score

Not yet allowed:

- cross-model generalization
- production-ready early warning
- a clean detector claim against transition prompts
- a broad token-level early-warning claim
- a universal `commit layer`
- attractor-level claims

## Reading map

Read these together:

1. `reports/findings/trajectory_detection_findings_2026-03-10.md`
2. `reports/findings/layer_bifurcation_findings_2026-03-10.md`
3. `reports/findings/regime_stability_findings_2026-03-10.md`
4. `reports/findings/dense_layer_sweep_findings_2026-03-10.md`
5. `reports/findings/token_level_lead_time_findings_2026-03-10.md`
6. `reports/findings/transition_countercase_findings_2026-03-10.md`

Primary synthesis artifacts:

- `reports/syntheses/trajectory_block_synthesis_2026-03-10.md`
- `reports/syntheses/current_trajectory_findings_2026-03-10.md`

## Next validation block

1. write the Friday internal review pack
2. decide whether one small control model is justified
3. keep cross-model work blocked unless Friday review says `go`
