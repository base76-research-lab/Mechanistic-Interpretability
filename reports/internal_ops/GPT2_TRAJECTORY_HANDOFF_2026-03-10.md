# GPT-2 Trajectory Handoff — 2026-03-10

Date: 2026-03-10
Status: active internal handoff
Audience: internal only

## Current position

The active microscopy program has shifted from mixed exploration toward a narrower GPT-2 trajectory
validation program.

Current strongest internal position:

- geometry-based observer metrics are more informative than entropy on the active GPT-2 panel
- `Layer 6` is the strongest current local onset candidate
- `10 -> 11` is the strongest current later expansion step in the dense sweep
- read-only traces and write-back traces must be treated as distinct evidence classes

This is a `Supported in current GPT-2 setup` position, not a cross-model or production claim.

## Runs completed so far

### 1. Unified observability and intervention stack

Key artifacts:

- `experiments/exp_004_unified_observability_stack/hallu_panel_baseline_2026-03-10/`
- `experiments/exp_004_unified_observability_stack/hallu_panel_baseline_recon_2026-03-10/`
- `experiments/exp_004_unified_observability_stack/hallu_panel_lsae_no_recon_2026-03-10/`
- `experiments/exp_004_unified_observability_stack/hallu_panel_lsae_r_2026-03-10/`

Takeaway:

- reconstruction / write-back behaves as intervention, not neutral observation
- no-write-back remains the compatible control surface

### 2. Read-only oscilloscope and shared-subspace trajectory comparison

Key artifacts:

- `experiments/exp_004_unified_observability_stack/hallu_panel_readonly_oscilloscope_subspace_2026-03-10/`
- `experiments/exp_004_unified_observability_stack/trajectory_compare_readonly_vs_baseline_2026-03-10/summary.json`
- `experiments/exp_004_unified_observability_stack/trajectory_compare_readonly_vs_recon_2026-03-10/summary.json`

Takeaway:

- read-only and unified baseline are near-identical in shared SAE subspace
- write-back produces strong trajectory distortion

### 3. First trajectory block

Key artifacts:

- `experiments/exp_005_trajectory_block/readonly_full_metrics_2026-03-10/trace.jsonl`
- `experiments/exp_005_trajectory_block/analysis_2026-03-10/detection/summary.json`
- `experiments/exp_005_trajectory_block/analysis_2026-03-10/bifurcation/summary.json`
- `experiments/exp_005_trajectory_block/analysis_2026-03-10/stability/summary.json`

Takeaway:

- geometry beat entropy at panel level
- `Layer 6` emerged as strongest local bifurcation candidate
- regime fingerprints were distinct at regime level

### 4. Dense `5-12` follow-up sweep

Key artifacts:

- `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/dense_layers_5_12_2026-03-10_readonly/trace.jsonl`
- `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/detection/summary.json`
- `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/bifurcation/summary.json`
- `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/stability/summary.json`
- `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/synthesis/summary.json`

Takeaway:

- geometry still beat entropy, though less strongly than in the first block
- `Layer 6` remained strongest local onset candidate
- the largest later hallucination expansion was resolved at `10 -> 11`

## Key insights worth preserving

1. Observation and intervention are now empirically separable in the current stack.
2. Hallucination-prone dynamics are not well described as entropy alone.
3. The current GPT-2 picture is better described as:
   - early local divergence onset at `Layer 6`
   - later expansion at `10 -> 11`
4. Transition prompts remain the main ambiguity boundary.
5. The current program should still be read as `GPT-2 supported`, not `generalized`.

## Claim boundary

Allowed now:

- geometry is more informative than entropy on the current GPT-2 panel
- `Layer 6` is the strongest current local onset candidate
- `10 -> 11` is the strongest current later expansion step in the dense sweep
- regime fingerprints are distinct at regime level in the present setup

Not yet allowed:

- cross-model generalization
- production-ready early warning
- a universal onset or commit layer
- attractor-level claims

## Canonical reading order

1. `reports/findings/summary_findings_2026-03-06.md`
2. `reports/findings/findings_2026-03-10.md`
3. `reports/findings/oscilloscope_hallu_summary_2026-03-10.md`
4. `reports/findings/observer_distortion_trajectory_compare_2026-03-10.md`
5. `reports/findings/trajectory_detection_findings_2026-03-10.md`
6. `reports/findings/layer_bifurcation_findings_2026-03-10.md`
7. `reports/findings/regime_stability_findings_2026-03-10.md`
8. `reports/findings/dense_layer_sweep_findings_2026-03-10.md`
9. `reports/syntheses/current_trajectory_findings_2026-03-10.md`
10. `reports/syntheses/trajectory_block_synthesis_2026-03-10.md`

## Next block

The active next block is already locked in:

- `reports/internal_ops/ESA_WEEK_PLAN_2026-03-10_GPT2_VALIDATION.md`

Operational next steps:

1. expand the GPT-2 observability panel
2. rerun the dense `5-12` trajectory block on the expanded panel
3. add token-level lead-time analysis
4. run a transition / regime-adjacent counter-case pass
5. write the Friday internal review pack before any control-model move

## First command for the next session

The next session should begin by preparing the expanded panel, not by rerunning models blindly.

If work resumes in code first, start by identifying or creating the expanded panel file under
`data/`, then lock the dense block rerun command against that panel in the active week plan.
