# Observer Distortion Trajectory Compare — 2026-03-10

## Purpose

Test whether the new read-only Transformer Oscilloscope follows the same layer trajectory as the
existing unified observability stack when both are projected into the same SAE-defined subspace.

The critical comparison is:

- read-only observer vs unified baseline
- read-only observer vs baseline + reconstruction/write-back

If read-only and baseline agree while read-only and reconstruction diverge, the distortion can be
attributed to write-back rather than to instrumentation in general.

## Setup

Shared setup:

- model: `gpt2`
- panel: `data/prompts_observability_panel_2026-03-07.jsonl`
- layers: `3 5 6 9 12`
- SAE state: `experiments/exp_001_sae_v3/sae_weights.pt`
- basis: `pc2`
- units: `472 468 57 156 346`

Read-only trace:

- `experiments/exp_004_unified_observability_stack/hallu_panel_readonly_oscilloscope_subspace_2026-03-10/trace.jsonl`

Comparison outputs:

- `experiments/exp_004_unified_observability_stack/trajectory_compare_readonly_vs_baseline_2026-03-10/`
- `experiments/exp_004_unified_observability_stack/trajectory_compare_readonly_vs_recon_2026-03-10/`

## Result

### 1. Read-only vs unified baseline

Summary:

- shared points: `80`
- prompts: `16`
- layers: `5`
- mean point distance: `2.089e-06`
- max point distance: `1.122e-05`

Interpretation:

- in the shared SAE subspace, the read-only oscilloscope and unified baseline are effectively
  trajectory-identical
- ordinary aligned instrumentation is not the source of large trajectory drift in the current setup

### 2. Read-only vs baseline + reconstruction

Summary:

- shared points: `80`
- prompts: `16`
- layers: `5`
- mean point distance: `3.197`
- max point distance: `12.512`

Interpretation:

- reconstruction/write-back produces large trajectory displacement relative to the read-only trace
- the distortion is not subtle: it is several orders of magnitude larger than the read-only vs
  baseline difference

## Most affected prompts

Highest mean trajectory displacement in the current comparison:

- `contrast_degeneracy_01` — `4.005`
- `random_baseline_math_01` — `3.966`
- `hallucination_01` — `3.800`
- `random_baseline_degeneracy_01` — `3.759`
- `reasoning_03` — `3.661`
- `hallucination_02` — `3.572`

## Claim boundary

Allowed claim:

- in the current GPT-2 Small setup, write-back reconstruction measurably distorts layer trajectories
  in a shared SAE subspace, while the read-only oscilloscope agrees almost exactly with the unified
  no-write-back baseline

Not yet allowed:

- the same distortion magnitude generalizes to other models
- every earlier microscopy surface was equally intervention-heavy
- trajectory displacement by itself explains hallucination causally

## Consequence

This supports a stronger observer/intervention rule:

- read-only oscilloscope can be treated as the default trajectory observer
- unified baseline remains compatible with that observer surface
- write-back traces must be treated as causal perturbation runs, not observational evidence
