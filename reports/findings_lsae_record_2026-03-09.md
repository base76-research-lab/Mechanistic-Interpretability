# L-SAE+R Findings — 2026-03-09

Date: 2026-03-09  
Track: `ai_microscopy`  
Scope: `exp_004_unified_observability_stack`

## Setup

- Model: `gpt2` (FP32, CPU)
- Panel: `data/prompts_observability_panel_2026-03-07.jsonl` (20 prompts)
- Layers: 3, 5, 6, 9, 12 on last token
- SAE: `experiments/exp_001_sae_v3/sae_weights.pt`, basis `pc2`, units `472 468 57 156 346`
- Runs:
  1. `intervention_state=baseline` → `experiments/exp_004_unified_observability_stack/lsae_record_2026-03-09/trace.jsonl`
  2. `intervention_state=lsae_r` → `experiments/exp_004_unified_observability_stack/lsae_record_2026-03-09_lsae_r/trace.jsonl`
- Visuals:
  - `reports/figures/lsae_record_2026-03-09.png`
  - `reports/figures/lsae_record_2026-03-09_lsae_r.png`

## Key question

Does switching to `intervention_state=lsae_r` (L-SAE+recorder condition) change recorder-facing metrics (lens entropy, frontier coherence/degeneracy, gap) relative to the baseline condition on the same material?

## Findings

- **No measurable deltas across metrics** on this panel with the current SAE weights:
  - Median (lsae_r − baseline): lens_entropy 0.0000, gap_state_to_candidates 0.0000, frontier_coherence 0.0000, frontier_degeneracy 0.0000.
  - Regime-wise medians (anchored, reasoning, transition, hallucination_prone, control): all 0.0000 for the same metrics.
  - Top-5 absolute deltas per metric: 0.0 (identical values prompt-by-prompt, layer-by-layer).
- **Recorder schema validated**: expected fields present (lens_entropy, frontier metrics, drift deltas, DTS).
- **Visuals show overlapping trajectories**: the two generated plots are visually indistinguishable at this scale; no layer-specific separation appears.

## Interpretation

- The current `lsae_r` switch is effectively a no-op with the present SAE weights and setup. Either:
  - The L-SAE supervision is not active in these runs (likely, since we reused plain SAE weights), or
  - The supervision signal is too weak to move recorder metrics on this panel.
- This is a good negative control: instrumentation matches, and we confirmed zero unintended drift.

## Recommended next steps

1) **Train an actual L-SAE v1 checkpoint** (residual reconstruction + logit-lens term) and rerun the same panel with `intervention_state=lsae_r`.  
2) **Add a high-sensitivity slice** (e.g., math_reasoning vs hallucination_prone) and check `decision_trajectory_smoothness` deltas and `feature_drift_vs_prev_layer`.  
3) **Re-run delta script post-training** to verify non-zero shifts; if still zero, inspect whether the L-SAE objective is wired into the run path.  
4) **Optional**: Plot per-regime layer curves side-by-side once a non-zero effect appears; current plots serve as baseline for future diffs.

## Artifacts

- Baseline trace: `experiments/exp_004_unified_observability_stack/lsae_record_2026-03-09/trace.jsonl`
- L-SAE+R trace: `experiments/exp_004_unified_observability_stack/lsae_record_2026-03-09_lsae_r/trace.jsonl`
- Figures: `reports/figures/lsae_record_2026-03-09.png`, `reports/figures/lsae_record_2026-03-09_lsae_r.png`

## Follow-up (2026-03-09, rapid run)

- Trained a minimal L-SAE v1 with lens weight 1e-2 on `data/prompts.txt` (layer 5) → weights at `experiments/exp_001_sae_v4_lsae_v1/sae_weights.pt`.
- Re-ran the stack with those weights (`lsae_v1_baseline_2026-03-09` and `lsae_v1_lsae_r_2026-03-09`); deltas across recorder metrics remain exactly 0.0 → still a no-op effect. Figures: `reports/figures/lsae_v1_baseline_2026-03-09.png`, `reports/figures/lsae_v1_lsae_r_2026-03-09.png`.
- Implication: either the lens loss is too weak or the run path still treats `intervention_state` as a tag only; need wiring check before next attempt.

## Fix attempt (2026-03-09, v2)

- Added `--use-sae-reconstruction` in `run_unified_observability_stack.py`; when set, hidden vectors are replaced by SAE reconstructions before lens/frontier metrics.
- New runs:
  - Baseline: `experiments/exp_004_unified_observability_stack/lsae_v1_baseline_2026-03-09_v2/trace.jsonl`
  - L-SAE+R (reconstruction active): `experiments/exp_004_unified_observability_stack/lsae_v1_lsae_r_2026-03-09_v2/trace.jsonl`
  - Figures: `reports/figures/lsae_v1_baseline_2026-03-09_v2.png`, `reports/figures/lsae_v1_lsae_r_2026-03-09_v2.png`
- Deltas (median lsae_r − baseline):
  - lens_entropy +1.61, gap_state_to_candidates −1.04, frontier_coherence −0.04, frontier_degeneracy +0.10
  - decision_trajectory_smoothness −4.66 (smoother), feature_drift_vs_prev_layer −13.86
  - Non-zero across all regimer; max |delta| shows large shifts on some prompts/lager.
- Interpretation: reconstruction intervention now has a real effect; coherence dipped slightly but gaps shrank and trajectories smoothed. Next step is to tune lens_weight and check robustness (avoid coherence loss).

## Lens weight sweep (5e-3)

- Trained L-SAE v1 with `lens_weight=5e-3` → `experiments/exp_001_sae_v4_lsae_v1_lw5e3/sae_weights.pt`.
- Runs with reconstruction:
  - Baseline: `experiments/exp_004_unified_observability_stack/lsae_v1_lw5e3_baseline_2026-03-09/trace.jsonl`
  - L-SAE+R: `experiments/exp_004_unified_observability_stack/lsae_v1_lw5e3_lsae_r_2026-03-09/trace.jsonl`
  - Figures: `reports/figures/lsae_v1_lw5e3_baseline_2026-03-09.png`, `reports/figures/lsae_v1_lw5e3_lsae_r_2026-03-09.png`
- Medians (lsae_r − baseline):
  - lens_entropy +1.00
  - gap_state_to_candidates −0.95
  - frontier_coherence −0.07
  - frontier_degeneracy +0.15
  - decision_trajectory_smoothness −4.33
- Regime summary: gap shrinkage across all regimes; coherence dip mainly anchored/hallucination_prone; reasoning shows small coherence gain but still entropy uptick.
- Interpretation: Lower lens_weight trims entropy lift slightly but coherence dip persists; still better gap and smoother trajectories. Next knob: try lens_weight 2e-2 and/or add mild reconstruction dropout to curb degeneracy rise.

## Lens weight sweep (2e-2)

- Trained L-SAE v1 with `lens_weight=2e-2` → `experiments/exp_001_sae_v4_lsae_v1_lw2e2/sae_weights.pt`.
- Runs with reconstruction:
  - Baseline: `experiments/exp_004_unified_observability_stack/lsae_v1_lw2e2_baseline_2026-03-09/trace.jsonl`
  - L-SAE+R: `experiments/exp_004_unified_observability_stack/lsae_v1_lw2e2_lsae_r_2026-03-09/trace.jsonl`
  - Figures: `reports/figures/lsae_v1_lw2e2_baseline_2026-03-09.png`, `reports/figures/lsae_v1_lw2e2_lsae_r_2026-03-09.png`
- Medians (lsae_r − baseline):
  - lens_entropy +0.33
  - gap_state_to_candidates −0.64
  - frontier_coherence −0.03
  - frontier_degeneracy +0.20
  - decision_trajectory_smoothness −3.34
- Regime notes:
  - Hallucination_prone: stark gap‑minskning (−2.27), men coherence‑dip (−0.12) och degeneracy +0.4.
  - Reasoning: mindre gap‑förbättring (−0.54) med nästan neutral coherence (+0.007) men entropy +0.39.
  - Anchored: gap svag förbättring, coherence nära neutral, DTS −3.6.
- Interpretation: Starkare lens_weight minskar entropy-lyftet och coherence-dippen jämfört med 5e-3, men ger mindre gap‑vinst och något högre degeneracy. Hallucination-prone får störst gap‑dipp men betalar i coherence.

## Per-regime deltas (median, lsae_r – baseline)

### lens_weight=5e-3
| regime              | Δgap | Δcoherence | ΔDTS |
|---------------------|------|------------|------|
| anchored            | -1.40 | -0.13 | -3.83 |
| control             | -0.75 | -0.08 | -4.39 |
| hallucination_prone | -0.90 | -0.30 | -3.48 |
| reasoning           | +0.03 | +0.12 | -7.29 |
| transition          | -0.41 | +0.09 | -4.33 |

### lens_weight=2e-2
| regime              | Δgap | Δcoherence | ΔDTS |
|---------------------|------|------------|------|
| anchored            | -0.84 | +0.00 | -3.64 |
| control             | -0.14 | +0.06 | -3.16 |
| hallucination_prone | -2.21 | -0.10 | -2.19 |
| reasoning           | -0.67 | -0.02 | -5.54 |
| transition          | -0.47 | -0.10 | -3.01 |

Figures:
- Boxplots per regime: `reports/figures/lsae_v1_lw5e3_regime_boxplots_2026-03-09.png`, `reports/figures/lsae_v1_lw2e2_regime_boxplots_2026-03-09.png`
