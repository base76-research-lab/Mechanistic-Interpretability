# STATUS — Mechanistic Interpretability

Last updated: 2026-03-10

## Current position

**Active dialogue with ESA Phi-lab** (Giuseppe Borghi -> Nicolas Longepe; awaiting follow-up).

The project now has a functioning observability stack for internal geometry in transformer models.
Current findings suggest that state-candidate misalignment in the residual stream is a measurable
precursor to hallucination-like behavior, and that this signal is not reducible to high entropy alone.

## Core findings

| Claim | Evidence |
|---|---|
| Latent state-space structure can be measured through subspace projection | `exp_001_sae_v3` artifacts |
| Four state regimes (A-D) are observable on controlled prompts | Field View JSON artifacts and triage figure |
| State-candidate misalignment correlates with hallucination-like behavior | `field_view_hallucination.json` |
| Entropy alone is insufficient as a hallucination signal | preliminary high-entropy regime comparison |
| Reconstruction/write-back acts as an intervention rather than a neutral observer | `reports/findings_2026-03-10.md`, `reports/oscilloscope_hallu_summary_2026-03-10.md` |
| Read-only oscilloscope traces show a recurrent decision-transition zone around L6-L9 | `experiments/exp_004_unified_observability_stack/transformer_oscilloscope_demo/`, `reports/oscilloscope_hallu_summary_2026-03-10.md` |
| Read-only oscilloscope and unified baseline are trajectory-identical in shared SAE subspace, while reconstruction diverges strongly | `reports/observer_distortion_trajectory_compare_2026-03-10.md`, `experiments/exp_004_unified_observability_stack/trajectory_compare_readonly_vs_baseline_2026-03-10/summary.json`, `experiments/exp_004_unified_observability_stack/trajectory_compare_readonly_vs_recon_2026-03-10/summary.json` |
| Geometry-driven detection separates hallucination-prone prompts better than entropy at full-panel level, including the denser `5-12` sweep | `reports/trajectory_detection_findings_2026-03-10.md`, `reports/dense_layer_sweep_findings_2026-03-10.md`, `experiments/exp_005_trajectory_block/analysis_2026-03-10/detection/summary.json`, `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/detection/summary.json` |
| Layer 6 remains the strongest current local bifurcation candidate, while the largest later hallucination expansion currently appears at `10->11` in the dense sweep | `reports/layer_bifurcation_findings_2026-03-10.md`, `reports/dense_layer_sweep_findings_2026-03-10.md`, `experiments/exp_005_trajectory_block/analysis_2026-03-10/bifurcation/summary.json`, `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/bifurcation/summary.json` |
| Regime stability fingerprints are distinct at regime level in the current GPT-2 panel | `reports/regime_stability_findings_2026-03-10.md`, `experiments/exp_005_trajectory_block/analysis_2026-03-10/stability/summary.json` |

## Active hypotheses

| ID | Hypothesis | Status |
|---|---|---|
| H1 | Token compression accelerates the emergence of states B/C/D | strong observation, but A/B comparison remains incomplete |
| H2 | The four-state taxonomy is robust across models and domains | GPT-2 only; Phi-2 control planned |
| H3 | `risk_refined = entropy_norm * gap_norm * (1 - coherence)` outperforms the current risk signal | not yet validated |
| H4 | Hallucination has a localizable onset point in layer dynamics | strengthened: dense sweep supports early local onset at Layer 6 plus later expansion at `10->11`; broader replication still needed |

## Recent runs (2026-03-10)

- Hallucination panel reruns: baseline, baseline+recon, L-SAE no recon, L-SAE+R (observability stack). Trace dirs under `experiments/exp_004_unified_observability_stack/`.
- Hallucination QA benchmarks (small panels): baseline, α-blends, entropy-gated recon; hallu-rate stayed 1.0, abstention collapsed when recon active.
- Transformer Oscilloscope: new read-only tracing tool; demo + smoke traces with PNGs/HTML report.
- Trajectory A/B in shared SAE subspace: read-only oscilloscope vs unified baseline (`mean distance ~= 2.1e-06`) and vs baseline+recon (`mean distance ~= 3.20`).
- First trajectory block analysis: geometry detection beat entropy at panel level (`AUC 0.821 vs 0.513`), Layer 6 remained strongest bifurcation candidate, and regime fingerprints cleared regime-level separation.
- Dense `5-12` layer sweep: geometry still beat entropy (`AUC 0.692 vs 0.462`), Layer 6 remained strongest local onset candidate, and the largest hallucination expansion shifted later to `10->11`.

## Experiment status

| Experiment | Description | Status |
|---|---|---|
| `exp_001_sae` v1/v2/v3 | SAE + Field View on GPT-2 layer 5 | complete |
| `exp_002_persona` | persona and traits artifacts | partial |
| `exp_003_compression_vectorized` | raw vs compressed vs vectorized proxy (`mean` / `attn_weighted` / `pca1`) | in progress; robust batch still missing |
| `exp_004_unified_observability_stack` | unified recorder traces, hallucination panel reruns, QA benchmarks, oscilloscope demo/smoke | exploratory runs complete; interpretation boundary updated |
| Transformer Oscilloscope | Read-only residual/attn/MLP/logit tracing + viz | new tool added; demo run done |

**Critical gap:** `exp_003` robust batch with `--require-compressor` still needed.

## Priority next steps

1. Run a robust batch for `exp_003` (see `reports/NEXT_STEPS_2026-03-05.md`)
2. Expand the prompt panel and rerun the dense `5-12` trajectory block to test whether Layer 6 onset and `10->11` expansion remain stable
3. Add token-level lead-time analysis before promoting any early-warning framing
4. Run a light Phi-2 notebook control to test H2 after the GPT-2 trajectory block is strengthened

## Active track plans

- Microscopy plan: `reports/MODEL_MICROSCOPY_PLAN_2026-03-07.md`
- Weekly operational plan: `reports/internal_ops/ESA_WEEK_PLAN_2026-03-09.md`
- Research design + instrument plan: `reports/RESEARCH_DESIGN_AND_INSTRUMENT_PLAN_2026-03-10.md`
- Compute elasticity plan: `reports/COMPUTE_ELASTICITY_PLAN_2026-03-10.md`

## ESA context

- Sent: outreach and research package on 2026-03-05
- Response: Giuseppe Borghi replied positively and forwarded the thread to Nicolas Longepe
- Current state: awaiting contact from Nicolas Longepe
- Next action: no immediate outbound action required
- Related material: `/media/bjorn/iic/ESA/`

## Key files

- `reports/exp_001_sae.md` — SAE + Field View results, risk formulation, and threats to validity
- `reports/preliminary_2026-03-05_epistemic_state_layer.md` — early four-state model and refined risk framing
- `reports/NEXT_STEPS_2026-03-05.md` — operational run plan with commands and exit criteria
- `reports/feature_dict.md` — feature dictionary with labels and clusters
- `reports/findings_2026-03-10.md` — current interpretation of `exp_004` panel and benchmark runs
- `reports/oscilloscope_hallu_summary_2026-03-10.md` — oscilloscope-specific summary and observer/intervention boundary
- `reports/observer_distortion_trajectory_compare_2026-03-10.md` — shared-subspace A/B showing near-zero read-only vs baseline drift and strong read-only vs recon divergence
- `reports/current_trajectory_findings_2026-03-10.md` — canonical current synthesis across the trajectory findings surface
- `reports/trajectory_detection_findings_2026-03-10.md` — first trajectory-block detection result
- `reports/layer_bifurcation_findings_2026-03-10.md` — layer divergence result for the first trajectory block
- `reports/regime_stability_findings_2026-03-10.md` — regime fingerprint result for the first trajectory block
- `reports/trajectory_block_synthesis_2026-03-10.md` — synthesis across the first trajectory block
- `reports/dense_layer_sweep_findings_2026-03-10.md` — dense `5-12` addendum resolving early onset vs later expansion
- `reports/figures/field_view_triage.png` — primary triage figure
- `experiments/exp_001_sae_v3/` — core JSON artifacts and run outputs
- `experiments/exp_003_compression_vectorized/` — vectorized proxy results
- `experiments/exp_004_unified_observability_stack/` — unified observability runs, traces, plots, and benchmark outputs
- `experiments/exp_005_trajectory_block/` — first trajectory-block observer traces and analysis outputs
