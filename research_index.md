# Research Index

## Current state

ANALYSIS

## Active protocols

- `../../experiments/protocols/vectorized_compression_experiment_v1.md`
- `reports/UNIFIED_OBSERVABILITY_STACK_PROTOCOL_2026-03-07.md`
- `reports/L_SAE_R_PROTOCOL_2026-03-07.md`
- `reports/L_SAE_R_RESEARCH_QUESTION_2026-03-07.md`
- `reports/COMPRESSION_VALIDATION_STANDARD_2026-03-07.md`
- `reports/COMPRESSION_REPAIR_PROTOCOL_2026-03-08.md`
- `reports/VECTOR_PROXY_LEGITIMACY_PROTOCOL_2026-03-08.md`

## Latest runs

- `experiments/layer_sweep_top_risk_2026-03-06.csv`
- `experiments/compression_results_2026-03-06.csv`
- `experiments/steering_vector_results_2026-03-06.csv`
- `experiments/exp_003_compression_vectorized/summary_20260308T140345Z.json`
- `experiments/exp_003_compression_vectorized/summary_20260308T140825Z.json`
- `experiments/exp_003_compression_vectorized/summary_20260308T142252Z.json`
- `experiments/exp_003_compression_vectorized/summary_20260308T143516Z.json`
- `experiments/exp_003_compression_vectorized/summary_20260308T144644Z.json`
- `experiments/exp_004_unified_observability_stack/hallu_panel_baseline_2026-03-10/trace.jsonl`
- `experiments/exp_004_unified_observability_stack/hallu_panel_lsae_r_2026-03-10/trace.jsonl`
- `experiments/exp_004_unified_observability_stack/transformer_oscilloscope_demo/trace.jsonl`
- `experiments/exp_004_unified_observability_stack/hallu_panel_readonly_oscilloscope_subspace_2026-03-10/trace.jsonl`
- `experiments/exp_004_unified_observability_stack/trajectory_compare_readonly_vs_baseline_2026-03-10/summary.json`
- `experiments/exp_004_unified_observability_stack/trajectory_compare_readonly_vs_recon_2026-03-10/summary.json`
- `experiments/exp_005_trajectory_block/readonly_full_metrics_2026-03-10/trace.jsonl`
- `experiments/exp_005_trajectory_block/analysis_2026-03-10/detection/summary.json`
- `experiments/exp_005_trajectory_block/analysis_2026-03-10/bifurcation/summary.json`
- `experiments/exp_005_trajectory_block/analysis_2026-03-10/stability/summary.json`
- `experiments/exp_005_trajectory_block/analysis_2026-03-10/synthesis/summary.json`
- `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/dense_layers_5_12_2026-03-10_readonly/trace.jsonl`
- `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/detection/summary.json`
- `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/bifurcation/summary.json`
- `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/stability/summary.json`
- `experiments/exp_005_trajectory_block/dense_layers_5_12_2026-03-10/analysis/synthesis/summary.json`
- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/block_manifest.json`
- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/detection/summary.json`
- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/bifurcation/summary.json`
- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/stability/summary.json`
- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/synthesis/summary.json`
- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/lead_time/summary.json`
- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/transition_countercase/summary.json`
- `experiments/exp_004_unified_observability_stack/hallu_benchmark_2026-03-10.png`

## Latest findings

- `reports/findings/summary_findings_2026-03-06.md`
- `reports/compression_analysis_2026-03-06.md`
- `reports/steering_vector_analysis_2026-03-06.md`
- `reports/findings/findings_2026-03-10.md`
- `reports/findings/oscilloscope_hallu_summary_2026-03-10.md`
- `reports/findings/observer_distortion_trajectory_compare_2026-03-10.md`
- `reports/syntheses/current_trajectory_findings_2026-03-10.md`
- `reports/findings/trajectory_detection_findings_2026-03-10.md`
- `reports/findings/layer_bifurcation_findings_2026-03-10.md`
- `reports/findings/regime_stability_findings_2026-03-10.md`
- `reports/syntheses/trajectory_block_synthesis_2026-03-10.md`
- `reports/findings/dense_layer_sweep_findings_2026-03-10.md`
- `reports/findings/expanded_panel_validation_findings_2026-03-10.md`
- `reports/findings/token_level_lead_time_findings_2026-03-10.md`
- `reports/findings/transition_countercase_findings_2026-03-10.md`
- `transformer_oscilloscope/` — read-only tracing/viz toolkit (trace, viz, report)

## Emerging protocol candidates

- `L-SAE+R` is now the next explicit microscopy protocol built on top of the unified observability stack baseline
- current claim level for `L-SAE+R`: `Exploratory`
- compression repair is now separated from vector proxy legitimacy
- vector proxy should be treated as an experimental representation, not a legitimate prompt substitute by default
- text compression is now a functioning research intervention in the current GPT-2 Small setup, but it is not yet the defensible best current method on the full robust panel
- the remaining compression blocker is narrow: `anchored_03` drift and recall-anchor loss on `hallucination_02` / `hallucination_03`
- reconstruction/write-back must now be treated as intervention, not neutral observation
- the read-only oscilloscope should be treated as the default observer when mapping layer transitions before causal write-back tests
- read-only oscilloscope and unified no-write-back baseline are now empirically aligned in shared SAE subspace, while write-back runs diverge strongly
- first trajectory block now supports three GPT-2-level claims: geometry beats entropy at panel level, Layer 6 remains strongest local bifurcation candidate, and regime fingerprints are distinct at regime level
- dense `5-12` sweep now sharpens the GPT-2 picture: earliest strongest local divergence remains at Layer 6, while the largest hallucination expansion currently appears later at `10->11`
- expanded-panel validation now preserves the onset picture while showing that transition prompts remain the main ambiguity boundary and that geometry does not yet dominate entropy on every regime boundary
- token-level lead time is now measurable in a conservative hallucination-prone slice before prompt end, but operational thresholding remains non-specific against transition-adjacent prompts
- transition counter-cases now suggest that the main difference from the hallucination slice is lower post-onset persistence, not cleaner onset score alone
- phase velocity is worth retaining as an observer metric, but the dense sweep and lead-time slice do not yet support it as a standalone early-warning signal

## Active instrumentation additions

- unified stack runner: `scripts/run_unified_observability_stack.py`
- trace plotter: `scripts/plot_unified_stack_traces.py`
- canonical evaluation panel: `data/prompts_observability_panel_2026-03-07.jsonl`
- Monday sanity runner: `scripts/run_monday_sanity_pass.py`
- Transformer Oscilloscope: `transformer_oscilloscope/`
- Trajectory block runner: `scripts/run_trajectory_block.py`
- Trajectory block analysis: `scripts/analyze_trajectory_block.py`

## Active planning documents

- `reports/RESEARCH_DESIGN_AND_INSTRUMENT_PLAN_2026-03-10.md`
- `reports/COMPUTE_ELASTICITY_PLAN_2026-03-10.md`
- `reports/internal_ops/FRIDAY_INTERNAL_REVIEW_PACK_2026-03-13.md`

## Operational layer

- GitHub repository: `https://github.com/base76-research-lab/mechanistic-interpretability-`
- GitHub Project: `https://github.com/users/base76-research-lab/projects/1/views/1`
- active backlog: issues `#1` to `#8`

## Evidence level

Supported

## Claim boundary

Current results support a structured residual-state interpretation in the current GPT-2 Small setup, but do not yet justify cross-model generalization or production-grade detection claims.

## Open questions

- does Layer 6 remain critical outside the current setup?
- is the real transition picture best described as `Layer 6 onset + later 10->11 expansion`, or is that still a panel artifact?
- is post-onset persistence enough to retain as the main disambiguator in Friday review, or does it still collapse under a broader panel?
- does vectorized conditioning outperform text-only compression in a stable way?
- which signals survive batch validation and cross-model controls?
- does lens supervision improve sparse-feature utility without collapsing microscope value?
- can type-aware, structure-protected text compression become the best current structure-preserving method on the full robust panel, rather than only a usable intervention?
- is the current vector proxy a prompt replacement, a diagnostic representation, or neither?
- how should observer traces be separated from write-back interventions in the canonical claims surface?

## Immediate execution order

1. write the Friday internal review pack with panel, lead-time, and transition-counter-case conclusions
2. decide go / no-go for one control-model block
3. if `go`, define the smallest valid control-model replication block

## Next required transition

ANALYSIS -> INTERNAL_REVIEW
