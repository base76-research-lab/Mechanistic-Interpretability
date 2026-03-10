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
- `experiments/exp_004_unified_observability_stack/hallu_benchmark_2026-03-10.png`

## Latest findings

- `reports/summary_findings_2026-03-06.md`
- `reports/compression_analysis_2026-03-06.md`
- `reports/steering_vector_analysis_2026-03-06.md`
- `reports/findings_2026-03-10.md`
- `reports/oscilloscope_hallu_summary_2026-03-10.md`
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

## Active instrumentation additions

- unified stack runner: `scripts/run_unified_observability_stack.py`
- trace plotter: `scripts/plot_unified_stack_traces.py`
- canonical evaluation panel: `data/prompts_observability_panel_2026-03-07.jsonl`
- Monday sanity runner: `scripts/run_monday_sanity_pass.py`
- Transformer Oscilloscope: `transformer_oscilloscope/`

## Active planning documents

- `reports/RESEARCH_DESIGN_AND_INSTRUMENT_PLAN_2026-03-10.md`
- `reports/COMPUTE_ELASTICITY_PLAN_2026-03-10.md`

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
- is the real transition zone better described as L6-L9 rather than a single privileged layer?
- does vectorized conditioning outperform text-only compression in a stable way?
- which signals survive batch validation and cross-model controls?
- does lens supervision improve sparse-feature utility without collapsing microscope value?
- can type-aware, structure-protected text compression become the best current structure-preserving method on the full robust panel, rather than only a usable intervention?
- is the current vector proxy a prompt replacement, a diagnostic representation, or neither?
- how should observer traces be separated from write-back interventions in the canonical claims surface?

## Immediate execution order

1. patch `anchored_03` and recall-anchor handling for `hallucination_02` / `hallucination_03`
2. run a read-only oscilloscope-assisted layer replication around L6-L9 on the hallucination panel
3. rerun robust text-only compression batch on the observability panel
4. if `compressed` becomes defensible, freeze the text-compression path as the current intervention baseline
5. only then resume `VECTOR_PROXY_LEGITIMACY_PROTOCOL_2026-03-08.md`

## Next required transition

ANALYSIS -> INTERNAL_REVIEW
