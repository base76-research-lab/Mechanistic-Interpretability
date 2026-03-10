# Reproducibility Guide

This repository is an active research surface. Reproducibility should therefore be read at the
level of:

- exact commands
- exact input panels
- exact dated findings notes
- exact artifact paths produced by each run

The current supported claims are scoped to the GPT-2 Small setup described in `STATUS.md` and the
dated findings notes under `reports/`.

## Environment

Set up a clean environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Default assumptions in the commands below:

- model: `gpt2`
- device: `cpu`
- prompt panel: `data/prompts_observability_panel_2026-03-07.jsonl`
- SAE weights: `experiments/exp_001_sae_v3/sae_weights.pt`

Large model weights are fetched from Hugging Face at runtime and are not stored in this repository.
Large tensor artifacts are treated as build outputs and may be ignored by git.

## Result A: read-only oscilloscope traces show a transition zone around L6-L9

Primary references:

- `reports/findings/oscilloscope_hallu_summary_2026-03-10.md`
- `STATUS.md`

Run:

```bash
PYTHONPATH=. python3 -m transformer_oscilloscope.cli trace \
  --prompt-jsonl data/prompts_observability_panel_2026-03-07.jsonl \
  --model gpt2 \
  --layers 1 6 9 11 \
  --out-dir experiments/exp_004_unified_observability_stack \
  --run-name transformer_oscilloscope_demo \
  --store-projections

PYTHONPATH=. python3 -m transformer_oscilloscope.cli report \
  --trace experiments/exp_004_unified_observability_stack/transformer_oscilloscope_demo/trace.jsonl \
  --out-dir experiments/exp_004_unified_observability_stack/transformer_oscilloscope_demo/plots \
  --report-name report.html
```

Expected outputs:

- `experiments/exp_004_unified_observability_stack/transformer_oscilloscope_demo/trace.jsonl`
- `experiments/exp_004_unified_observability_stack/transformer_oscilloscope_demo/plots/report.html`

Review target:

- inspect entropy, gap, and PCA trajectories in the HTML report
- compare the dated interpretation against `reports/findings/oscilloscope_hallu_summary_2026-03-10.md`

## Result B: reconstruction/write-back behaves as intervention, not neutral observation

Primary references:

- `reports/findings/findings_2026-03-10.md`
- `reports/findings/oscilloscope_hallu_summary_2026-03-10.md`

Run baseline panel:

```bash
python3 scripts/run_unified_observability_stack.py \
  --prompt-jsonl data/prompts_observability_panel_2026-03-07.jsonl \
  --sae-state experiments/exp_001_sae_v3/sae_weights.pt \
  --layers 3 5 6 9 12 \
  --intervention-state baseline \
  --run-name hallu_panel_baseline_2026-03-10 \
  --device cpu
```

Run reconstructed panel:

```bash
python3 scripts/run_unified_observability_stack.py \
  --prompt-jsonl data/prompts_observability_panel_2026-03-07.jsonl \
  --sae-state experiments/exp_001_sae_v3/sae_weights.pt \
  --layers 3 5 6 9 12 \
  --intervention-state baseline_recon \
  --use-sae-reconstruction \
  --run-name hallu_panel_baseline_recon_2026-03-10 \
  --device cpu
```

Expected outputs:

- `experiments/exp_004_unified_observability_stack/hallu_panel_baseline_2026-03-10/trace.jsonl`
- `experiments/exp_004_unified_observability_stack/hallu_panel_baseline_2026-03-10/metadata.json`
- `experiments/exp_004_unified_observability_stack/hallu_panel_baseline_recon_2026-03-10/trace.jsonl`
- `experiments/exp_004_unified_observability_stack/hallu_panel_baseline_recon_2026-03-10/metadata.json`

Review target:

- compare `intervention_state`, `gap_state_to_candidates`, `frontier_coherence`, and
  `lens_entropy` across matched prompts/layers
- use the dated findings note rather than treating raw trace differences as self-interpreting

## Result C: hallucination rate stays high while abstention collapses under recon-active settings

Primary references:

- `reports/findings/findings_2026-03-10.md`
- `STATUS.md`

Run baseline benchmark:

```bash
python3 scripts/hallu_benchmark.py \
  --model gpt2 \
  --run-name baseline_hallu_bench_v3
```

Run gated reconstruction benchmark:

```bash
python3 scripts/hallu_benchmark.py \
  --model gpt2 \
  --run-name gate_L6L9_alpha013_tau02 \
  --use-sae-reconstruction \
  --sae-state experiments/exp_001_sae_v3/sae_weights.pt \
  --layers 6 9 \
  --alpha-per-layer 0.1 0.3 \
  --tau-entropy 0.2 \
  --tau-delta 0.2
```

Expected outputs:

- `experiments/exp_004_unified_observability_stack/baseline_hallu_bench_v3_hallu_benchmark.jsonl`
- `experiments/exp_004_unified_observability_stack/gate_L6L9_alpha013_tau02_hallu_benchmark.jsonl`
- printed JSON summaries with `factual_accuracy`, `hallucination_rate`, and `abstention_rate`

Review target:

- verify that hallucination rate does not materially improve in the small benchmark
- verify that abstention decreases when reconstruction is active

## Claim boundary

These commands reproduce the current repository-level evidence surface. They do not establish:

- cross-model generalization
- production-grade hallucination mitigation
- that write-back behavior can be interpreted as passive observation

For the current claim boundary, always read:

- `README.md`
- `STATUS.md`
- `research_index.md`
- the dated findings notes referenced above
