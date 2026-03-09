# Unified Observability Stack Protocol

Date: 2026-03-07
Scope: `ESA/research/mechanistic-interpretability/`

## Purpose

This protocol defines a shared microscopy stack for running:

1. `SAE` feature telemetry
2. `Logit Lens` decision projection
3. frontier metrics
4. aligned recorder traces

on the same prompts, layers, and intervention labels.

The purpose is not to replace the current `Field View` stack, but to extend it into a more complete observability layer for mechanistic analysis.

This protocol also serves as the baseline observability substrate for the next microscopy protocol:

- `L-SAE+R`

## Canonical evaluation panel

The first shared panel is:

- `data/prompts_observability_panel_2026-03-07.jsonl`

It contains a compact fixed set of:

- anchored prompts
- reasoning prompts
- transition-like prompts
- hallucination-prone prompts

The same panel should feed every layer of the stack.

## Stack order

The implementation order is deliberately staged:

1. `SAE`
2. `Logit Lens`
3. frontier metrics
4. recorder

`Tuned Lens` is not the starting point. The current default is to establish a `Logit Lens` baseline first and only add Tuned Lens if it contributes signal beyond that baseline.

## Recorder grain

The recorder emits one record per:

- prompt id
- layer
- token index
- intervention state

Minimum fields:

- `residual_norm`
- `subspace_coords`
- `subspace_operator_strength`
- `sae_top_features`
- `lens_entropy`
- `lens_topk`
- `frontier_coherence`
- `frontier_degeneracy`
- `gap_state_to_candidates`

Derived drift fields:

- `feature_drift_vs_prev_layer`
- `lens_entropy_delta_vs_prev_layer`
- `frontier_gap_delta_vs_prev_layer`
- `operator_strength_delta_vs_prev_layer`

## Reference scripts

- stack runner: `scripts/run_unified_observability_stack.py`
- trace plotter: `scripts/plot_unified_stack_traces.py`

## Example run

```bash
python3 scripts/run_unified_observability_stack.py \
  --prompt-jsonl data/prompts_observability_panel_2026-03-07.jsonl \
  --model gpt2 \
  --layers 3 5 6 9 12 \
  --sae-state experiments/exp_001_sae_v3/sae_weights.pt \
  --basis-mode pc2 \
  --topk 10 \
  --device cpu
```

Then:

```bash
python3 scripts/plot_unified_stack_traces.py \
  --trace-jsonl experiments/exp_004_unified_observability_stack/<run_name>/trace.jsonl \
  --out reports/figures/unified_stack_traces_<run_name>.png
```

## Success criteria

The stack is useful when it enables questions such as:

- does feature drift precede logit drift?
- does logit drift precede frontier collapse?
- which layers diverge earliest between reasoning and hallucination-prone traces?
- which top SAE features remain stable in reasoning and destabilize in hallucination-prone traces?

## Claim boundary

The current implementation should be treated as a microscopy instrumentation layer.

Allowed claims:

- the stack provides aligned feature, lens, frontier, and recorder outputs on the same material
- the stack supports layer-wise and regime-wise comparison
- the stack is now sufficient to make `L-SAE+R` testable on shared material

Not yet allowed:

- Tuned Lens improves the system
- the ordering of collapse is already established
- the recorder alone proves a general hallucination mechanism
