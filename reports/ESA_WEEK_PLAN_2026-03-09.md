# ESA Week Plan: 2026-03-09 to 2026-03-13

Goal: produce ESA-ready evidence, not broad exploration.

## Status entering Monday

As of 2026-03-08, the compression workstream has already cleared the old integration blockers:

- the token-compressor loader is correctly wired
- the `ollama` Python dependency is installed and the compressor runs
- the Monday sanity path has passed once for text-only compression

The current blocker is now narrow and methodological rather than infrastructural:

- robust full-panel text compression still fails as a defensible best-current method
- remaining weak cases are centered on `anchored_03` drift and recall-anchor loss in `hallucination_02` / `hallucination_03`
- vector proxy remains deferred until text compression is stable enough to freeze as the baseline intervention

## Locked assumptions

- Primary model: GPT-2 Small
- Secondary control: Phi-2 (light Colab only)
- Priority: robust batch + targeted stress-tests + narrow cross-model check
- Non-goal: full cross-model generalization or new flagship metric

## Monday: compression stabilization reset

- Patch the remaining weak text-compression cases:
  - `anchored_03`
  - `hallucination_02`
  - `hallucination_03`
- Rerun the robust text-only panel, not the older mixed text/vector sanity path
- Internal-only agent runner available for bounded Monday execution and stop/go recommendation
- Keep `VECTOR_PROXY_LEGITIMACY_PROTOCOL_2026-03-08.md` deferred until text compression becomes the stable baseline

Historical sanity command:

```bash
cd "/media/bjorn/iic/workspace/Base76_Research_Lab/Mechanistic Interpretability"
python3 scripts/compare_compression_vectorized.py \
  --prompts-file data/prompts_sanity_2026-03-09.txt \
  --vector-methods mean attn_weighted pca1 \
  --require-compressor \
  --exclude-invalid-compression \
  --device cpu
```

Internal runner:

```bash
python3 scripts/run_monday_sanity_pass.py
```

## Tuesday: robust batch

Gate:

- only proceed if Monday yields a defensible text-only `compressed` method on the full observability panel
- if Monday still fails, Tuesday remains a narrow repair pass rather than vector or cross-model work

- Use `data/prompts_robust_2026-03-09.jsonl`
- Run stratified batch across:
  - relation/analogy
  - factual
  - instruction/planning
  - verbose/noise
- Save annotated JSON/CSV and scatter plot

Commands:

```bash
python3 scripts/run_exp003_week_batch.py \
  --prompt-jsonl data/prompts_robust_2026-03-09.jsonl \
  --vector-methods mean attn_weighted pca1 \
  --require-compressor \
  --exclude-invalid-compression \
  --device cpu
```

Then:

```bash
python3 scripts/plot_vector_mode_scatter.py \
  --results experiments/exp_003_compression_vectorized/week_of_2026-03-09/<annotated_results>.json \
  --out reports/figures/vector_mode_degeneracy_vs_coherence_week_2026-03-09.png
```

## Wednesday: analysis and method choice

- Summarize medians + percentiles for:
  - `delta_vs_raw_gap_state_to_candidates`
  - `delta_vs_raw_candidate_coherence`
  - `delta_vs_raw_degeneracy_ratio_topk`
  - `delta_vs_raw_logit_entropy`
- Produce prompt-type breakdown
- Choose winner by this order:
  1. lower gap median
  2. higher coherence median
  3. lower degeneracy median
  4. lower entropy median

Interpretation rule:

- text compression must become the defensible baseline intervention before vector proxy work is resumed

Command:

```bash
python3 scripts/summarize_exp003_results.py \
  --results experiments/exp_003_compression_vectorized/week_of_2026-03-09/<annotated_results>.json \
  --out-json experiments/exp_003_compression_vectorized/week_of_2026-03-09/method_summary.json \
  --out-md reports/exp_003_method_summary_2026-03-12.md
```

## Thursday: targeted stress-tests

- Trap A/B batch using `data/prompts_traps_2026-03-09.jsonl`
- Layer sweep for hallucination prompt at layers `3, 6, 9, 12`
- W_U projection check on 3-5 prompts

Recommended checks:

```bash
python3 scripts/run_exp003_week_batch.py \
  --prompt-jsonl data/prompts_traps_2026-03-09.jsonl \
  --vector-methods mean attn_weighted pca1 \
  --require-compressor \
  --exclude-invalid-compression \
  --device cpu
```

```bash
python3 scripts/wu_projection_check.py \
  --prompt "who was the president of france in 1200?" \
  --model gpt2 \
  --layer 6 \
  --topk 10
```

## Friday: Phi-2 light control + packaging

- Run only 4 prompts from `data/prompts_phi2_smoke_2026-03-09.jsonl`
- Record one of:
  - same pattern observed
  - partial pattern observed
  - inconclusive due to setup/compute
- Complete summary pack

Deliverables due Friday:

- robust exp_003 JSON/CSV with at least 50 valid prompts
- method comparison markdown
- 2-3 figures
- one findings note
- one ESA-facing summary note

## Claims policy for Friday package

Allowed:

- semantic/vectorized preprocessing affects frontier stability in GPT-2 Small
- one method appears more stable in current setup
- layer sweep either reinforces or weakens the Layer 6 hypothesis
- Phi-2 gives a preliminary control outcome

Not allowed:

- semantic compression proven in general
- `risk_refined` is batch-robust classifier
- results generalize across models

## Post-week handoff

If the week closes successfully, the next phase should shift from setup-cleaning to model scaling.

The default next research step is:

- larger-model microscopy beyond GPT-2 Small and light Phi-2 control

The purpose of that phase is:

- to test whether the current residual-state picture survives in more capable models
- to determine which signals remain stable as model size and representational richness increase
- to separate small-model artifacts from microscopy findings that are likely to generalize

This post-week scaling phase should begin only after:

- the robust `exp_003` batch is complete
- the Layer 6 replication pass is complete
- the multi-prompt steering pass is complete
- the Phi-2 light control is recorded
- the current claim boundary is written clearly
