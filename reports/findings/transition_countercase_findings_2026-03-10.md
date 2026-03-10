# Transition Counter-Case Findings — 2026-03-10

Date: 2026-03-10
Owner: Bjorn / Base76

## Goal

Test whether the `q90` token-level lead-time triggers in `transition` and regime-adjacent prompts are
best interpreted as genuine hallucination-like onset or as a distinct ambiguity boundary.

## Setup

- Source run: `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/`
- Input artifacts:
  - `analysis/lead_time/per_token_summary.csv`
  - `analysis/lead_time/prompt_lead_time_operational_q90.csv`
  - `analysis/lead_time/prompt_lead_time_conservative_q95.csv`
- Counter-case analysis output:
  - `analysis/transition_countercase/summary.json`
  - `analysis/transition_countercase/regime_countercase_summary.csv`
  - `analysis/transition_countercase/detected_prompt_profiles_q90.csv`
  - `analysis/transition_countercase/detected_prompt_profiles_q95.csv`
  - `analysis/transition_countercase/transition_countercase_scatter.png`

## Results

- `q90` detected prompts:
  - `transition`: `3`
  - `hallucination_prone`: `2`
  - `reasoning`: `1`
- `q95` detected prompts:
  - `hallucination_prone`: `2`
  - `transition`: `0`
  - `reasoning`: `0`
- Median post-onset persistence among `q90` detections:
  - hallucination-prone: `0.75`
  - transition: `0.333`
  - reasoning: `0.4`
- Median lead tokens among `q90` detections:
  - hallucination-prone: `2`
  - transition: `11`
  - reasoning: `4`
- Median tail score delta among `q90` detections:
  - hallucination-prone: `-1.751`
  - transition: `-1.955`
  - reasoning: `-1.983`
- Summary verdict:
  - `transition_countercases_differ_by_persistence_but_remain_boundary_cases`

## Interpretation

- The `transition` detections are not best read as simple duplicates of the hallucination slice.
- Their main difference is not onset score alone, but what happens after onset:
  - hallucination-prone detections persist longer after onset
  - transition detections are earlier in longer prompts, but decay faster and do not survive the
    conservative threshold
- This supports a two-stage reading of the current lead-time signal:
  - onset-like geometry excursions can appear in transition prompts
  - sustained persistence is more characteristic of the current hallucination-prone slice
- The counter-case pass therefore sharpens the ambiguity boundary, but does not eliminate it.

## Threats to validity

- The counter-case set is very small: `3` transition prompts, `2` hallucination prompts, `1`
  reasoning prompt at `q90`.
- The comparison still depends on the present token-delta construction and threshold scheme.
- Transition prompts are intentionally mixed and long-form, so lead length is partly shaped by prompt
  length.

## Claim boundary

Allowed:

- current transition counter-cases differ from the hallucination slice mainly by lower
  post-onset persistence
- conservative thresholding isolates the present hallucination-prone slice without transition or
  reasoning detections
- transition remains a real ambiguity boundary rather than resolved noise

Not yet allowed:

- a claim that transition is now solved as a category
- a broad early-warning detector claim
- a claim that persistence alone fully separates stable complexity from hallucination

## Next step

Write the Friday internal review pack and decide whether the GPT-2 evidence is now strong enough to
justify one control-model block.
