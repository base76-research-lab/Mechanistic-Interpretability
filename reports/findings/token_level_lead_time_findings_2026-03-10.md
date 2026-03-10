# Token-Level Lead-Time Findings — 2026-03-10

Date: 2026-03-10
Owner: Bjorn / Base76

## Goal

Test whether the expanded GPT-2 read-only trajectory block contains measurable token-level lead time
before prompt end, without reframing the result as a production early-warning system.

## Setup

- Model: `gpt2`
- Layer(s): `5 6 7 8 9 10 11 12`
- Dataset / prompts: `data/prompts_observability_panel_expanded_2026-03-10.jsonl`
- Method: token-level read-only oscilloscope analysis on the completed expanded-panel run
- Artifact root:
  - `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/lead_time/`

## Results

- Token-level delta AUC:
  - geometry delta: `0.545`
  - entropy delta: `0.384`
- Pairwise token-level delta AUC:
  - hallucination vs anchored: geometry `0.549`, entropy `0.349`
  - hallucination vs reasoning: geometry `0.560`, entropy `0.638`
  - hallucination vs transition: geometry `0.522`, entropy `0.353`
- Operational threshold (`q90` against anchored + reasoning token deltas):
  - hallucination-prone detection rate: `0.333`
  - transition detection rate: `0.375`
  - reasoning detection rate: `0.167`
- Conservative threshold (`q95` against anchored + reasoning token deltas):
  - hallucination-prone detection rate: `0.333`
  - transition detection rate: `0.000`
  - reasoning detection rate: `0.000`
  - anchored detection rate: `0.000`
  - control detection rate: `0.000`
- Conservative hallucination slice:
  - median lead time: `2.0` tokens before prompt end
  - median relative lead: `0.192`
  - median post-onset persistence: `0.75`
  - detected prompts:
    - `hallucination_01` -> onset token `8`, lead `1`
    - `hallucination_06` -> onset token `8`, lead `3`

## Interpretation

- Token-level lead time is measurable in the current GPT-2 setup, but only as a narrow slice.
- Broad thresholding leaks heavily into `transition`, so the signal is not yet specific enough for a
  general early-warning claim.
- A conservative threshold isolates a small hallucination-prone subset without transition,
  reasoning, anchored, or control detections.
- This means the current signal is useful as characterization:
  - there are cases where geometry shifts before prompt end
  - but the transition boundary is still the main obstacle to broad runtime specificity

## Threats to validity

- The lead-time result depends on the current token-delta construction and a small prompt panel.
- Prompt lengths differ substantially; lead is therefore reported both in absolute tokens and
  relative prompt position.
- The signal is still internal to GPT-2 Small and one observer stack.

## Claim boundary

Allowed:

- token-level lead time is measurable in a conservative hallucination-prone slice of the current
  expanded GPT-2 run
- geometry delta remains more informative than entropy delta at token level overall
- broad token-level lead-time thresholding remains non-specific against transition-adjacent prompts

Not yet allowed:

- production-ready early warning
- a claim that token-level onset detection is broadly clean across regime boundaries
- cross-model lead-time claims

## Next step

Run the explicit transition and regime-adjacent counter-case pass, then decide in the Friday
internal review whether the GPT-2 evidence is strong enough for one control-model block.
