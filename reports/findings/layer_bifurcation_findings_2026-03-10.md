# Layer Bifurcation Findings — 2026-03-10

Date: 2026-03-10
Owner: Bjorn / Base76

## Goal

Locate where hallucination-prone trajectories diverge most strongly from anchored, reasoning, and
transition trajectories, and determine whether `Layer 6` should remain the main bifurcation layer.

## Setup

- Model: `gpt2`
- Layer(s): `3 5 6 9 12`
- Dataset / prompts: `data/prompts_observability_panel_2026-03-07.jsonl`
- Method: read-only oscilloscope trajectory analysis on shared SAE subspace
- Params: SAE `exp_001_sae_v3`, basis `pc2`, units `472 468 57 156 346`
- Artifacts written to:
  - `experiments/exp_005_trajectory_block/analysis_2026-03-10/bifurcation/`
  - observer/control traces as in the trajectory block protocol

## Results

- Strongest divergence layer: `6`
- Composite divergence ranking:
  - Layer `6`: `0.734`
  - Layer `5`: `0.151`
  - Layer `3`: `-0.116`
  - Layer `9`: `-0.298`
  - Layer `12`: `-0.471`
- Block conclusion: `Layer 6 remains strongest local bifurcation candidate`

## Interpretation

- In the current GPT-2 panel, the strongest separation between hallucination-prone and non-
  hallucination trajectories occurs at `Layer 6`, not across a broad `L6-L9` band.
- Layer `5` shows some preparatory shift, but the main break still concentrates at `6`.
- The present block therefore narrows the transition picture relative to the earlier oscilloscope
  suggestion of a broader `L6-L9` zone.

## Threats to validity

- The layer set is sparse: `3 5 6 9 12`
- A denser sweep between `6` and `9` could still reveal a broader band that the current block does
  not resolve
- The conclusion is panel-dependent and should not yet be generalized beyond the current setup

## Claim boundary

Allowed:

- `Layer 6` is the strongest current local bifurcation candidate in the active GPT-2 panel

Not yet allowed:

- `Layer 6` is universally privileged across models
- `L6-L9` is ruled out in general rather than only deprioritized in this block

## Next step

Run a denser read-only sweep around `Layers 5-9` before freezing the final bifurcation narrative.
