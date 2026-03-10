# Friday Internal Review Pack — 2026-03-13

Prepared: 2026-03-10
Review date: 2026-03-13
Scope: GPT-2 trajectory validation block only
Decision target: `go / no-go` for one small control-model block next week

## Purpose

This pack is the canonical Friday review surface for the current GPT-2 validation week.

It exists to force a narrow decision from artifacts rather than memory:

- is the current GPT-2 onset picture strong enough to retain as `Supported`
- is the ambiguity boundary characterized tightly enough to justify one control-model block
- which claims remain blocked even if the answer is `go`

## Governing claim boundary

This review is about:

- onset localization in the current GPT-2 Small setup
- later expansion dynamics in the current GPT-2 Small setup
- whether geometry and persistence together form a defensible observability signal

This review is not about:

- proving a general hallucination detector
- claiming production-ready early warning
- claiming cross-model replication before a control-model block exists

## Canonical evidence bundle

Read in this order:

1. `reports/syntheses/current_trajectory_findings_2026-03-10.md`
2. `reports/findings/expanded_panel_validation_findings_2026-03-10.md`
3. `reports/findings/token_level_lead_time_findings_2026-03-10.md`
4. `reports/findings/transition_countercase_findings_2026-03-10.md`

Primary artifact bundle:

- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/detection/summary.json`
- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/bifurcation/summary.json`
- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/stability/summary.json`
- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/lead_time/summary.json`
- `experiments/exp_005_trajectory_block/expanded_panel_dense_2026-03-10/analysis/transition_countercase/summary.json`

## Supported now in GPT-2

- Geometry remains more informative than entropy at panel level on the expanded GPT-2 panel:
  - geometry AUC `0.603`
  - entropy AUC `0.423`
- `Layer 6` remains the strongest current local onset candidate.
- `10 -> 11` remains the strongest current later expansion step.
- Regime stability fingerprints remain distinct at regime level.
- Token-level geometry delta remains more informative than entropy delta overall:
  - geometry delta AUC `0.545`
  - entropy delta AUC `0.384`
- A conservative token-level slice exists:
  - `2/6` hallucination-prone prompts detected
  - median lead time `2.0` tokens
  - zero transition detections at `q95`
- Transition counter-cases are not identical to the hallucination slice:
  - median post-onset persistence `0.75` for hallucination-prone
  - median post-onset persistence `0.333` for transition

## Exploratory only

- Broad `q90` thresholding remains non-specific:
  - `3` transition detections
  - `1` reasoning detection
- Transition remains an ambiguity boundary rather than a solved category.
- Persistence currently looks like the strongest disambiguator, but it has only been shown in a
  small detected slice.
- The present lead-time slice is sparse; coverage is too low for strong recall claims.

## Blocked claims

- Cross-model generalization
- Production-ready early warning
- Clean detector claim against transition prompts
- Universal onset or commit layer
- Attractor-level claims

## Decision frame for Friday

### `Go` for one small control-model block if all remain true

- geometry still beats entropy in the locked expanded-panel GPT-2 artifacts
- `Layer 6` still stands as the strongest current local onset candidate
- `10 -> 11` still stands as the strongest later expansion step
- conservative token-level lead-time slice remains intact
- transition counter-cases remain lower in persistence than the hallucination slice
- no new artifact undermines the observer/intervention boundary

### `No-go` if any of these collapse

- the expanded-panel evidence is found to be internally inconsistent
- persistence does not actually separate the transition counter-cases from the hallucination slice
- the token-level slice is judged too brittle to count as meaningful support
- the current findings look too panel-sensitive to justify even one control-model step

## Recommended Friday outcome logic

### Preferred outcome

`Go` for one small control-model block with hard constraints:

- one model only
- reuse the same panel taxonomy where possible
- preserve observer/intervention separation
- treat the first control-model pass as a control, not a replication claim

### Conservative fallback

If the review judges the token-level slice too sparse, remain in GPT-2 and tighten:

- transition counter-case coverage
- persistence formulation
- prompt-panel breadth

## Proposed control-model shape if Friday says `go`

Use one small control model only, selected for practical reproducibility rather than frontier scale.

Control-block requirements:

- same observability panel taxonomy
- same dense layer logic, adapted to model depth
- same separation between read-only observer and intervention runs
- no public claim beyond `initial control-model check`

## Decision record

Friday reviewer should fill this section:

- Decision: `GO` / `NO-GO`
- Why:
- Main retained supported claims:
- Main blocked claims:
- Next week target:
