# ESA Week Plan: 2026-03-10 to 2026-03-13

Goal: validate the current GPT-2 trajectory onset picture hard enough to support Friday internal
review.

## Canonical weekly status

This is the canonical week plan for the active research week.

It supersedes the older `exp_003`-driven week logic as the active execution plan.

Primary model: `gpt2`
Scope: GPT-2 validation only
Friday endpoint: `internal review pack`
Non-goal for this week: cross-model replication

Governing claim boundary:

- this week is about onset localization and trajectory validation in the current GPT-2 setup
- this week is not about proving a general hallucination detector
- read-only observer traces remain the only valid onset evidence surface

## Locked weekly hypothesis

The active hypothesis to validate this week is:

- early local divergence onset at `Layer 6`
- later expansion at `10 -> 11`
- geometry remains more informative than entropy
- observer traces and write-back traces must remain separate evidence classes

## Locked defaults

Use these defaults across all evidence-bearing runs this week:

- model: `gpt2`
- observer surface: read-only oscilloscope
- comparison surfaces:
  - unified baseline without write-back
  - write-back only as intervention/control
- layer sweep: `5 6 7 8 9 10 11 12`
- panel base: `data/prompts_observability_panel_2026-03-07.jsonl`
- SAE state: `experiments/exp_001_sae_v3/sae_weights.pt`
- basis: `pc2`
- units: `472 468 57 156 346`

Minimum output set for the week:

- one expanded panel file
- one dense block manifest
- one read-only trace
- one detection summary
- one bifurcation summary
- one stability summary
- one lead-time summary
- one dense validation findings note
- one Friday internal review synthesis note

## Tuesday: panel expansion and run lock

Purpose:

- formalize an expanded GPT-2 panel around the current observability panel
- preserve prompt taxonomy and IDs well enough for reruns and comparisons

Tasks:

- create an expanded panel that keeps existing strata and adds:
  - harder transition prompts
  - regime-adjacent prompts
  - more difficult anchored vs hallucination-prone boundaries
- document panel rules:
  - prompts keep stable IDs
  - new prompts are additive, not replacements
  - taxonomy remains compatible with current analysis code
- lock the dense block command and output paths for the week

Deliverables by end of Tuesday:

- expanded panel file in `data/`:
  - `data/prompts_observability_panel_expanded_2026-03-10.jsonl`
- short panel-definition note if schema or taxonomy changes need explanation
- one exact dense rerun command recorded in the working note or run plan

## Wednesday: dense rerun on expanded panel

Purpose:

- test whether the current onset picture survives stronger GPT-2 coverage

Run order:

1. read-only dense sweep on the expanded panel
2. unified baseline without write-back
3. write-back intervention control
4. trajectory block analysis bundle

Required outputs:

- dense block manifest
- read-only trace
- detection summary
- bifurcation summary
- stability summary

Interpretation rule:

- no onset claim may be written from write-back traces
- the read-only trace remains the primary evidence surface

## Thursday: lead-time and counter-case pass

Purpose:

- characterize how early the geometry signal moves before output collapse
- inspect whether transition prompts remain the main ambiguity boundary

Tasks:

- add token-level lead-time analysis on top of the read-only dense trace
- measure:
  - earliest stable divergence token
  - divergence lead time before bad output or collapse
  - whether the signal survives on the harder transition prompts
- write one short counter-case note if transition/regime-adjacent prompts still blur the boundary

Required outputs:

- one lead-time summary artifact
- one short transition ambiguity / counter-case note if needed
- one dense validation findings note tying together rerun + lead-time

## Friday: internal review pack

Purpose:

- force a go / no-go decision before any model expansion

The Friday internal review pack must answer all of these:

- does geometry still beat entropy on the expanded GPT-2 panel?
- does `Layer 6` remain the strongest local onset candidate?
- does `10 -> 11` remain the strongest later expansion step?
- is there measurable token-level lead time before hallucination-prone output collapse?
- are transition prompts still the main ambiguity boundary?
- should next week begin with one small control model, or is GPT-2 still too unstable?

The Friday review note must separate:

- `Supported now in GPT-2`
- `Exploratory only`
- `Blocked claims`
- `Go / no-go for cross-model control`

Canonical Friday review surface:

- `reports/internal_ops/FRIDAY_INTERNAL_REVIEW_PACK_2026-03-13.md`

## Acceptance criteria

The week is only complete if:

- the old weekly plan is clearly superseded by this one
- this plan remains the only canonical weekly execution plan
- repo planning and Research-OS point to the same active week program
- the Friday review can be written from artifacts rather than chat memory
- cross-model work is still explicitly deferred unless Friday says `go`

## Deferred this week

Do not treat these as active week goals:

- cross-model replication
- ESA-facing packaging
- broad compression-method work as the main program
- production-grade early-warning framing

## Next transition after Friday

If the internal review passes:

- start one light control-model block next week

If the internal review fails:

- stay in GPT-2 and repeat validation with a stronger panel or tighter lead-time analysis
