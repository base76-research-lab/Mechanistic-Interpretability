# Expanded Observability Panel — 2026-03-10

Canonical file:

- `prompts_observability_panel_expanded_2026-03-10.jsonl`

## Purpose

This panel expands the original observability panel without replacing any existing prompts.

The goal is to make the next GPT-2 validation block harder in three specific ways:

- more anchored and reasoning references for cleaner same-regime comparison
- more transition and regime-adjacent prompts for boundary stress
- more hallucination-prone prompts that still look superficially answerable

## Design rules

- all original prompt IDs are preserved unchanged
- new prompts are additive only
- taxonomy remains compatible with the current analysis stack:
  - `anchored`
  - `reasoning`
  - `transition`
  - `hallucination_prone`
  - `control`
- controls remain under `random_prompt_baseline`

## Added prompt groups

- `anchored_04-06`
  - add factual, arithmetic, and lexical anchors
- `reasoning_04-06`
  - add analogy and reproducibility-planning references
- `transition_04-07`
  - add noisier, regime-adjacent boundary cases
- `hallucination_04-06`
  - add more impossible-but-plausible factual prompts
- `random_baseline_*_02`
  - add matched random baselines for new anchored / transition / hallucination cases

## Counts

- original panel size: `16`
- expanded panel size: `32`

## Intended next use

This panel is the intended input for the next dense GPT-2 validation block in:

- `reports/internal_ops/ESA_WEEK_PLAN_2026-03-10_GPT2_VALIDATION.md`
