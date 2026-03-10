# Data Surface

This directory contains the prompt panels and small datasets used by the current research program.

## Current canonical prompt panels

- `prompts_observability_panel_2026-03-07.jsonl`
  - canonical observability panel for anchored, reasoning, transition, and hallucination-prone prompts
- `prompts_observability_panel_expanded_2026-03-10.jsonl`
  - expanded GPT-2 validation panel for the current dense trajectory block
- `prompts_robust_2026-03-09.jsonl`
  - robust panel for evidence-bearing compression and vectorization runs
- `prompts_traps_2026-03-09.jsonl`
  - trap and adversarial stress-test panel
- `prompts_phi2_smoke_2026-03-09.jsonl`
  - light cross-model control panel
- `prompts_sanity_2026-03-09.txt`
  - small sanity panel
- `prompts.txt`
  - legacy smaller prompt set used in earlier runs

## Data role

This is a research data surface, not a benchmark packaging layer.

Prompt panels here should be:

- named with dates when materially versioned
- stable enough to support reproducible comparison
- linked from protocols and findings when they become canonical

Panel notes:

- `prompt_panel_expanded_2026-03-10.md`
  - rationale and design rules for the expanded GPT-2 observability panel
