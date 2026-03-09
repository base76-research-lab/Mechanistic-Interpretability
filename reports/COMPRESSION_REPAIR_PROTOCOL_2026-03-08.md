---
autonomy_level: semi-autonomous
principle_ref: RCP-2026-01
last_human_review: null
artifact_status: agent-drafted
internal_only: true
---

# Compression Repair Protocol

Date: 2026-03-08
Track: `ai_microscopy`
State: `PROTOCOL_DRAFT`
Evidence level at initialization: `Exploratory`

## Purpose

Repair the text-compression path so that it:

- produces real token reduction
- preserves prompt meaning and conditional structure
- minimizes decision drift relative to the raw prompt

This protocol is narrower than the broader vectorized-compression experiment. It treats text compression as a standalone engineering and observability problem before any vector proxy is considered legitimate.

## Why this protocol exists

The Monday sanity pass on `2026-03-08` established that:

- the compression integration now loads and runs
- the current compression path does not yet yield a defensible structure-preserving method
- several prompts return `raw_fallback`
- some prompts expand rather than compress
- the current summary logic masks negative compression by clamping `tokens_saved` at zero

This means the next step is not broader batch analysis. The next step is repair and validation of the compression path itself.

## Core questions

1. Why do many prompts return `raw_fallback` rather than valid compression?
2. Under what prompt types does the current compressor expand the prompt instead of reducing it?
3. Can the text-only compressed prompt preserve decision behavior better than the current raw baseline within acceptable thresholds?

## Scope

In scope:

- compressor availability and preflight behavior
- `compressed` text variant only
- negative compression visibility
- conditionality preservation
- prompt-type-specific compression behavior
- structure-preserving validation against raw

Out of scope:

- vectorized proxy legitimacy
- cross-model generalization
- external packaging
- claim strengthening beyond the current GPT-2 Small setup

## Required fixes before evaluation

The evaluation loop should first correct measurement so the failure modes are visible.

### A. Compression accounting

The analysis layer must distinguish:

- `compressed_shorter`
- `raw_fallback`
- `compressed_longer`

The protocol should stop masking negative token savings. If a compressed prompt becomes longer, that should be recorded explicitly rather than clipped to zero.

Recommended fields:

- `token_delta`
- `compression_outcome`
- `compression_effective`

### B. Prompt preservation review

For each sanity prompt, store:

- raw prompt
- compressed prompt
- compression mode
- token delta
- top-1 match vs raw
- top-k overlap vs raw

This is required so the human reviewer can inspect whether the compressor is preserving the operative content.

## Evaluation panel

Use the current sanity panel first:

- `data/prompts_sanity_2026-03-09.txt`

Then, only if the repair loop improves outcomes:

- `data/prompts_robust_2026-03-09.jsonl`

## Success criteria

Compression is repair-ready for robust batch only if:

- at least one non-raw compressed method produces genuine positive median token reduction
- no prompt-type family is dominated by `compressed_longer`
- fallback remains low enough to avoid masking the method entirely
- top-k overlap and top-1 behavior remain within a defensible claim boundary for the current setup
- the best current label is no longer `None`

## Failure categories

Use these categories during repair:

- `raw_fallback dominance`
- `compression expansion`
- `decision drift`
- `semantic loss`
- `conditionality loss`
- `measurement masking`

## Immediate next actions

1. expose negative token deltas instead of clamping them to zero
2. write a per-prompt inspection table for raw vs compressed
3. isolate which prompts are `raw_fallback`
4. isolate which prompts become longer
5. rerun sanity pass before any robust batch

## Claim boundary

Allowed after this protocol if evidence supports it:

- the current compressor does or does not achieve real token reduction in the GPT-2 Small sanity setup
- some prompt classes are more compressible than others
- some prompt classes trigger fallback or expansion

Not allowed:

- general semantic compression claims
- cross-model compression claims
- claims that compression is production-ready

