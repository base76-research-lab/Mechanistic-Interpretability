---
autonomy_level: semi-autonomous
principle_ref: RCP-2026-01
last_human_review: null
artifact_status: agent-drafted
internal_only: true
---

# Vector Proxy Legitimacy Protocol

Date: 2026-03-08
Track: `ai_microscopy`
State: `PROTOCOL_DRAFT`
Evidence level at initialization: `Exploratory`

## Purpose

Determine whether the current `compressed_vectorized_proxy_*` variants are legitimate as:

- prompt replacements
- diagnostic representations
- or neither

This protocol treats vector proxy legitimacy as a research question distinct from text compression repair.

## Why this protocol exists

The latest sanity run showed:

- near-zero top-k overlap for most vector proxy variants
- zero top-1 agreement in many prompts
- anchor packets populated with tokens that are embedding-near but inferentially poor
- evidence that the proxy may be destroying decision-relevant structure even when the compressed text itself is acceptable

This means the current proxy should not be treated as a valid prompt substitute by default.

## Core questions

1. Does the external vectorization preserve any useful neighborhood relation to the model's own internal spaces?
2. Does the proxy preserve decision-relevant structure better as an analysis representation than as a prompt replacement?
3. Are the current anchor packets dominated by generic, subword-like, or task-irrelevant tokens?

## Scope

In scope:

- `compressed_vectorized_proxy_mean`
- `compressed_vectorized_proxy_attn_weighted`
- `compressed_vectorized_proxy_pca1`
- external vector vs internal LLM representation comparison
- anchor quality inspection
- neighborhood and alignment checks

Out of scope:

- production use of vector proxy prompts
- claims that the proxy is already a viable compressed representation
- broader L-SAE+R conclusions

## Working distinction

This protocol separates three possible roles:

1. `Prompt replacement`
   - the vector proxy can replace prompt text directly
2. `Diagnostic representation`
   - the vector proxy is useful for analysis but not for generation
3. `Invalid representation`
   - the vector proxy does not preserve enough structure for either use

The current default assumption should be:

- not legitimate as prompt replacement until shown otherwise

## Required analyses

### A. Anchor quality inspection

For each prompt, record:

- compressed text
- vector method
- anchor tokens
- anchor stopword ratio
- obvious artifact tokens
- top-1 match vs raw
- top-k overlap vs raw

This should identify whether the anchor selection step is dominated by:

- generic high-frequency tokens
- subword fragments
- semantically distant nearest neighbors

### B. External vs internal comparison

Compare the external vector proxy representation against:

- input embedding mean
- attention-weighted embedding mean
- early residual states
- the current microscopy control layer where feasible

Recommended metrics:

- cosine similarity
- nearest-neighbor preservation
- prompt-pair neighborhood consistency
- alignment with frontier metrics

### C. Prompt replacement test

Only after anchor quality is understood, evaluate whether proxy prompts can preserve:

- top-1 candidate identity
- top-k overlap
- rank correlation
- coherence
- gap behavior

## Success criteria

The proxy is legitimate as prompt replacement only if:

- top-k overlap rises materially above the current near-zero baseline
- top-1 agreement is no longer systematically lost
- anchor packets stop being dominated by irrelevant or artifact tokens
- at least one vector method shows stable alignment with raw behavior

The proxy may still be useful as a diagnostic representation if:

- it aligns with internal embedding or residual geometry
- but still fails as a direct prompt replacement

## Immediate next actions

1. add per-prompt anchor inspection output
2. compare external proxy vectors with model-internal representations
3. classify the proxy as:
   - `prompt_replacement_candidate`
   - `diagnostic_only`
   - `invalid`
4. do not route vector proxy variants into robust batch as prompt substitutes until this protocol passes

## Claim boundary

Allowed after this protocol if evidence supports it:

- the current vector proxy is not a legitimate prompt replacement in the GPT-2 Small setup
- one proxy method may still be useful as an analysis representation
- anchor quality is a major determinant of proxy behavior

Not allowed:

- claims that vectorized conditioning is validated in general
- claims that external vectorization preserves internal decision geometry across models

