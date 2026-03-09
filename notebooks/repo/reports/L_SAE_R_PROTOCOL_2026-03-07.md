# L-SAE+R Protocol

Date: 2026-03-07
Track: `ai_microscopy`
Scope: `ESA/research/mechanistic-interpretability/`
Protocol state at initialization: `QUESTION -> PROTOCOL`

## Purpose

Define the first explicit `L-SAE+R` protocol as the next microscopy item after the unified observability baseline.

`L-SAE+R` means:

- `L-SAE`: a lens-supervised sparse autoencoder
- `+R`: evaluated through the existing recorder stack

This protocol should test whether decision-relevant supervision improves sparse-feature utility without destroying microscope value.

## Shared evaluation substrate

This protocol must reuse the same material as the unified stack:

- canonical prompt panel: `data/prompts_observability_panel_2026-03-07.jsonl`
- current layer selections used for microscopy comparison
- same intervention labels
- same recorder schema and drift fields

This is required so that plain SAE, lens-only, and `L-SAE+R` can be compared honestly.

## Model variants

The initial comparison set is:

1. `Plain SAE`
2. `Lens-only`
3. `L-SAE v1`
4. `L-SAE+R`

`L-SAE+R` is not a separate model. It is the `L-SAE` variant evaluated through the existing recorder and frontier stack.

## Minimal training objective

Start with the smallest valid supervised extension:

- keep the standard residual reconstruction objective
- add one logit-lens supervision term
- do not add frontier metrics to the training loss in v1

The purpose of v1 is interpretability and comparison, not maximum predictive performance.

## Recommended sequence

1. establish plain SAE and lens-only baselines on the current panel
2. test whether SAE latents can predict lens behavior with simple probes
3. train `L-SAE v1`
4. run recorder traces on the same panel and layers
5. compare `Plain SAE` vs `Lens-only` vs `L-SAE+R`

## Required recorder additions

The shared trace schema should gain only the minimum extra fields needed for the comparison:

- `model_variant`
- `lsae_loss`
- `reconstruction_error`
- `lens_alignment_score`

Any further fields should be justified by direct comparison needs.

## Outputs

This protocol should produce:

- one comparison report: `plain SAE vs lens vs L-SAE+R`
- one figure set for feature drift, lens drift, and frontier behavior by model variant
- one updated claim-boundary note

## Success criteria

The protocol is successful if:

- all model variants are run on the same material
- at least one decision-relevant metric improves over plain SAE
- the result remains interpretable through the current recorder stack
- the protocol yields a clean null result if no gain appears

## Claim boundary

Allowed if evidence supports it:

- `L-SAE+R` changes the relationship between sparse features and decision-relevant telemetry
- one or more decision-relevant metrics improve over plain SAE in the current GPT-2 setup

Not allowed without stronger replication:

- `L-SAE+R` is the correct general sparse-feature objective
- lens supervision is broadly superior across models
- `L-SAE+R` already solves regime detection
