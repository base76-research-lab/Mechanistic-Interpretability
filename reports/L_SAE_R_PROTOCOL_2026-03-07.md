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

The protocol also includes a secondary contrast panel for testing whether the stack separates:

- `math_reasoning`
- `factual_recall`
- `degeneracy_probe`
- `random_prompt_baseline`

## Shared evaluation substrate

This protocol must reuse the same material as the unified stack:

- canonical prompt panel: `data/prompts_observability_panel_2026-03-07.jsonl`
- current layer selections used for microscopy comparison
- same intervention labels
- same recorder schema and drift fields

This is required so that plain SAE, lens-only, and `L-SAE+R` can be compared honestly.

Within the canonical panel, the contrast panel should be treated as an orthogonal evaluation slice rather than a replacement for the current regime split.

`random_prompt_baseline` should be treated as a matched methodological control, not as a new regime class. Its purpose is to test whether apparent separation is driven by real cognitive structure rather than prompt length, token-shape effects, or shallow syntax patterns.

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

The protocol should also use the existing trace data to derive:

- `decision_trajectory_smoothness`

where:

- `DTS = Σ |lens_entropy(layer_i+1) − lens_entropy(layer_i)|`

Lower DTS indicates a smoother or flatter decision trajectory. Higher DTS indicates a jumpier or more unstable trajectory. DTS measures trajectory volatility, not correctness by itself.

The contrast panel should also use matched random baselines to test:

- whether `math_reasoning` still separates from a length- and shape-matched control
- whether `factual_recall` still separates from a fact-shaped but semantically degraded control
- whether `degeneracy_probe` differs from random incoherence, or whether both are mostly capturing shallow disorder

## Outputs

This protocol should produce:

- one comparison report: `plain SAE vs lens vs L-SAE+R`
- one figure set for feature drift, lens drift, and frontier behavior by model variant
- one class-level comparison slice for `math_reasoning`, `factual_recall`, `degeneracy_probe`, and `random_prompt_baseline`
- one updated claim-boundary note

## Success criteria

The protocol is successful if:

- all model variants are run on the same material
- at least one decision-relevant metric improves over plain SAE
- the result remains interpretable through the current recorder stack
- the protocol yields a clean null result if no gain appears
- the stack can test whether DTS helps separate `math_reasoning`, `factual_recall`, and `degeneracy_probe`
- the stack can test whether observed separation survives comparison against `random_prompt_baseline`

Expected exploratory pattern for the contrast panel:

- `math_reasoning`: gradual multi-layer change and moderate DTS
- `factual_recall`: flatter or early-stable trajectory and lower DTS
- `degeneracy_probe`: larger jumps or irregular changes and higher DTS
- `random_prompt_baseline`: degraded semantic structure that helps test whether apparent separation is merely surface-driven

## Claim boundary

Allowed if evidence supports it:

- `L-SAE+R` changes the relationship between sparse features and decision-relevant telemetry
- one or more decision-relevant metrics improve over plain SAE in the current GPT-2 setup

Not allowed without stronger replication:

- `L-SAE+R` is the correct general sparse-feature objective
- lens supervision is broadly superior across models
- `L-SAE+R` already solves regime detection
