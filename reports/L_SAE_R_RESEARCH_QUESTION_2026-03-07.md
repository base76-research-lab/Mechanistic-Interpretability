# L-SAE+R Research Question

Date: 2026-03-07
Track: `ai_microscopy`
Evidence level at initialization: `Exploratory`
Scope: `ESA/research/mechanistic-interpretability/`

## Research question

Does lens supervision produce sparse features that track decision-relevant residual dynamics more cleanly than a plain SAE?

## Why this matters

The current microscopy stack now provides aligned:

- residual-state telemetry
- SAE feature telemetry
- logit-lens projection
- frontier metrics
- recorder traces

This makes it possible to move beyond feature discovery and ask whether sparse features can be trained to preserve microscope value while becoming more tightly aligned with decision-relevant dynamics.

If that is possible, the microscopy track gains a stronger bridge between:

- internal representation structure
- output-readiness by layer
- regime divergence across reasoning and hallucination-prone traces

The protocol also now supports a smaller contrast slice for:

- `math_reasoning`
- `factual_recall`
- `degeneracy_probe`
- `random_prompt_baseline`

This makes it possible to ask whether the stack separates structured reasoning, stable recall, and trajectory instability rather than only broader regime classes.

It also makes it possible to ask whether any observed separation survives comparison against matched random controls rather than being explained by prompt length, token distribution, or shallow syntax patterns alone.

## Primary hypothesis

An `L-SAE` that combines residual reconstruction with logit-lens supervision will preserve useful sparse structure while improving alignment with:

- layer-level projection behavior
- frontier instability signals
- early divergence between reasoning and hallucination-prone traces

## Counter-hypothesis

Lens supervision will mostly task-shape the representation, reducing microscope value and failing to improve regime separation or recorder interpretability in a meaningful way.

## Metrics

- reconstruction error against plain SAE baseline
- lens alignment score against the logit-lens target
- regime separation on the canonical observability panel
- feature-to-frontier stability across layers
- feature drift vs logit drift ordering quality
- divergence quality between reasoning and hallucination-prone traces
- decision trajectory smoothness (`DTS`) across contrast-panel prompt classes
- separation quality relative to `random_prompt_baseline`

## Falsification criteria

The protocol should be treated as unsuccessful if one or more of the following holds:

- `L-SAE` degrades reconstruction substantially relative to plain SAE
- lens alignment improves, but regime separation does not
- lens alignment improves, but frontier or recorder interpretability becomes noisier
- sparse features become less stable or less comparable across layers
- the apparent gain disappears when evaluated on the same canonical panel and recorder schema

## Minimum success threshold

`L-SAE+R` is only interesting if it shows measurable improvement over plain SAE on at least one of:

- regime separation
- stability of feature-to-frontier relationships
- prediction of lens or frontier drift
- earlier detection of divergence between reasoning and hallucination-prone traces
- cleaner separation in trajectory volatility between reasoning, recall, and degeneracy probes
- separation that remains visible after comparison against matched random controls

## Claim boundary

Allowed at this stage:

- `L-SAE+R` is a valid microscopy research direction
- the unified stack now makes the protocol testable
- the initial claim level should remain `Exploratory`

Not allowed at this stage:

- `L-SAE+R` improves mechanistic interpretability in general
- lens supervision is already the correct sparse-feature objective
- the protocol already establishes a general hallucination mechanism
