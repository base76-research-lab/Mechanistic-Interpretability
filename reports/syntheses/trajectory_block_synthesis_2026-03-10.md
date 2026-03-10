# Trajectory Block Synthesis — 2026-03-10

Date: 2026-03-10
Scope: first GPT-2 trajectory block integrating detection, bifurcation, regime stability, and dense
layer follow-up

## Supported now in GPT-2

- A geometry-driven detection score separates hallucination-prone prompts better than entropy alone
  at the current full-panel level.
- `Layer 6` remains the strongest current local bifurcation candidate.
- A denser `5-12` sweep separates early divergence from later expansion: strongest local onset at
  `Layer 6`, largest hallucination expansion at `10 -> 11`.
- Regime stability fingerprints are distinct at regime level in the current setup.
- Read-only oscilloscope remains the primary observer surface, with unified no-write-back baseline
  compatible and write-back explicitly interventional.

## Exploratory only

- How stable the detection score remains under panel expansion or prompt perturbation
- Whether the `10 -> 11` expansion finding is stable under panel expansion or a prompt artifact
- Which subset of stability metrics should become canonical for runtime monitoring
- Whether `phase velocity` adds lead time beyond geometry score components

## Blocked claims

- Cross-model generalization
- Production-ready early warning system
- Attractor-level dynamics claims

## Block interpretation

The first trajectory block supports a coherent microscopy picture:

- hallucination-prone prompts occupy a more unstable geometric regime than anchored and reasoning
  prompts
- the strongest divergence still concentrates at `Layer 6`
- the strongest later expansion is now resolved at `10 -> 11`, which should be treated as a later
  regime-acceleration event rather than the initial onset
- regime-level trajectory structure is strong enough to justify continued GPT-2 block work before
  moving to a cross-model phase

## Next transition

Recommended next block:

1. expand the prompt panel for detection validation
2. repeat the dense `5-12` sweep on the expanded panel
3. add token-level lead-time analysis before any early-warning claim is strengthened
