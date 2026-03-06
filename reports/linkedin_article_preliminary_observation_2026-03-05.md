# Preliminary observation: two different high-entropy states in LLM inference

Senast uppdaterad: 2026-03-05

Over the last few days, we have explored a simple but reproducible signal in mechanistic interpretability: the relationship between latent state geometry and the candidate token frontier immediately before next-token selection.

If this distinction holds across models, it may provide a simple observability signal for detecting hallucination-prone states before token generation.

By *candidate frontier*, we mean the set of highest-probability tokens the model is considering before selecting the next token.

Measurements are based on SAE projections of the residual stream and analysis of top-k token candidates prior to unembedding.

In an initial GPT-2 setup, we observe a clear distinction between two cases that both show high entropy:

## Reasoning state

Entropy is high and state-token misalignment is present, but the top-k candidates remain semantically coherent.

Example (analogy prompt): `man`, `queen`, `woman`, `king`, `wife`, `mother`.

Interpretation: the model appears to be in an operator-first regime, where the relation is active but the final operand has not yet collapsed.

## Hallucination-prone state

Entropy remains high.

However the candidate frontier collapses into generic fallback tokens:

`The`, `I`, `And`, `It`, punctuation, whitespace.

Interpretation: the latent state appears to move far from regions that map to meaningful token candidates.

## Concrete snapshot

| scenario | gap | \|coords\| | degeneracy_ratio |
|---|---:|---:|---:|
| reasoning | 3.95 | 3.76 | 0.10 |
| hallucination | 6.81 | 6.68 | 1.00 |

Same model family and setup; values are from the current preliminary run set and will be re-tested on larger benchmarks.

## Working hypothesis

A more precise formulation may be:

hallucination =
high entropy
+ extreme state-token misalignment
+ degenerate candidate frontier

rather than:

hallucination = high entropy

## Observability additions

To test this systematically, we extended the inference observability pipeline with candidate-front metrics:

- candidate coherence
- candidate variance
- degeneracy ratio
- state-to-centroid distance

These are logged per run alongside entropy and state-space coordinates.

## Current limitations

This is still preliminary:

- small prompt set
- single-model initial experiments
- thresholds not yet calibrated on larger benchmarks

## Next verification steps

- 50-100 prompt benchmark (facts / reasoning / hallucination)
- layer sweep
- cross-model replication (GPT-2 -> Phi-2)

If you work on interpretability, calibration, or hallucination detection, I am happy to discuss protocol and replication design.
