# Findings Summary

Date: 2026-03-06
Source: Google Colab notebook run (`ASE_phi2.ipynb`)

## Scope note

The notebook was intended for Phi-2, but the completed run fell back to GPT-2 Small because of Colab resource constraints.

The findings below therefore apply to GPT-2 Small in this run.

## Main findings

- Four latent state regimes were reproduced: anchored, reasoning, transition, and hallucination-prone.
- A layer-wise sweep on a hallucination-prone prompt identified Layer 6 as the highest-risk layer in the current setup.
- The refined risk metric diverges more clearly from entropy in hallucination-prone behavior than in reasoning behavior.
- A simple compression intervention at Layer 6 destabilized hallucination-prone states more strongly than reasoning states.
- A steering vector derived from the difference between reasoning and hallucination states causally modulated the model output in the expected direction.
- On a 20-sample mini-dataset, `risk_refined` did not separate reasoning and hallucination groups reliably.

## Layer-wise refined risk

Top 5 layers by `risk_refined` from the Colab run:

| rank | layer | risk_refined | entropy_norm | gap_norm | coherence_proxy |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 6 | 0.105682 | 0.471918 | 0.337645 | 0.336755 |
| 2 | 10 | 0.092467 | 0.678459 | 0.162349 | 0.160519 |
| 3 | 7 | 0.092126 | 0.507418 | 0.261435 | 0.305535 |
| 4 | 3 | 0.077358 | 0.432774 | 0.297725 | 0.399620 |
| 5 | 9 | 0.072300 | 0.635953 | 0.121183 | 0.087668 |

Peak reported value:

- Layer 6 `risk_refined`: `0.105682`

## Risk validation

Correlation between `risk_refined` and `logit_entropy`:

- Reasoning prompt: `r = 0.9141`
- Hallucination-prone prompt: `r = 0.7218`

Interpretation:

The refined risk metric tracks entropy relatively closely in reasoning-like behavior, but diverges more clearly in hallucination-prone behavior. This suggests it captures structural conflict beyond uncertainty alone.

## Compression experiment

Compression setup used in the notebook:

- intervention layer: `Layer 6`
- method: average current residual state with previous states
- setting: `k = 2`

Observed results:

| prompt_type | compression | logit_entropy | state_norm |
| --- | --- | ---: | ---: |
| Reasoning | Baseline | 5.097636 | 459.885834 |
| Reasoning | Compressed (k=2) | 5.326714 | 446.953918 |
| Hallucination | Baseline | 6.495615 | 461.249023 |
| Hallucination | Compressed (k=2) | 6.261742 | 379.787720 |

Interpretation:

- Reasoning state norm changed modestly: `459.885834 -> 446.953918`
- Hallucination-prone state norm dropped sharply: `461.249023 -> 379.787720`

This suggests hallucination-prone states are more fragile under simple compression-like smoothing than reasoning states in the current setup.

Additional observed threshold:

- breakage threshold for the hallucination state appeared at `k = 2`
- at `k = 10`, hallucination entropy reportedly rose to `9.57`, indicating full structural decoherence while reasoning remained functional

## Steering vector intervention

The notebook summary reports a steering-vector intervention at Layer 6.

Construction:

- steering vector = `Reasoning state - Hallucination state`
- layer used: `Layer 6`
- reported vector magnitude: L2 norm `64.18`

Observed effect on a hallucination-prone prompt:

- baseline entropy: `7.20`
- steered entropy with coefficient `+1.0`: `3.53`

Interpretation:

Applying the steering vector with a positive coefficient materially sharpened the output distribution and shifted the model away from the hallucination-prone continuation. This is the strongest causal result in the current Colab run, because it goes beyond observation and demonstrates directed behavioral modulation.

## Mini-dataset validation limit

The notebook also reports a mini-dataset validation with 20 samples.

Reported group means for `risk_refined` at Layer 6:

- Hallucination: `0.062597`
- Reasoning: `0.062635`

Interpretation:

Although `risk_refined` looks useful in single-case deep dives, it did not distinguish the two categories on this short mini-dataset. That means the metric currently looks promising for mechanistic diagnosis, but not yet robust as a general classifier.

## Practical takeaway

The current Colab run strengthens the working thesis that hallucination-prone behavior is better described as a structured latent regime than as high entropy alone. At the same time, it shows that the current `risk_refined` formulation still needs recalibration before it can be treated as a reliable batch-level discriminator.
