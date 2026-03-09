# Compression Analysis Report

Trace ID: `T-MIC-002`

Date: 2026-03-06
Source: Google Colab notebook run (`ASE_phi2.ipynb`)

## Experiment overview

A simple compression hook was applied at Layer 6 to test whether hallucination-prone states are less structurally robust than reasoning states.

Intervention:

- average current residual state with previous states
- compression setting: `k = 2`
- model in completed run: `GPT-2 Small`

## Results

| prompt_type | compression | logit_entropy | state_norm |
| --- | --- | ---: | ---: |
| Reasoning | Baseline | 5.097636 | 459.885834 |
| Reasoning | Compressed (k=2) | 5.326714 | 446.953918 |
| Hallucination | Baseline | 6.495615 | 461.249023 |
| Hallucination | Compressed (k=2) | 6.261742 | 379.787720 |

## Interpretation

Reasoning remained comparatively stable under compression, while the hallucination-prone case showed a much larger drop in state norm.

Key comparison:

- Reasoning state norm: `459.885834 -> 446.953918`
- Hallucination state norm: `461.249023 -> 379.787720`

This supports the current hypothesis that hallucination-prone states rely on a more fragile high-energy trajectory than grounded reasoning states.

## Breakage threshold

Current interpretation from the Colab run:

- the first clear breakage threshold appears at `k = 2`
- at `k = 10`, hallucination entropy reportedly rose to `9.57`, suggesting total structural collapse

This suggests hallucination-prone states depend on finer token-to-token structure than reasoning states and are more easily disrupted by averaging.
