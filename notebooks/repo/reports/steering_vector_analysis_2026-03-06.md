# Steering Vector Analysis

Date: 2026-03-06
Source: Google Colab notebook run (`ASE_phi2.ipynb`)

## Experiment overview

A steering vector was derived at Layer 6 by subtracting the hallucination activation state from the reasoning activation state:

- steering vector = `Reasoning - Hallucination`

## Key values

- layer: `6`
- steering vector L2 norm: `64.18`
- baseline hallucination entropy: `7.20`
- steered entropy at coefficient `+1.0`: `3.53`

## Interpretation

Injecting the steering vector causally modulated the model behavior in the intended direction.

The reported effect was:

- lower entropy
- sharper output distribution
- movement away from the hallucination-prone continuation

This is important because it is an intervention result rather than a purely observational one.
