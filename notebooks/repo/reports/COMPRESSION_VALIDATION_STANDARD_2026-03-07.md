# Compression Validation Standard

Date: 2026-03-07
Track: `ai_microscopy`
Scope: `mechanistic-interpretability`

## Purpose

Define how compression methods are evaluated in the microscopy track.

The main objective is not “compress as much as possible.” The main objective is to identify the current best structure-preserving compression method in the current setup.

## Canonical standard

Compression methods must be judged on four axes:

1. structural fidelity
2. behavioral fidelity
3. compression efficiency
4. robustness

## Winner definitions

Every run family should define:

- `Best structure-preserving method`
- `Best balanced method`
- `Most compressive method`

For this track, the primary winner is:

- `Best structure-preserving method`

## Claim boundary

Allowed:

- a method is the best current structure-preserving option in the current GPT-2 Small setup
- a method beats text-only compression on specific structure metrics in the current panel

Not allowed:

- a method is globally optimal
- token savings alone justify the method
- the current winner generalizes beyond the current panel, layer choice, or model family

## Reporting rule

Compression reports must include:

- same-material comparison
- median + IQR
- regime-wise breakdown
- pass/fail
- failure-mode diagnosis

## Current canonical artifacts

- protocol: `../../experiments/protocols/vectorized_compression_experiment_v1.md`
- comparison runner: `../scripts/compare_compression_vectorized.py`
