# Model Microscopy Plan

Date: 2026-03-07
Scope: `ESA/research/mechanistic-interpretability/`

## Purpose

Define the next concrete work for the microscopy track:

- SAE as a microscope on internal geometry
- residual-state regime analysis
- layer sensitivity and bifurcation detection
- causal interventions in residual space
- vectorized conditioning as a bridge between semantic and residual representations

This is the parent research track. Hallucination is treated as one important phenomenon exposed by this microscope, not as the whole research object.

## Primary research questions

1. What stable latent/residual regimes can be measured reproducibly?
2. Which layers behave as bifurcation or control points?
3. Which interventions actually modulate model behavior causally?
4. How should vector/embedding representations be aligned to residual space?
5. Which microscopy findings generalize beyond one prompt or one model?

## Current strongest signals

- Layer 6 appears to be a critical layer in the current GPT-2 setup
- steering vectors can causally modulate a hallucination-prone continuation
- compression-like smoothing exposes structural fragility in some states
- `gap`, `coherence`, and `degeneracy` behave like meaningful geometry signals
- vectorized conditioning is promising but not yet validated robustly

## Workstreams

## 1. Regime mapping

Goal:
- make the regime taxonomy more stable and less anecdotal

Tasks:
- reproduce anchored / reasoning / transition / hallucination-prone regimes on a controlled prompt panel
- define minimum regime evidence in terms of metrics and figures
- document where regime transitions emerge across layers, not only at the final output

Deliverables:
- one short regime note
- one updated triage figure
- one table of regime signatures

## 2. Layer microscopy

Goal:
- identify where the model structure changes meaningfully

Tasks:
- run layer sweeps on selected prompts across multiple layers
- compare `state_norm`, `gap`, `coherence`, `degeneracy`, and entropy
- determine whether Layer 6 is a local artifact or a recurring control layer

Deliverables:
- one layer-sweep report
- one layer comparison figure
- one conclusion on whether Layer 6 remains the best microscopy layer

## 3. Causal interventions

Goal:
- move from observation to manipulation

Tasks:
- repeat steering-vector tests on more than one prompt
- compare intervention strength and failure modes
- repeat compression-hook tests with explicit guardrails
- classify interventions as stabilizing, neutral, or destructive

Deliverables:
- one intervention summary
- one steering replication note
- one compression robustness note

## 4. Vectorized conditioning

Goal:
- test whether vectorized semantic representations improve frontier stability

Tasks:
- run robust exp_003 batch with strict compressor guards
- compare `mean`, `attn_weighted`, and `pca1`
- evaluate whether any method improves `gap`, `coherence`, or `degeneracy` against raw and text-only compression
- document failure cases where vectorization damages structure

Deliverables:
- annotated results set
- method comparison markdown
- one recommendation for the current best vector mode

## 5. Cross-model microscopy

Goal:
- determine how much of the microscopy picture survives beyond GPT-2 Small

Tasks:
- run a light Phi-2 control on a small prompt panel
- record one of: same pattern, partial pattern, inconclusive
- avoid overclaiming generality

Deliverables:
- one cross-model note
- one explicit claims boundary section

## Claims policy

Allowed if evidence holds:

- certain latent regimes appear reproducible in the current setup
- some layers are more informative than others for runtime observability
- steering and compression can causally perturb residual-state behavior
- vectorized conditioning changes frontier geometry in measurable ways

Not allowed without stronger replication:

- general model-family claims
- production-ready risk detection
- batch-robust hallucination classification
- semantic compression proven as a general intervention

## Immediate priority order

1. robust exp_003 batch with valid compression
2. layer sweep replication around Layer 6
3. steering replication on multiple prompts
4. cross-model light control
5. synthesis note separating microscopy claims from hallucination claims

## Success condition for this track

The microscopy track is succeeding when it can state, with evidence:

- what internal structures are actually being observed
- where in the model they matter most
- which interventions change them causally
- which findings are stable enough to support downstream hallucination research
