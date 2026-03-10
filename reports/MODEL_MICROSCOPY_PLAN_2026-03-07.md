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
- read-only oscilloscope traces now suggest a broader decision-transition zone around L6-L9
- reconstruction/write-back behaves like a commitment operator rather than neutral observation
- `L-SAE` without write-back is close to baseline in the current panel runs

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
- run read-only oscilloscope traces before write-back interventions on the same prompts
- run layer sweeps on selected prompts across multiple layers
- compare `state_norm`, `gap`, `coherence`, `degeneracy`, and entropy
- determine whether Layer 6 is a local artifact, or whether the real transition zone spans L6-L9

Deliverables:
- one layer-sweep report
- one layer comparison figure
- one conclusion on whether Layer 6 remains the best microscopy layer or should be reframed as part of an L6-L9 transition band

## 3. Causal interventions

Goal:
- move from observation to manipulation

Tasks:
- keep a hard distinction between read-only observer runs and write-back interventions
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

## 6. Decision-relevant feature learning (`L-SAE+R`)

Goal:
- test whether sparse features can become more decision-relevant without losing microscope value

Tasks:
- treat the unified observability stack as the baseline instrumentation layer
- compare `Plain SAE`, `Lens-only`, and `L-SAE+R` on the same canonical panel
- keep logit-lens supervision as the first supervision target
- evaluate whether `L-SAE+R` improves regime separation, recorder interpretability, or feature-to-frontier stability
- explicitly isolate `write-back active` versus `read-only supervised observation`
- document whether any apparent gain is real or just task-shaped noise

Deliverables:
- one `L-SAE+R` research question note
- one `L-SAE+R` protocol note
- one comparison report against plain SAE

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
2. read-only oscilloscope-assisted layer replication around L6-L9
3. steering replication on multiple prompts
4. cross-model light control
5. synthesis note separating microscopy claims from hallucination claims

## Next phase after current priority block

Once the current priority block is closed, the next microscopy phase should be:

## 7. Larger-model scaling

Goal:
- test whether the current microscopy picture survives in larger and more capable models

Questions:
- which current signals remain stable as model size increases?
- does the candidate bifurcation picture remain concentrated around the same layers?
- do steering and compression interventions remain informative in larger models?
- which findings are truly microscopy-level and which are small-model artifacts?

Expected outputs:
- one scaling protocol
- one model-comparison note
- one updated claim-boundary section separating supported small-model results from larger-model replication status

## Success condition for this track

The microscopy track is succeeding when it can state, with evidence:

- what internal structures are actually being observed
- where in the model they matter most
- which interventions change them causally
- which findings are stable enough to support downstream hallucination research
