# Compute Elasticity Plan

Date: 2026-03-10
Scope: `ESA/research/mechanistic-interpretability/`

## Purpose

Design a compute strategy that fits the current hardware reality while preserving the ability to
run heavier experiments when they are genuinely justified.

The aim is not to simulate a large lab locally. The aim is to create a hybrid system:

- local machine for design, instrumentation, smoke tests, and report generation
- offloaded compute for runs that exceed local CPU/RAM limits

## Current local baseline

Observed on 2026-03-10:

- CPU: Intel i5-1235U
- RAM: 7.3 GiB total
- swap pressure already present
- no visible NVIDIA GPU in the current environment
- storage is workable, but root resources remain modest

Conclusion:
- the current machine is appropriate for editing, scripting, visualization, and very light model
  work
- it is not the primary execution surface for heavier microscopy or scaling work

## Compute policy

## 1. Separate workloads by compute class

Goal:
- stop treating all jobs as if they belong on the same machine

Classes:

- `local-light`
  - markdown
  - plotting
  - report synthesis
  - smoke tests
  - prompt curation
  - artifact inspection
- `local-bounded`
  - tiny-model checks
  - CPU-only sanity runs
  - parser and runner validation
- `offload-required`
  - larger-model inference
  - robust benchmark batches
  - layer sweeps with large artifact volume
  - any run needing substantial VRAM or long wall-clock execution

Deliverable:
- one compute-class note tied to the main runner scripts

## 2. Make offload a first-class workflow

Goal:
- treat remote compute as normal infrastructure, not a special event

Tasks:
- define one standard offload packet per experiment:
  - code revision
  - prompt file
  - command
  - expected outputs
  - acceptance criteria
- ensure all runs can be resumed from saved artifacts rather than live notebooks
- prefer output bundles that can be pulled back and analyzed locally

Deliverables:
- one offload-run checklist
- one artifact handoff format

## 3. Reduce local resource waste

Goal:
- preserve local responsiveness for the work the machine is actually good at

Tasks:
- keep plotting, HTML report generation, and markdown synthesis local
- avoid long CPU-bound experiment runs on the laptop unless they are true smoke tests
- store large run outputs on the spacious external volume, not only on root
- define retention rules for bulky temporary artifacts

Deliverables:
- one storage and retention note
- one local-only smoke-test command set

## 4. Build a staged execution model

Goal:
- make every expensive run earn its cost

Stages:

1. local design check
2. local smoke test
3. offloaded bounded pilot
4. offloaded evidence-bearing batch
5. local analysis and packaging

Rule:
- no job should move to stage 4 unless stages 1-3 are already clean

Deliverable:
- one stage-gate note for all expensive runs

## 5. Define the minimum useful future upgrade path

Goal:
- avoid vague hardware desire and turn it into a concrete ladder

Near-term ladder:

- priority 1: any access to a 24 GB VRAM NVIDIA machine
- priority 2: 64-128 GB RAM on the main research box
- priority 3: a stable local NVMe scratch space for artifacts and caches
- priority 4: occasional rental or institutional access for HBM-class jobs only when justified

Interpretation:
- the first meaningful compute step is not a luxury workstation
- it is a usable hybrid setup with one stronger execution surface

Deliverable:
- one hardware ladder note with cost bands and target use-cases

## 6. Protect the research program from compute glamour

Goal:
- ensure compute expansion follows research need rather than fantasy scaling

Rules:
- do not widen model scope until the current claim boundary is clean
- do not rent expensive compute for ambiguous research questions
- use compute upgrades to increase replication, not only novelty
- treat every offloaded run as a decision about evidence, not just throughput

Deliverable:
- one compute justification rubric

## Priority order

1. workload classes
2. offload packet/checklist
3. staged execution model
4. storage and retention rules
5. hardware ladder
6. compute justification rubric

## Success condition

This plan is succeeding when:

- local work remains fast and cognitively light
- heavy runs are prepared cleanly and executed elsewhere when needed
- artifact return and local analysis are routine
- compute spending, borrowing, or upgrading is tied to concrete evidence gains
- the project stops being blocked by the fantasy of a future lab and starts using a real hybrid lab model now
