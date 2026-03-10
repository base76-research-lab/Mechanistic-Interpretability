# Research Design and Instrument Plan

Date: 2026-03-10
Scope: `ESA/research/mechanistic-interpretability/`

## Purpose

Strengthen the research program where the current leverage is highest:

- experiment design
- instrument quality
- claims discipline
- run comparability
- analysis reproducibility

This plan assumes that better design and instrumentation will produce more defensible progress than
raw compute alone in the current phase.

## Current constraint picture

The project already has a functioning observability stack and a real findings layer, but the main
risks are still methodological:

- too many results can remain prompt-specific or anecdotal
- observation can drift into interpretation too quickly
- interventions can look informative without enough controlled comparison
- reports can accumulate without a single minimum evidence standard per finding type

## Plan structure

## 1. Lock a minimum evidence contract

Goal:
- make each new claim easier to classify as exploratory, supported, or not yet defensible

Tasks:
- define minimum evidence requirements for:
  - regime claims
  - layer claims
  - intervention claims
  - cross-model claims
- force every findings note to separate:
  - observation
  - interpretation
  - alternative explanations
  - claim boundary
- require source artifact paths for every core metric table or figure

Deliverables:
- one compact evidence checklist
- one updated findings-note workflow tied to `templates/template_findings.md`

## 2. Standardize the canonical prompt panels

Goal:
- reduce drift between runs and make comparisons more meaningful

Tasks:
- formalize one small canonical panel for fast sanity checks
- formalize one robust panel for evidence-bearing runs
- tag each prompt by function:
  - anchored
  - reasoning
  - transformation
  - hallucination-prone
  - trap / adversarial
- document when a prompt is allowed to enter or leave a canonical panel

Deliverables:
- one panel-definition note
- one prompt taxonomy table

## 3. Upgrade the instrumentation stack from useful to decision-relevant

Goal:
- make the instruments answer research questions, not just generate traces

Tasks:
- lock the default metric bundle for all microscopy runs:
  - entropy
  - gap
  - candidate coherence
  - degeneracy ratio
  - state norm
- ensure each runner emits the same minimum metadata:
  - model
  - layer
  - prompt id
  - intervention mode
  - seed if applicable
  - artifact path
- add explicit run-class labels:
  - sanity
  - benchmark
  - stress-test
  - ablation
  - control

Deliverables:
- one instrumentation standard note
- one common run-metadata schema

## 4. Force controlled comparison into the workflow

Goal:
- stop interesting single runs from silently becoming program direction

Tasks:
- require every new intervention result to be paired with a baseline comparison
- require a counter-case whenever a new metric looks promising
- create a short ablation template:
  - what is removed
  - what stays fixed
  - what failure would mean
- write down what counts as a real replication in the current setup

Deliverables:
- one ablation template
- one replication rule note

## 5. Tighten the report layer

Goal:
- make the findings surface easier to trust and easier to package externally

Tasks:
- split short internal notes from evidence-bearing findings
- keep one rolling summary of:
  - strongest supported signals
  - active uncertainties
  - blocked claims
- ensure every figure has a corresponding interpretation note and limitation note

Deliverables:
- one report-layer map
- one rolling summary update routine

## 6. Build a weekly research operating rhythm

Goal:
- reduce cognitive fragmentation and improve compounding progress

Tasks:
- reserve Monday for repair/sanity
- reserve Tuesday-Wednesday for benchmark runs and analysis
- reserve Thursday for stress-tests or ablations
- reserve Friday for synthesis, claim pruning, and packaging
- explicitly defer new side-paths unless they survive Friday review

Deliverables:
- one standing weekly cadence note
- one Friday claim-pruning checklist

## Priority order

1. minimum evidence contract
2. canonical prompt panels
3. instrumentation metadata standard
4. controlled comparison / ablation rule
5. report-layer tightening
6. weekly operating rhythm

## Success condition

This plan is succeeding when:

- the same claim can be rechecked quickly from artifacts
- findings are easier to compare across runs
- new metrics are harder to overinterpret
- the project produces fewer but stronger conclusions
- external packaging requires less cleanup because evidence discipline is already present upstream
