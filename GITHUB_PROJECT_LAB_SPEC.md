# GitHub Project Lab Spec

Last updated: 2026-03-07
Repository: `base76-research-lab/mechanistic-interpretability-`
Project: `MECHANISTIC_INTERPRETABILITY_PROJECT`
Project URL: `https://github.com/users/base76-research-lab/projects/1/views/1`

## Purpose

This document defines how the GitHub repository and GitHub Project should be used as the operational layer for the `mechanistic-interpretability` research program.

The Project is an operations board.

The repository remains the canonical scientific record.

## Canonical scientific sources

Use these as the source of truth:

- `research_index.md`
- `reports/`
- `experiments/`
- `reports/MODEL_MICROSCOPY_PLAN_2026-03-07.md`

The GitHub Project should not become a second source of truth for findings or claims.

## Project model

The board should represent the Base76 research state machine:

- `IDEA`
- `QUESTION`
- `PROTOCOL`
- `RUN`
- `ANALYSIS`
- `INTERNAL_REVIEW`
- `PACKAGE_READY`
- `EXTERNAL_COMMUNICATION`

Recommended board grouping:

- group by `State`

## Required custom fields

Set up these custom fields in the GitHub Project:

- `State`
- `Track`
- `Evidence`
- `Artifact Type`
- `Model`
- `Layer`
- `Claim Boundary`
- `Linked Canonical Artifact`

### Recommended field values

`Track`

- `ai_microscopy`
- `hallucinations`

`Evidence`

- `Exploratory`
- `Supported`
- `Replicated`

`Artifact Type`

- `question`
- `protocol`
- `run`
- `analysis`
- `report`
- `package`
- `external`
- `replication`

`Claim Boundary`

- `open`
- `defined`
- `needs review`

## Issue taxonomy

The repository should support four main issue classes:

1. `Research Question`
2. `Protocol / Run`
3. `Analysis / Insight`
4. `Package / External Communication`

Issue templates for these classes live in `.github/ISSUE_TEMPLATE/`.

## Label taxonomy

These labels should exist in the repository:

Track labels:

- `track:microscopy`
- `track:hallucinations`

Evidence labels:

- `evidence:exploratory`
- `evidence:supported`
- `evidence:replicated`

Artifact labels:

- `artifact:question`
- `artifact:protocol`
- `artifact:run`
- `artifact:analysis`
- `artifact:report`
- `artifact:package`
- `artifact:external`

Research workflow labels:

- `research`
- `follow-up`
- `insight`
- `replication`
- `blocked`
- `priority:high`
- `priority:medium`
- `priority:low`

Model/intervention labels:

- `model:gpt2`
- `model:phi2`
- `intervention:compression`
- `intervention:steering`
- `intervention:layer-sweep`

## Lab operating rules

- Every substantive research item should exist as an issue.
- Every issue should point to a canonical artifact in the repository.
- Every issue should have an explicit state and evidence level.
- Follow-up work should be opened as a linked issue, not buried in comments.
- Insights that materially change interpretation should be captured as `Analysis / Insight` issues and promoted into `reports/`.
- Packaging and outbound communication should not proceed unless claim boundary is defined.

## Core views

Recommended views:

1. `Research State`
   Board grouped by `State`

2. `Track`
   Table or board grouped by `Track`

3. `Evidence`
   Table grouped by `Evidence`

4. `Packaging`
   Filter: `Artifact Type` in `package`, `external`

5. `Replication`
   Filter: label `replication`

6. `Insights`
   Filter: label `insight`

7. `Follow-ups`
   Filter: label `follow-up`

8. `High Priority`
   Filter: label `priority:high`

9. `Blocked`
   Filter: label `blocked`

10. `Cross-model`
    Filter: labels `model:gpt2`, `model:phi2`

11. `Hallucinations`
    Filter: label `track:hallucinations`

12. `External Readiness`
    Filter: labels `artifact:package`, `artifact:external`

## Suggested view configuration

Use these layouts:

- `Research State` -> board, grouped by `State`, sorted by priority
- `Track` -> board, grouped by `Track`
- `Evidence` -> table, grouped by `Evidence`
- `Packaging` -> table, filtered to package/external work
- `Replication` -> table, filtered to `replication`
- `Insights` -> table, filtered to `insight`
- `Follow-ups` -> table, filtered to `follow-up`
- `High Priority` -> board or table, filtered to `priority:high`
- `Blocked` -> board or table, filtered to `blocked`
- `Cross-model` -> table, filtered to cross-model work
- `Hallucinations` -> board or table, filtered to the hallucinations sub-track
- `External Readiness` -> table, focused on package and outbound communication status

## Field behavior

Recommended usage:

- `State` is the main workflow field and should be visible in every view
- `Track` should separate `ai_microscopy` from `hallucinations`
- `Evidence` should be mandatory for analysis, report, package, and external items
- `Artifact Type` should distinguish protocol, run, analysis, package, and external work
- `Model` and `Layer` should stay visible in run-oriented views
- `Claim Boundary` should be visible in `Evidence`, `Packaging`, and `External Readiness`
- `Linked Canonical Artifact` should be filled for every issue that materially contributes to the scientific record

## Initial seeded backlog

The first operational backlog should include:

- robust `exp_003` compression batch with valid compressor
- Layer 6 replication and comparison against nearby layers
- multi-prompt steering validation
- light Phi-2 cross-model control
- microscopy vs hallucination claim-boundary split note
- next microscopy package refresh for external sharing

Those seed items now exist as repository issues:

- `#1` robust `exp_003` compression batch
- `#2` Layer 6 replication
- `#3` steering validation
- `#4` light Phi-2 control
- `#5` microscopy vs hallucination claim split
- `#6` next microscopy package refresh

## Relationship to the hallucinations sub-track

`hallucinations` is currently a sub-track within `ai_microscopy`, not a standalone repository.

Hallucination work may have:

- its own issues
- its own state entries
- its own insights
- its own package-preparation items

But it should remain attached to the same repository until the repo boundary criteria in the Base76 research operations system are met.

## Automation status

The repository-side GitHub lab layer is already in place:

- issue templates are available in `.github/ISSUE_TEMPLATE/`
- research labels exist in the repository
- the initial backlog is seeded as issues

The remaining Project-specific configuration requires GitHub project scope for `gh`:

- `gh auth refresh -s project`

Once that scope is available, the setup script below can be used:

- `scripts/setup_github_project_lab.sh`
