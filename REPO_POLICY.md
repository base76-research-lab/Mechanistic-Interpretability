# Repository Policy — Research-Grade Hygiene

## Inclusion
- Commit only: runnable code/tools, stable configs/panels, finalized reports/figures.
- Exclude: draft notes, raw run outputs, weights, large artifacts, throwaway scripts.
- Experiments under `experiments/**` stay untracked (see `.gitignore`). Publish only derived tables/plots referenced in reports.

## Branching & Merging
- `main`: stable, reviewed changes only.
- Feature work: short-lived branches (e.g., `feat/prompt-vector`, `bench/hallu`); merge via squash after review.

## Versioned Tools
- Tag tool releases (e.g., `transformer_oscilloscope`) when interfaces change; update `CHANGELOG.md`.
- Pin dependencies (done) and keep quickstart snippets in READMEs.

## Documentation Sources of Truth
- `README.md`, `STATUS.md`, `research_index.md` must stay current.
- Reviewer-facing scientific navigation should remain explicit through `findings/README.md` and `reports/REVIEWER_INDEX.md`.
- Mixed-purpose documents that remain in git must be clearly marked or indexed so they are not confused with the main scientific findings surface.

## Repro Guidance
- Reports should include exact commands, model IDs (HF), seeds if applicable, and note that weights are not stored in git.
- Avoid committing weights; reference public sources.

## Review Checklist (before merge)
- `git status` clean from runs/drafts/weights.
- Run a lightweight smoke (e.g., oscilloscope trace or unit) on CPU.
- Update `CHANGELOG.md` for user-visible tool changes.
