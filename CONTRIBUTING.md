# Contributing

This repository is research-first: prioritize clarity, reproducibility, and reviewable findings.

It also follows the Base76 `#research` operating model at the repository layer:

- use issue templates for new research questions, runs, analyses, and packages
- treat GitHub Issues and the GitHub Project as the operational layer
- treat `research_index.md`, `reports/`, and `experiments/` as the canonical scientific layer
- keep state, evidence level, and claim boundary explicit

## Principles

- Prefer small, inspectable changes over large refactors.
- Every new result should be backed by an artifact (JSON/MD) and a short explanation in `reports/`.
- Avoid committing large tensors/model weights. If needed, use a dedicated storage strategy (LFS or external).

## What to add with an experiment

- A new folder under `experiments/exp_###_*`
- A findings note under `reports/` (use `reports/templates/template_findings.md`)
- A command snippet to reproduce (or to approximate) the result
- A linked GitHub issue with the correct state and artifact type when the work is substantial

## GitHub operations

The repository's GitHub operations standard is defined in:

- `GITHUB_PROJECT_LAB_SPEC.md`
- `https://github.com/users/base76-research-lab/projects/1/views/1`

Use the issue templates in:

- `.github/ISSUE_TEMPLATE/`

## Style

- Python: keep scripts runnable from repo root; use explicit CLI args.
- Reports: be explicit about model/version/layer/params and link to artifact files.
