# Contributing

This repository is research-first: prioritize clarity, reproducibility, and reviewable findings.

## Principles

- Prefer small, inspectable changes over large refactors.
- Every new result should be backed by an artifact (JSON/MD) and a short explanation in `reports/`.
- Avoid committing large tensors/model weights. If needed, use a dedicated storage strategy (LFS or external).

## What to add with an experiment

- A new folder under `experiments/exp_###_*`
- A findings note under `reports/` (use `reports/template_findings.md`)
- A command snippet to reproduce (or to approximate) the result

## Style

- Python: keep scripts runnable from repo root; use explicit CLI args.
- Reports: be explicit about model/version/layer/params and link to artifact files.

