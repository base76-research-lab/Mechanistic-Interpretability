# STATUS — Mechanistic Interpretability

Last updated: 2026-03-06

## Current position

**Active dialogue with ESA Phi-lab** (Giuseppe Borghi -> Nicolas Longepe; awaiting follow-up).

The project now has a functioning observability stack for internal geometry in transformer models.
Current findings suggest that state-candidate misalignment in the residual stream is a measurable
precursor to hallucination-like behavior, and that this signal is not reducible to high entropy alone.

## Core findings

| Claim | Evidence |
|---|---|
| Latent state-space structure can be measured through subspace projection | `exp_001_sae_v3` artifacts |
| Four state regimes (A-D) are observable on controlled prompts | Field View JSON artifacts and triage figure |
| State-candidate misalignment correlates with hallucination-like behavior | `field_view_hallucination.json` |
| Entropy alone is insufficient as a hallucination signal | preliminary high-entropy regime comparison |

## Active hypotheses

| ID | Hypothesis | Status |
|---|---|---|
| H1 | Token compression accelerates the emergence of states B/C/D | strong observation, but A/B comparison remains incomplete |
| H2 | The four-state taxonomy is robust across models and domains | GPT-2 only; Phi-2 control planned |
| H3 | `risk_refined = entropy_norm * gap_norm * (1 - coherence)` outperforms the current risk signal | not yet validated |
| H4 | Hallucination has a localizable onset point in layer dynamics | layer sweep planned |

## Experiment status

| Experiment | Description | Status |
|---|---|---|
| `exp_001_sae` v1/v2/v3 | SAE + Field View on GPT-2 layer 5 | complete |
| `exp_002_persona` | persona and traits artifacts | partial |
| `exp_003_compression_vectorized` | raw vs compressed vs vectorized proxy (`mean` / `attn_weighted` / `pca1`) | in progress; robust batch still missing |

**Critical gap:** `exp_003` has been run, but several runs used `compression_mode=unavailable`.
A robust batch with `--require-compressor` remains the next priority.

## Priority next steps

1. Run a robust batch for `exp_003` (see `reports/NEXT_STEPS_2026-03-05.md`)
2. Run a hallucination-focused layer sweep at layers 3, 6, 9, and 12 to test H4
3. Run a light Phi-2 notebook control to test H2
4. Run an A/B comparison with and without token compression to test H1

## Active track plans

- Microscopy plan: `reports/MODEL_MICROSCOPY_PLAN_2026-03-07.md`
- Weekly operational plan: `reports/ESA_WEEK_PLAN_2026-03-09.md`

## ESA context

- Sent: outreach and research package on 2026-03-05
- Response: Giuseppe Borghi replied positively and forwarded the thread to Nicolas Longepe
- Current state: awaiting contact from Nicolas Longepe
- Next action: no immediate outbound action required
- Related material: `/media/bjorn/iic/ESA/`

## Key files

- `reports/exp_001_sae.md` — SAE + Field View results, risk formulation, and threats to validity
- `reports/preliminary_2026-03-05_epistemic_state_layer.md` — early four-state model and refined risk framing
- `reports/NEXT_STEPS_2026-03-05.md` — operational run plan with commands and exit criteria
- `reports/feature_dict.md` — feature dictionary with labels and clusters
- `reports/figures/field_view_triage.png` — primary triage figure
- `experiments/exp_001_sae_v3/` — core JSON artifacts and run outputs
- `experiments/exp_003_compression_vectorized/` — vectorized proxy results
