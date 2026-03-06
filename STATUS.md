# STATUS — Mechanistic Interpretability (Base76)

Senast uppdaterad: 2026-03-06

## Var vi är nu

**Aktiv dialog med ESA Phi-lab** (Giuseppe Borghi → Nicolas Longepe, inväntar kontakt).

Projektet har ett fungerande observability-stack för intern geometri i transformer-modeller.
Preliminära fynd visar att state-candidate misalignment i residualströmmen är en mätbar
föregångare till hallucination — distinkt från hög entropi ensamt.

## Kärnfynd (verifierade)

| Claim | Stöd |
|---|---|
| Latent state-space (Field) kan mätas med subspaceprojektion | exp_001_sae_v3 artefakter |
| 4 tillståndsregimer (A–D) är observerbara i kontrollerade prompts | field_view JSON + triage-figur |
| State-candidate misalignment korrelerar med hallucination-scenario | field_view_hallucination.json |
| Hög entropi ensamt är otillräckligt som hallucinationssignal | Two high-entropy regimes (preliminary report) |

## Aktiva hypoteser (ej bevisade)

| ID | Hypotes | Status |
|---|---|---|
| H1 | Token compressor accelererar framträdandet av states B/C/D | Stark observation, A/B saknas |
| H2 | 4-state taxonomin är robust över modeller och domäner | GPT-2 only, Phi-2 planerad |
| H3 | `risk_refined = entropy_norm * gap_norm * (1 - coherence)` är bättre signal än nuvarande risk | Ej testat |
| H4 | Hallucination har lokaliserbar uppkomstpunkt i lagerdynamiken | Layer-sweep planerad |

## Experimentstatus

| Experiment | Beskrivning | Status |
|---|---|---|
| exp_001_sae v1/v2/v3 | SAE + Field View, GPT-2 layer 5 | Klar |
| exp_002_persona | Persona/traits-artefakter | Partiell |
| exp_003_compression_vectorized | Raw vs compressed vs vectorized proxy (mean/attn_weighted/pca1) | Igång — robust batch saknas |

**Kritisk lucka:** exp_003 har körts men med `compression_mode=unavailable` i flera runs.
Robust batch (50–100 prompts, `--require-compressor`) är nästa prioriterade körning.

## Nästa steg (prioriterat)

1. Robust batch-körning av exp_003 (se `reports/NEXT_STEPS_2026-03-05.md`)
2. Layer-sweep för hallucination-prompt (lager 3, 6, 9, 12) — testar H4
3. Cross-model: Phi-2 notebook (testar H2)
4. A/B med/utan token compressor — testar H1

## ESA-dialog

- **Skickat:** Outreach + research package (2026-03-05)
- **Svar:** Giuseppe Borghi — positivt, routat till Nicolas Longepe
- **Inväntar:** Kontakt från Nicolas Longepe
- **Nästa action:** Inget från vår sida tills svar inkommer
- **Underlag:** `ESA/` i repots rot (`/media/bjorn/iic/ESA/`)

## Nyckelfiler

- `reports/exp_001_sae.md` — SAE + Field View (v1–v3), risk-formel, threats to validity
- `reports/preliminary_2026-03-05_epistemic_state_layer.md` — 4-state modell, refined risk, nästa verifiering
- `reports/NEXT_STEPS_2026-03-05.md` — körplan med kommandon och exit-kriterier
- `reports/feature_dict.md` — feature dictionary (labels + kluster)
- `reports/figures/field_view_triage.png` — central triage-figur
- `experiments/exp_001_sae_v3/` — alla artefakter (JSON, runs)
- `experiments/exp_003_compression_vectorized/` — vectorized proxy-resultat
