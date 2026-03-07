# Reports / Findings

Senast uppdaterad: 2026-03-07

Den här mappen innehåller *reviewbara* resultat: vad som kördes, vad som observerades, och vad som faktiskt är nytt.

I Base76 `#research`-systemet är detta den primära claims-ytan för repo:t. Notebooks får leda hit, inte ersätta detta lager.

## Index

- `MODEL_MICROSCOPY_PLAN_2026-03-07.md` — aktiv spårplan för `ai_microscopy`
- `summary_findings_2026-03-06.md` — huvudsammanfattning av nuvarande microscopy-fynd
- `compression_analysis_2026-03-06.md` — kompressionsintervention och strukturell fragilitet
- `steering_vector_analysis_2026-03-06.md` — kausal steering i residual space
- `NEXT_STEPS_2026-03-05.md` — konkret körplan (guards, batch, analys, exit-kriterier)
- `preliminary_2026-03-05_epistemic_state_layer.md` — preliminar 4-state modell, "epistemiskt kvantlager" (kvant-liknande) och hallucinationskoppling
- `exp_001_sae.md` — SAE på GPT-2 layer 5 + Field View risk-signal (v1–v3)
- `feature_dict.md` — feature dictionary (labels + exempel)
- `logs/2026-03-04.md` — körlogg + hotfixar + nästa steg
- `figures/` — bilder som används i rapporter (genereras av `scripts/make_figures.py`)

## Report standard

Varje findings-rapport ska så långt möjligt innehålla:

- current state
- evidence level
- claim boundary
- observation
- interpretation
- limitations
- next transition eller next experiment

Miniminivå för setup:
- Setup: modell, layer, prompts/dataset, parametrar
- Resultat: metriker (med fil-stigar till JSON/artefakter)
- Observationer: *vad som faktiskt händer* (inte tolkning först)
- Tolkning: hypotes, mekanism, prediction
- Threats to validity: vad som kan vara artefakt
- Next steps: 1-3 konkreta körningar

Mall: `template_findings.md`

## Evidence policy

Använd explicit evidensnivå när sakpåståenden görs:

- `Exploratory`
- `Supported`
- `Replicated`
