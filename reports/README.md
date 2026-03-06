# Reports / Findings

Senast uppdaterad: 2026-03-05

Den här mappen innehåller *reviewbara* resultat: vad som kördes, vad som observerades, och vad som faktiskt är nytt.

## Index

- `NEXT_STEPS_2026-03-05.md` — konkret körplan (guards, batch, analys, exit-kriterier)
- `preliminary_2026-03-05_epistemic_state_layer.md` — preliminar 4-state modell, "epistemiskt kvantlager" (kvant-liknande) och hallucinationskoppling
- `exp_001_sae.md` — SAE på GPT-2 layer 5 + Field View risk-signal (v1–v3)
- `feature_dict.md` — feature dictionary (labels + exempel)
- `logs/2026-03-04.md` — körlogg + hotfixar + nästa steg
- `figures/` — bilder som används i rapporter (genereras av `scripts/make_figures.py`)

## Rapportstandard (kort)

Varje findings-rapport ska ha:
- Setup: modell, layer, prompts/dataset, parametrar
- Resultat: metriker (med fil-stigar till JSON/artefakter)
- Observationer: *vad som faktiskt händer* (inte tolkning först)
- Tolkning: hypotes, mekanism, prediction
- Threats to validity: vad som kan vara artefakt
- Next steps: 1-3 konkreta körningar

Mall: `template_findings.md`
