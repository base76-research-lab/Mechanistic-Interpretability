# Mechanistic Interpretability (Base76)

English TL;DR: We build reviewable mechanistic interpretability experiments (SAEs + subspace probes) and a
geometry-based reliability signal ("Field View") that separates reasoning vs hallucination-like regimes.

Language: English-first, with Swedish notes where it improves precision/speed.

Mål: Kartlägga interna kretsar, representationer och *subspaces* i små/mellanstora språkmodeller.
Fokus: polysemanticitet, superposition, circuit discovery och feature-dictionaries via Sparse Autoencoders (SAE).

Det här spåret används också för att bygga en *reliability signal* ("Field View"):
en geometrisk risk-score som jämför residual-state i ett valt subspace mot ett "moln" av top-k kandidat-tokens
innan kollaps (unembedding).

## Läs detta först (rapporter/findings)

- Experimentrapport: `reports/exp_001_sae.md`
- Feature dictionary: `reports/feature_dict.md`
- Logg: `reports/logs/2026-03-04.md`
- Körningsartefakter (JSON): `experiments/exp_001_sae_v3/`
- Figur: `reports/figures/field_view_triage.png`
- Notebooks: `notebooks/README.md`
- Notes: `notes/README.md`

## Delmål

1. SAE på residual-/MLP-latenter för att extrahera glesa, tolkbara features.
2. Circuit discovery via patching/ablation på kända fenomen (t.ex. induction heads).
3. Feature dictionaries: katalog över identifierade features med exempel, labels och patch-effekter.
4. Subspace-baserad risk/triage: separera "reasoning" vs "hallucination" via state–candidate misalignment.

## Quickstart

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kör SAE (exempel):
```bash
python3 scripts/run_sae.py --model gpt2 --layer 5 --prompts data/prompts.txt --out experiments/exp_001_sae_local
```

Kör Field View (risk-signal):
```bash
python3 scripts/field_view.py --prompt "the opposite of hot is" --model gpt2 --layer 5 --units 472 468 57 156 346 --mode pc2 --topk 8
```

Generera figurer till rapporter:
```bash
python3 scripts/make_figures.py
```

## Artefakter och reproducibilitet

- Stora tensorfiler (t.ex. `activations.pt`, `sae_weights.pt`) behandlas som *build artifacts* och ignoreras i git.
- Rapportering och JSON-artefakter (metriker, top_features, field_view runs) ligger kvar för att göra findings reviewbara.

Mer detaljer: `experiments/README.md` och `reports/README.md`.

## Struktur

```
Mechanistic Interpretability/
├── data/                 # prompts, små datasets
├── experiments/          # exp-runs + JSON-artefakter
├── notebooks/            # explorations (Colab/GPU när relevant)
├── reports/              # findings, loggar, feature dict
└── scripts/              # körbara verktyg (SAE, field_view, patching)
```
