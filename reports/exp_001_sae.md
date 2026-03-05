# exp_001_sae — GPT2-small, layer 5, SAE

## Versioner
- v1: 7 prompts, 75 tokens, steps=200, l1=5e-3 → mse~0.234, mean|z|~12.7 (överaktiverad), sparsitetsmått saknades.
- v2: 12 prompts, 128 tokens, LayerNorm på aktiveringar, steps=400, l1=1e-3 → mse~3.4e-4, mean|z|~0.73, ~62% nära noll.

## Setup (v2)
- Modell: gpt2-small
- Lager: hidden_states[5]
- Prompts: data/prompts.txt (12 prompts)
- Aktiveringar: 128 tokenpositioner, dim 768 (LayerNorm innan SAE)
- SAE: dict_size=256, steps=400, lr=1e-3, L1=1e-3

## Resultat (v2)
- Slut-MSE: ~3.4e-4, L1-term ~7.3e-4 (vid step 360/400)
- Sparsity: mean|z|=0.7265, frac|z|<1e-6 ≈ 0.618, frac|z|<1e-3 ≈ 0.621
- Top features: `experiments/exp_001_sae_v2/top_features.json` (10 units × 5 tokens). Exempel på topp-tokens (unit 0–2):
  - Enheter tar ofta ord som “the”, “of”, “to” (stopword-ish); några få fångar mönster som “reverse”, “repeat”, “queen”. Behöver labeling.

## Filer (v2)
- activations: experiments/exp_001_sae_v2/activations.pt
- SAE weights: experiments/exp_001_sae_v2/sae_weights.pt
- metrics: experiments/exp_001_sae_v2/metrics.json
- top features: experiments/exp_001_sae_v2/top_features.json

## Nästa steg
- Labela 5–10 units: läs top_features.json och dekoda token-exempel → lägg i feature-dictionary (ny fil). 
- Utöka prompts med mer varierade mönster (talserier, code, QA) för bredare features.
- Lägg sparsity-mått direkt i metrics (klart) och lägg till reconstruction histogram i nästa kör.
- Gör patching/ablation på en hypotes-feature (t.ex. en unit som svarar på “repeat”) och mät logit-shift.

## v3 (klar) — LayerNorm, dict=512, 800 steg, l1=5e-4
- 17 prompts, lager 5, activations LN: `experiments/exp_001_sae_v3/activations.pt`
- Viktfiler: `sae_weights.pt`, `metrics.json`, `top_features.json`, `polysemanticity.json`
- Sparsity: mean|z|≈0.56, ~62% nära noll
- Polysemanticitet: topp 20 enheter kartlagda i `polysemanticity.json`

### Viktiga features / kluster (v3)
- Antonym-subspace (472/468/57/156/346): fångar “opposite”/polaritet (cold, dark, light, tall). Viktigt: riktning ligger i subspace, inte i en enskild neuron.
- Analogi-pivot (132/133): “as/like/:” relationer.
- Polarity temp (421 ~ hot, 144 ~ cold) — kontrollpar.
- Parentheses/struktur (212/360/279) — paren open/close.
- Greeting/translation (396/478/217) — “good morning” + översättningscue.
- Role pairs (137/87) — president/queen slotting.

### Residual-insikt (“var är opposite?”)
Opposite ligger som riktning i residualen, inte i en nod. För att observera den:
1) Basprojektioner: cos mot W_E/W_U och c_proj visar att antonym-riktningen pekar mot cold/dark/tall och specifika c_proj-rader i lager 5.
2) Subspace-patch: patcha klustrets medel/PCA-komponent i stället för en enhet; layer-svep kan lokalisera kretsen.
3) Residualprofil: logga proj_pc1/pc2, ‖residual‖ och logit-entropi per token. Script: `scripts/opposite_probe.py`.

### Varför kompressorn känns “magisk”
Normal transformer: tokens → embedding → residual → features byggs gradvis över lager.
Token-kompressorn: tokens → SAE‑projektion → feature-space → (rest av modellen).
Effekt:
- Vi hoppar direkt till ett semantiskt basrum; subspace-kluster (antonym/analogi) blir explicita tidigt.
- Patch i feature-space injicerar riktningar som annars skulle uppstå senare; layer-svep visar att tidiga lager tar mest effekt medan mittlager kan absorbera.
- Residualen blir tolkbar: riktningar ≈ semantiska features snarare än rå token-basis.

### Latent future‑state: vad vi vet och inte vet
- Evidens: state_rollout visar att vi kan följa koordinater i antonym‑subspacet över flera steg, och att logit‑entropi/valda tokens hänger ihop med den rörelsen. Subspace_patch ändrar sannolikheter olika per lager → det finns ett “future‑state” som modelleras i residualen.
- Vad vi inte ser: vi ser inte var state “uppstår” – det finns redan som riktning i residualen, inte som explicit nod. Vi vet inte exakt vilka heads/MLP‑rader som bär det, eller varför just de koordinaterna stabiliseras.
- Plan för lokalisering:
  1) Lokalisering: cos‑sök mellan subspace och c_proj/W_O per lager + head‑mask/patching för att se var effekten dör.
  2) Dynamik: fortsätt state_rollout för fler fenomen (analogi, syntax) och mät Δcoords, logit‑entropi per steg.
  3) Semantik: projicera state på tokens (W_U) och på andra subspaces (analogi/syntax) för att se överlapp och konflikt → epistemisk osäkerhet.

Körningsexempel:
```
python3 scripts/opposite_probe.py --prompt "the opposite of hot is" --units 472 468 57 156 346
```
Output sparas till `experiments/exp_001_sae_v3/opposite_probe.json` och skriver topp-tokens för PC1/PC2 + profil per token.

### Nästa steg (v3+)
- Layer-svep & head-mask för subspace-patch (lokalisera krets).
- Lägg logit-shift-mätning för klusterpatch (analogt med patch_demo men på PCA-vektor).
- Fyll feature_dict med fler v3-labeler (pågående).

## Research alignment — FNC / Field / Future-state
- FNC-lab-idén (Field → Node → Cockpit → Collapse → State) matchar våra observationer: latent future-state finns som riktningar i residualen (Field), besluts-/kollaps sker först vid unembedding.
- Operator vs operand: SAE-subspaces (antonym/analogi) beter sig som operatoraxlar; tokenidentitet (operander) projiceras svagt på dessa axlar men väljs via andra dimensioner. Det stöder “shared mind”/latent fält-dynamik i Applied-Ai-Philosophy-repos.
- Field_view visar gapet: state långt på operator-PC, kandidater nära 0. Det tyder på att Field rymmer flera samverkande baser (operator + entity + syntax).

### TODO kopplat till FNC-hypotesen
- Alignment-test: korrelera (W_U·v_PC)·state_PC mot logits (operator vs operand).
- Multi-subspace-proj: projicera residual + kandidater på {antonym, analogi, entity/syntax}; leta efter mötespunkt i högre dim.
- “Residual + α·W_U[token]”: simulera lokal kollaps och se hur Field-coords flyttas.

## Snabbtest: Field View risk-signal (v3, antonym-subspace pc2)
Prompts körda med `scripts/field_view.py --topk 10` (units 472/468/57/156/346, layer 5):

![Field View triage (GPT-2, antonym subspace)](figures/field_view_triage.png)

| Prompt | H (logit-entropi) | Risk | Field coords | |coords| (operator-styrka) | Kandidat-gap | Observation |
|---|---|---|---|---|---|---|
| `2 + 2 =` | 5.12 | 0.55 | (-0.28, -0.47) | 0.55 | litet | State ≈ kandidatmoln → låg faktisk osäkerhet (nästan kollapsat) |
| `king is to queen as man is to` | 6.47 | 1.00 | (-3.69, -0.70) | 3.76 | stor | Operator stark, operander obestämda → resonemangsläge |
| `who was the president of france in 1200?` | 6.15 | 1.00 | (-5.92, -3.11) | 6.73 | mycket stor | State långt från kandidatmoln → klassisk hallucinationsgeometri (misalignment) |

Filer:  
- `experiments/exp_001_sae_v3/field_view_math.json`  
- `experiments/exp_001_sae_v3/field_view_analogy_run2.json`  
- `experiments/exp_001_sae_v3/field_view_hallucination.json`

Insikt: risk-signalen mäter “state–token misalignment”. Operator-styrka (|coords|) ger triage:
- hög H + stark operator → reasoning  
- hög H + svag operator → hallucination  
- medium H + litet gap → låst fakta.

Formell risk-definition (som i `field_view.py`):

```
entropy_norm = H / 10.0
gap_norm     = gap_state_to_candidates / (1 + candidate_spread_mean)
risk_score   = min(1.0, 0.5 * entropy_norm + 0.5 * gap_norm)
```

Där:
- `H` är logit-entropi (naturlig log) på sista token.
- `gap_state_to_candidates` beräknas som euklidiskt avstånd i valt subspace: 
  - state_coords = projektion av residualen på SAE-subspacet (oftast PCA/cluster-mean).
  - candidate_coords = projektion av topp-k tokenvektorer (W_U[topk]) på samma subspace.
  - gap = ‖state_coords − mean(candidate_coords)‖₂. *(Vi loggar även min‑dist i JSON men risk använder mean-dist för stabilitet.)*
- `candidate_spread_mean` = medel ‖candidate_coords‖.
- `|coords|` (operator-styrka) = ‖field_coords‖, rapporteras separat och används i triage men inte i risk-formeln än.

### Threats to Validity (v3)
- **SAE-basisdependens**: Features är inte unika; annan init kan ge andra baser. *Mitigation*: använd kluster-PCA (subspace) istället för enskild neuron och rapportera seeds.
- **Modellstorlek**: GPT‑2 small kan överdriva effekter eller sakna fenomen. *Mitigation*: parallell Phi‑2-körning (se plan) och logga modellversion.
- **Prompt-set storlek**: 17 prompts är litet. *Mitigation*: expandera promptlistan och redovisa varje körning med fil-stig + checksum.
