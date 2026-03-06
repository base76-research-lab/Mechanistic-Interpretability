# Preliminary Report (2026-03-05)

Senast uppdaterad: 2026-03-05

## Titel

Epistemiskt "kvantlager" (kvant-liknande, ej kvantfysik) i residualströmmen:
4 preliminära states och koppling till hallucination.

## Sammanfattning

Gårdagens körningar i `exp_001_sae_v3` visar att modellen verkar passera genom ett latent operatorlager
innan token-kollaps. Vi kallar detta preliminärt ett "epistemiskt kvantlager": ett tillståndsrum med
diskreta riktningar/subspaces som beter sig superpositions-liknande, men utan claim om fysisk kvantmekanik.

Observationerna indikerar 4 tydliga tillståndsregimer. Minst en av dessa verkar starkt kopplad till
hallucination-liknande output via hög state-candidate misalignment.

## Dataunderlag (från 2026-03-04)

Källa:
- `reports/logs/2026-03-04.md`
- `experiments/exp_001_sae_v3/field_view_math.json`
- `experiments/exp_001_sae_v3/field_view_analogy_run2.json`
- `experiments/exp_001_sae_v3/field_view_antonym.json`
- `experiments/exp_001_sae_v3/field_view_hallucination.json`

Mätvärden (PC2 i antonym-subspace):

1) `math_det` ("2 + 2 =")
- H = 5.123
- state_norm = 0.549
- gap = 0.691
- risk = 0.549

2) `analogy_reason` ("king is to queen as man is to")
- H = 6.467
- operator_strength = 3.761
- gap = 3.951
- risk = 1.000

3) `antonym` ("the opposite of hot is")
- H = 5.822
- state_norm = 4.138
- gap = 4.246
- risk = 1.000

4) `hallucination` ("who was the president of france in 1200?")
- H = 6.150
- operator_strength = 6.688
- gap = 6.815
- risk = 1.000

## Preliminär 4-state modell

State A: Anchored/near-collapse
- Liten operatorstyrka + liten gap.
- Kandidatmoln och state ligger nära varandra.
- Exempel: `math_det`.

State B: Structured reasoning mode
- Hög operatorstyrka + stor gap, men uppgiften är semantiskt legitim.
- Modellen verkar vara i "operator-först"-läge där operandval ännu inte kollapsat.
- Exempel: `analogy_reason`.

State C: Semantic operator transition
- Hög state_norm/gap på semantiska transformationsprompter.
- Tyder på stark intern omskrivning i feature-space innan tokenval.
- Exempel: `antonym`.

State D: Misaligned high-energy state (hallucination-prone)
- Mycket hög operatorstyrka + mycket stor gap mot kandidater.
- Tolkning: modellen är "långt bort" i latent operatorrum relativt sannolika tokenkandidater.
- Exempel: `hallucination`.

## Token compressor: preliminär effekt

Arbetshypotes:
Token compressor/SAE-projektion verkar accelerera framträdandet av dessa latenta tillståndsstrukturer
(särskilt B/C/D) jämfört med normal token→embedding→residual progression.

Status:
- Detta är en stark observation/hypotes i nuvarande data.
- Vi har ännu inte en full A/B-kvantifiering med samma prompts, samma seeds, med/utan kompressor i identisk setup.

## Hallucinationskoppling (preliminär)

Nuvarande data stöder:
- Hallucination-scenariot har högst gap och hög operatorstyrka i testsetet.
- Risk-signalen (H + gap) saturerar till 1.0 i analogi/antonym/hallucination, men geometrin skiljer sig:
  hallucination ligger längst från kandidatmolnet.

Praktisk tolkning:
- Inte alla high-risk states är "fel". Vissa är resonemangslägena.
- Hallucination verkar uppstå när misalignment blir extrem i kombination med hög latent operatorenergi.

## Candidate semantics under misalignment (new observation)

Direkt jämförelse av top-k kandidater visar en viktig skillnad:

- `hallucination` (`who was the president of france in 1200?`):
  top-k är generiska/formella tokens (`<space>`, `I`, `He`, `)`, `And`, `The`, `It`, `(`, `That`).
- `analogy_reason` (`king is to queen as man is to`):
  top-k är semantiskt relaterade till uppgiften (`man`, `queen`, `woman`, `king`, `wife`, `mother`).

Tolkning:
- Hög risk i sig är inte tillräckligt för hallucination.
- Hallucination-fallet visar dessutom semantisk kollaps i kandidatfronten:
  modellen är i ett hög-energi state men kandidatmolnet är svagt uppgiftsförankrat.

## Two high-entropy regimes (refined interpretation)

Vi ser nu två olika tillstånd med hög entropi + stor gap, men olika struktur:

1) Structured uncertainty (reasoning-like)
- H hög, gap stor
- kandidatfront fortfarande semantiskt koherent
- tolkning: operator aktiv, men operand ej kollapsad

2) Semantic collapse (hallucination-prone)
- H hög, gap stor/extrem
- kandidatfront degenererar till syntax/filler-tokens
- tolkning: state driver långt i operatorrum utan motsvarande tokenregion

Detta betyder att "hög entropi" inte är tillräckligt som hallucinationssignal.
Skillnaden verkar ligga i kandidatfrontens semantiska struktur.

## Proposed metrics for next run

För att skilja "reasoning" från "hallucination" bättre, lägg till:

1) `semantic_coherence_topk`
- Mål: mäta hur semantiskt sammanhållet top-k är relativt promptens uppgift.
- Enkel operationalisering:
  - bygg embedding-vektorer för top-k kandidater (t.ex. W_E/W_U eller extern sentence embedding)
  - beräkna medel cosinus-similaritet mellan kandidater och mot promptnyckelord
  - låg koherens + hög gap indikerar hallucination-prone state

2) `candidate_variance`
- Mål: mäta hur brett/splittrat kandidatmolnet är i valt subspace.
- Definition (subspace):
  - `candidate_variance = trace(cov(candidate_coords))`
- Hjälper separera:
  - strukturerat resonemang: hög operatorstyrka men mer sammanhållen kandidatgeometri
  - hallucination: hög operatorstyrka + hög misalignment + ofta hög varians/låg koherens

3) `degeneracy_ratio_topk`
- Mål: mäta hur stor andel av top-k som är generiska syntax/filler-tokens.
- Enkel operationalisering:
  - markera tokens som stopword/punctuation/pronoun/starter (`The`, `I`, `And`, `It`, `<space>`, etc.)
  - ratio = count(generic_tokens) / k
- Hög degeneracy_ratio + hög gap + låg coherence bör starkt indikera hallucination-prone state.

## Refined risk proposal

Ny hypotes för riskscore:

`risk_refined = entropy_norm * gap_norm * (1 - candidate_coherence)`

Intuition:
- reasoning-like states: hög entropy + gap, men också hög coherence → risk dämpas
- hallucination-prone states: hög entropy + extrem gap + låg coherence → risk skjuter upp

Denna signal bör testas mot nuvarande `risk_score` på samma promptset och sedan på större benchmark.

## Layer-dynamics test (immediate)

Nästa snabba test för hallucinationsprompten:
- kör samma analys över flera lager (t.ex. 3, 6, 9, 12)
- mäta per lager: `state_norm`, `gap`, `candidate_coherence`, `degeneracy_ratio`

Arbetshypotes:
- tidiga lager har mer semantiskt koherent kandidatfront
- senare lager visar degeneration i hallucinationfallet

Om detta hålls får vi en lokaliserbar uppkomstpunkt för hallucination i lagerdynamiken.

## Vad som är claim vs hypotes

Claim (stöds av artefakter):
- Ett latent state-space (Field) kan mätas och loggas med subspaceprojektion.
- State-candidate misalignment korrelerar med hallucination-liknande scenario i kontrollerad prompt.

Hypotes (ej slutbevisad):
- Ett "epistemiskt kvantlager" med diskreta tillståndsregimer är generell mekanism över modeller.
- Token compressor får dessa tillstånd att uppstå tidigare/snabbare.
- 4-state taxonomin är robust över fler domäner och seeds.

## Nästa verifieringssteg (prioriterat)

1) A/B-test med och utan token compressor
- Samma prompts, seeds, lager, top-k.
- Mät: tid till state-separation, gap, risk, feltyp.

2) Cross-model replika
- GPT-2 small vs Phi-2 (planerad notebook) med samma protokoll.

3) Hallucination benchmark
- 30-100 prompts med kända falska/fuzzy fakta.
- Klassificera output + jämför mot states A-D.

4) State classifier
- Träna enkel klassificerare på `field_coords`, `state_norm`, `gap`, `H` för att separera A/B/C/D.

## Slutsats

Den viktigaste preliminära slutsatsen är inte "kvantfysik", utan:
det finns ett mätbart, diskretiserbart epistemiskt mellanlager i modellens interna geometri,
och vissa hög-energi misalignment-states verkar vara en direkt mekanistisk rot till hallucination.
