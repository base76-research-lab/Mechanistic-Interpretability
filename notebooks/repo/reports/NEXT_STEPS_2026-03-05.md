# Next Steps Plan (2026-03-05)

Senast uppdaterad: 2026-03-05

## Mål

Validera robust om semantic compression och vectoriseringsmetoder påverkar frontier-stabilitet:

- `degeneracy_ratio_topk`
- `candidate_coherence`
- `gap_state_to_candidates`
- `logit_entropy`

med korrekta guards för kompressor-tillgänglighet.

## Fas 1: Körbar baseline (idag)

1. Säkerställ att kompressorn körs på riktigt:
   - använd `--require-compressor`
   - använd `--exclude-invalid-compression`
2. Kör liten sanity-batch (5-10 prompts) med metoder:
   - `mean`
   - `attn_weighted`
   - `pca1`
3. Verifiera att inga `compressed`-rader har `compression_mode=unavailable`.

Kommando:

```bash
cd "/media/bjorn/iic/workspace/Base76_Research_Lab/Mechanistic Interpretability"
python3 scripts/compare_compression_vectorized.py \
  --prompts-file data/prompts_sanity.txt \
  --vector-methods mean attn_weighted pca1 \
  --require-compressor \
  --exclude-invalid-compression \
  --device cpu
```

## Fas 2: Robust batch (den viktiga)

1. Bygg promptfil med 50-100 prompts, stratifierat:
   - relation/analogy
   - factual
   - instruction/planning
   - verbose/noise
2. Kör samma pipeline och spara resultat-json/csv.
3. Generera scatter-plots per körning:
   - `x=degeneracy`, `y=coherence`, color=`vector_mode`

Plottkommando:

```bash
python3 scripts/plot_vector_mode_scatter.py \
  --results experiments/exp_003_compression_vectorized/results_<ts>.json \
  --out reports/figures/vector_mode_degeneracy_vs_coherence_<ts>.png
```

## Fas 3: Analys och beslut

1. Jämför metoder via delta mot raw:
   - `delta_vs_raw_degeneracy_ratio_topk` (lägre bättre)
   - `delta_vs_raw_candidate_coherence` (högre bättre)
   - `delta_vs_raw_gap_state_to_candidates` (lägre bättre)
2. Använd median + percentiler (inte bara medel).
3. Bedöm prompt-typberoende effekt:
   - var `attn_weighted` vinner
   - var `pca1` är stabilare
   - var `mean` fallerar

## Fas 4: Targeted stress-tests (Gemini-förslag, anpassat)

1. A/B vector mode stress-test (trap prompts)
   - kör 50 trap-prompts i `raw` och giltig `compressed` setup
   - analysera `delta_vs_raw_gap_state_to_candidates`
   - mål: se om gap krymper i legitima resonemang men exploderar i hallucinationsfällor

2. Layer-sweep för State D (GPT-2 small, lager 4-8)
   - fokusera på hallucinationsprompter
   - mät per lager: `gap`, `coherence`, `degeneracy`, `state_norm`
   - mål: hitta var wideningen börjar (pre-emptive trigger)

3. W_U projection check
   - kör direkt cosine mellan residual state och unembedding-matris `W_U`
   - jämför top-k cosine tokens med top-k logits
   - mål: verifiera om state pekar mot prompt-irrelevanta tokens före kollaps

Kommando (W_U-check):

```bash
python3 scripts/wu_projection_check.py \
  --prompt "who was the president of france in 1200?" \
  --model gpt2 \
  --layer 5 \
  --topk 10
```

## Exit-kriterier (för ESA-ready findings)

- Minst 50 giltiga prompts (ej no-op compression).
- Reproducerbar trend i minst 2 metrics.
- Minst en figur + en tabell som tydligt visar metodskillnader.
- Kort findings-rapport i `reports/` med:
  - setup
  - resultat
  - threats to validity
  - nästa experiment

## Risker att bevaka

- Kompressor-fail som maskeras som `compressed`.
- Stopword-tunga anchor-packets (hög `anchor_stopword_ratio`).
- För liten sample size som ger falsk metodranking.
