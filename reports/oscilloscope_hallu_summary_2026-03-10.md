# Transformer Oscilloscope + Hallucination Runs — 2026-03-10

## New tool
- Added `transformer_oscilloscope/` (read-only trace/viz/report, no write-back).
- CLI: `trace` (hooks residual/attn/MLP/logits), `viz` (PNG + CSV), `report` (HTML bundle).
- Optional SAE support: logs top-k feature activations per token when `--sae-state` is provided.

## Runs (stored locally under experiments/exp_004_unified_observability_stack/)
- `transformer_oscilloscope_demo`: layers 1/6/9/11, GPT-2 small, projections on; PNGs + HTML in `.../plots/`.
- Hallucination panel traces: baseline, baseline+recon, L-SAE no recon, L-SAE+R (80 traces vardera).
- Hallucination QA benchmarks: baseline, α-blends, entropy-gated recon; hallucination rate stayed 1.0, abstention collapsed when recon active.

## Key observations
- Reconstruction acts as a commitment operator: stabiliserar beslut, sänker abstention, men ökar inte faktuell korrekthet.
- L-SAE-supervision formar perturbationen först när recon skrivs tillbaka; utan write-back ≈ baseline.
- Oscilloscope-panelerna (entropy/gap/PCA) visar tydliga övergångszoner runt L6–L9 där decision dynamics skiftar.

## Next suggestions
- Keep experiments uncommitted; rerun oscilloscope on larger model/control when needed.
- Extend viz to include SAE feature heatmaps if SAE is supplied.
