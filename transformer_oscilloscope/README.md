# Transformer Oscilloscope

Read-only observability toolkit for transformer dynamics.

## Components
- `trace` — collect per-layer/per-token telemetry via forward hooks (no write-back).
- `viz` — generate quick PNGs (entropy, gap, PCA scatter if stored).

## Usage
### Trace
```bash
python -m transformer_oscilloscope.cli trace \
  --prompt-jsonl data/prompts_observability_panel_2026-03-07.jsonl \
  --model gpt2 \
  --layers 1 6 9 11 \
  --device cpu \
  --out-dir experiments/exp_004_unified_observability_stack \
  --run-name transformer_oscilloscope_demo \
  --store-projections \
  --sae-state experiments/exp_001_sae_v4_lsae_v1_lw2e2/sae_weights.pt \
  --sae-topk 8
```
Outputs: `.../transformer_oscilloscope_demo/trace.jsonl`

### Viz
```bash
python -m transformer_oscilloscope.cli viz \
  --trace experiments/exp_004_unified_observability_stack/transformer_oscilloscope_demo/trace.jsonl \
  --out-dir experiments/exp_004_unified_observability_stack/transformer_oscilloscope_demo/plots
```
Outputs: heatmaps and PCA scatter PNGs.

### Report
```bash
python -m transformer_oscilloscope.cli report \
  --trace experiments/exp_004_unified_observability_stack/transformer_oscilloscope_demo/trace.jsonl \
  --out-dir experiments/exp_004_unified_observability_stack/transformer_oscilloscope_demo/plots \
  --report-name report.html
```
Outputs: `report.html` linking the generated PNGs.

## Examples
- Sample trace schema: `examples/trace_sample.jsonl`
- Prompt panel used in tests: `data/prompts_observability_panel_2026-03-07.jsonl`

## Notes
- Defaults are read-only; no reconstruction or injection paths are touched.
- Hashes (SHA256) are stored for hidden/MLP vectors to avoid leaking raw activations by default.
- SAE activations are optional; if `--sae-state` is provided, top-k features per token lagras i trace.
