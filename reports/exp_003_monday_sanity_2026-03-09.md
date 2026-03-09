# exp_003 Monday Sanity — 2026-03-09

Status: **BLOCKED — structure degradation** (gate = `blocked_structure_degradation` from `run_monday_sanity_pass.py`).
Command: `python3 scripts/compare_compression_vectorized.py --prompts-file data/prompts_sanity_2026-03-09.txt --require-compressor --exclude-invalid-compression --device cpu`
Artifacts:
- Summary: `experiments/exp_003_compression_vectorized/summary_20260309T064142Z.json`
- Results: `experiments/exp_003_compression_vectorized/results_20260309T064142Z.json`
- Inspection: `experiments/exp_003_compression_vectorized/inspection_20260309T064142Z.md`
- Sanity note: `experiments/exp_003_compression_vectorized/sanity_passes/sanity_pass_20260309T064511Z.{json,md}`

## Gate outcome (why blocked)
- **Decision drift vs raw** on sanity set (8 prompts): median `topk_overlap_vs_raw` for `compressed` = 0.74; `top1_match` only 0.75; best_structure_preserving_method=null.
- **Gap inflation**: median `gap_state_to_candidates` raw 4.18 → compressed 6.89 (+65%).
- **Compression not meaningful**: median tokens_saved = 2, but several prompts have token_delta=0 despite mode=compressed.

## Targeted diagnostics (3 weak prompts, `--prompts-jsonl ''`)
Artifacts: `summary_20260309T065133Z.{json,md}`, `inspection_20260309T065133Z.md`.
- `p01 the opposite of cold is`: compressed prompt `Opposite of cold?`; **topk_overlap=0.0**, tokens_saved=0, gap 7.29 vs raw 4.52 → pure drift.
- `p02 who was the president of france in 1200?`: compressed prompt shorter (8 tokens), **topk_overlap=0.6**, gap 7.29 vs raw 4.52, still degraded.
- `p03 a dragon that guards a castle`: compressor fell back to raw (`coverage_below_threshold`), tokens_saved=0 → non-compressive.

## Blockers to clear before robust batch
1) **Structure preservation**: need ≥0.9 topk_overlap/top1_match parity on sanity set; current 0.74/0.75.
2) **Gap control**: compressed median gap must not exceed raw by >0.25; currently +2.7.
3) **Real compression**: several prompts return token_delta=0 while claiming `compressed`; fix compressor or coverage thresholds.

## Recommended next steps (Tuesday plan)
- Patch compressor prompts that drift: keep anchors for short antonym/comparative questions; ensure rewrite retains target token (`hot/cold`) and question mark placement.
- Tighten `coverage_threshold` or disable rewrite when attempted_tokens_out >= raw tokens.
- Re-run sanity: `python3 scripts/run_monday_sanity_pass.py --prompts-file data/prompts_sanity_2026-03-09.txt --device cpu`.
- If pass => run robust batch (`run_exp003_week_batch.py --prompt-jsonl data/prompts_robust_2026-03-09.jsonl --require-compressor --exclude-invalid-compression --device cpu`).

## Notes
- Compressor is available locally (no `compression_mode=unavailable`), so issue is semantic drift, not availability.
- Vector proxies remain deferred; focus on text compression stability first.
