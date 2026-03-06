#!/usr/bin/env python3
"""
compare_compression_vectorized.py

Runs a 3-way comparison on the same prompts:
1) raw prompt
2) token-compressed prompt
3) compressed + vectorized-proxy prompt

The script reuses run_field_view_logged.py so metrics stay consistent with
existing observability outputs.

Outputs:
- experiments/exp_003_compression_vectorized/results_<ts>.json
- experiments/exp_003_compression_vectorized/results_<ts>.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RUN_LOGGED = ROOT / "scripts" / "run_field_view_logged.py"
OUT_DIR = ROOT / "experiments" / "exp_003_compression_vectorized"
DEFAULT_PROMPTS = [
    "king is to queen as man is to",
    "who was the president of france in 1200?",
    "the opposite of hot is",
]
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "in",
    "on",
    "for",
    "as",
    "at",
    "by",
    "with",
    "from",
    "was",
    "were",
    "be",
    "been",
    "has",
    "have",
    "had",
    "and",
    "or",
    "but",
    "that",
    "this",
    "it",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
}


def sanitize_label(text: str, max_len: int = 36) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return s[:max_len] or "prompt"


def try_load_token_compressor() -> Any | None:
    tc_root = ROOT.parent / "products" / "token-compressor"
    if not tc_root.exists():
        return None
    if str(tc_root) not in sys.path:
        sys.path.insert(0, str(tc_root))
    try:
        from compressor import LLMCompressEmbedValidate  # type: ignore

        # Lower min_tokens to allow compression in this experiment.
        return LLMCompressEmbedValidate(min_tokens=1)
    except Exception:
        return None


def compress_prompt(text: str, compressor: Any | None) -> tuple[str, dict[str, Any]]:
    if compressor is None:
        return text, {"mode": "unavailable", "coverage": None, "tokens_saved": None}
    try:
        res = compressor.process(text)
        return res.output_text, {
            "mode": res.mode,
            "coverage": float(res.coverage),
            "tokens_saved": int(res.tokens_saved),
        }
    except Exception:
        # Keep run alive even if local Ollama/models are unavailable.
        return text, {"mode": "error_fallback_raw", "coverage": None, "tokens_saved": None}


def compression_mode_ok(mode: str) -> bool:
    # "compressed" and "skipped" are valid pipeline outcomes.
    # "raw_fallback" means compression ran but failed semantic coverage.
    return mode in {"compressed", "skipped", "raw_fallback"}


def token_is_good_anchor(t: str) -> bool:
    if not t:
        return False
    if len(t) > 24:
        return False
    if not re.search(r"[A-Za-z0-9]", t):
        return False
    if t.lower() in STOPWORDS:
        return False
    return True


def ids_to_anchors(ids: list[int], tokenizer: Any, topk: int) -> list[str]:
    anchors: list[str] = []
    seen: set[str] = set()
    for idx in ids:
        t = tokenizer.decode([idx]).strip()
        if not token_is_good_anchor(t):
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        anchors.append(t)
        if len(anchors) >= topk:
            break
    if not anchors:
        anchors = ["semantic", "relation", "core"]
    return anchors


def stopword_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 1.0
    sw = sum(1 for t in tokens if t.lower() in STOPWORDS)
    return sw / len(tokens)


def build_vector_direction(
    *,
    prompt: str,
    tokenizer: Any,
    model: Any,
    method: str,
) -> torch.Tensor:
    toks = tokenizer(prompt, return_tensors="pt")
    input_ids = toks["input_ids"].to(model.device)
    emb_table = model.get_input_embeddings().weight.detach().cpu()
    token_emb = emb_table[input_ids[0].detach().cpu()]  # (n, d)

    if method == "mean":
        vec = token_emb.mean(dim=0)
        return vec / (vec.norm() + 1e-9)

    if method == "pca1":
        centered = token_emb - token_emb.mean(dim=0, keepdim=True)
        if centered.shape[0] < 2:
            vec = token_emb.mean(dim=0)
            return vec / (vec.norm() + 1e-9)
        _, _, v = torch.pca_lowrank(centered, q=1, center=False)
        vec = v[:, 0]
        return vec / (vec.norm() + 1e-9)

    if method == "attn_weighted":
        with torch.no_grad():
            out = model(**toks.to(model.device), output_attentions=True)
        if not out.attentions:
            vec = token_emb.mean(dim=0)
            return vec / (vec.norm() + 1e-9)
        att_layers = [a for a in out.attentions if a is not None]
        if not att_layers:
            vec = token_emb.mean(dim=0)
            return vec / (vec.norm() + 1e-9)
        # Weight tokens by average attention received from final position.
        att = torch.stack([a[0].mean(dim=0) for a in att_layers], dim=0).mean(dim=0)  # (seq, seq)
        w = att[-1].detach().cpu()
        w = torch.softmax(w, dim=0).unsqueeze(1)  # (seq,1)
        vec = (token_emb * w).sum(dim=0)
        return vec / (vec.norm() + 1e-9)

    raise ValueError(f"Unknown vector method: {method}")


def build_vectorized_proxy(prompt: str, tokenizer: Any, model: Any, topk: int = 8, method: str = "mean") -> tuple[str, list[str]]:
    mean_vec = build_vector_direction(prompt=prompt, tokenizer=tokenizer, model=model, method=method)
    emb_table = model.get_input_embeddings().weight.detach().cpu()  # (vocab, d_model)
    emb_norm = emb_table / (emb_table.norm(dim=1, keepdim=True) + 1e-9)
    sims = emb_norm @ mean_vec

    # Collect a larger pool first, then filter/dedupe.
    pool_k = max(topk * 8, 40)
    top_ids = torch.topk(sims, k=min(pool_k, sims.shape[0])).indices.tolist()
    anchors = ids_to_anchors(top_ids, tokenizer=tokenizer, topk=topk)

    packet = "semantic anchor packet: " + " | ".join(anchors)
    return packet, anchors


def latest_run_json(runs_dir: Path, before: set[str]) -> Path:
    after = {p.name for p in runs_dir.glob("*.json")}
    created = sorted(after - before)
    if not created:
        raise RuntimeError("No new run JSON was created.")
    return runs_dir / created[-1]


def run_variant(
    *,
    scenario: str,
    prompt: str,
    model_id: str,
    layer: int,
    units: list[int],
    mode: str,
    topk: int,
    device: str,
) -> dict[str, Any]:
    runs_dir = ROOT / "experiments" / "exp_001_sae_v3" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in runs_dir.glob("*.json")}

    cmd = [
        "python3",
        str(RUN_LOGGED),
        "--scenario",
        scenario,
        "--prompt",
        prompt,
        "--model",
        model_id,
        "--layer",
        str(layer),
        "--units",
        *[str(u) for u in units],
        "--mode",
        mode,
        "--topk",
        str(topk),
        "--device",
        device,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)

    run_json = latest_run_json(runs_dir, before)
    return json.loads(run_json.read_text())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--prompts-file", type=str, default="", help="Optional text file, one prompt per line")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--units", nargs="+", type=int, default=[472, 468, 57, 156, 346])
    ap.add_argument("--mode", choices=["mean", "pc1", "pc2"], default="pc2")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--vector-topk", type=int, default=8)
    ap.add_argument(
        "--vector-methods",
        nargs="+",
        choices=["mean", "attn_weighted", "pca1"],
        default=["mean"],
        help="One or more methods for compressed->vectorized proxy variants",
    )
    ap.add_argument(
        "--require-compressor",
        action="store_true",
        help="Fail fast if token-compressor is unavailable",
    )
    ap.add_argument(
        "--exclude-invalid-compression",
        action="store_true",
        help="Exclude compressed/vectorized rows when compression mode is unavailable/error",
    )
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    prompts = list(args.prompts)
    if args.prompts_file:
        pf = Path(args.prompts_file)
        file_prompts = [line.strip() for line in pf.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]
        prompts = file_prompts if file_prompts else prompts

    if not prompts:
        raise SystemExit("No prompts to run.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    compressor = try_load_token_compressor()
    if args.require_compressor and compressor is None:
        raise SystemExit("Token compressor unavailable. Start Ollama/models or remove --require-compressor.")

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device)
    llm.eval()

    rows: list[dict[str, Any]] = []

    for i, raw in enumerate(prompts, start=1):
        pid = f"p{i:02d}_{sanitize_label(raw)}"

        compressed, cmeta = compress_prompt(raw, compressor)

        variants = [
            ("raw", raw, {"compression": {"mode": "none"}}),
            ("compressed", compressed, {"compression": cmeta}),
        ]
        for vector_method in args.vector_methods:
            vector_prompt, anchors = build_vectorized_proxy(
                compressed,
                tok,
                llm,
                topk=args.vector_topk,
                method=vector_method,
            )
            variants.append(
                (
                    f"compressed_vectorized_proxy_{vector_method}",
                    vector_prompt,
                    {
                        "compression": cmeta,
                        "vectorization": {
                            "method": vector_method,
                            "anchor_tokens": anchors,
                            "anchor_stopword_ratio": stopword_ratio(anchors),
                        },
                    },
                )
            )

        for variant, text_in, meta in variants:
            scenario = f"exp003_{pid}_{variant}"
            run = run_variant(
                scenario=scenario,
                prompt=text_in,
                model_id=args.model,
                layer=args.layer,
                units=args.units,
                mode=args.mode,
                topk=args.topk,
                device=args.device,
            )

            row = {
                "ts": ts,
                "prompt_id": pid,
                "variant": variant,
                "raw_prompt": raw,
                "effective_prompt": text_in,
                "logit_entropy": run.get("metrics", {}).get("logit_entropy"),
                "gap_state_to_candidates": run.get("metrics", {}).get("gap_state_to_candidates"),
                "operator_strength": run.get("metrics", {}).get("operator_strength"),
                "risk_score": run.get("metrics", {}).get("risk_score"),
                "candidate_coherence": run.get("candidate_front", {}).get("coherence"),
                "candidate_variance": run.get("candidate_front", {}).get("variance"),
                "degeneracy_ratio_topk": run.get("candidate_front", {}).get("degeneracy_ratio_topk"),
                "degenerate": run.get("candidate_front", {}).get("degenerate"),
                "generic_tokens": run.get("candidate_front", {}).get("generic_tokens"),
                "run_record": run.get("artifact_paths", {}).get("field_view_json"),
                "meta": meta,
            }
            cmode = str(meta.get("compression", {}).get("mode", "none"))
            row["compression_mode"] = cmode
            row["compression_valid"] = compression_mode_ok(cmode) or variant == "raw"

            # Hard guard option for robust compression analyses.
            if args.exclude_invalid_compression and variant != "raw" and not row["compression_valid"]:
                continue
            rows.append(row)

    # Attach prompt-level deltas relative to raw baseline.
    by_prompt: dict[str, dict[str, Any]] = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], {})[r["variant"]] = r

    delta_keys = [
        "degeneracy_ratio_topk",
        "candidate_coherence",
        "gap_state_to_candidates",
        "logit_entropy",
    ]
    for prompt_rows in by_prompt.values():
        raw_row = prompt_rows.get("raw")
        if not raw_row:
            continue
        for variant, row in prompt_rows.items():
            for k in delta_keys:
                rv = row.get(k)
                bv = raw_row.get(k)
                if rv is None or bv is None:
                    row[f"delta_vs_raw_{k}"] = None
                else:
                    row[f"delta_vs_raw_{k}"] = float(rv) - float(bv)

    out_json = OUT_DIR / f"results_{ts}.json"
    out_csv = OUT_DIR / f"results_{ts}.csv"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    csv_fields = [
        "ts",
        "prompt_id",
        "variant",
        "compression_mode",
        "compression_valid",
        "logit_entropy",
        "gap_state_to_candidates",
        "operator_strength",
        "risk_score",
        "candidate_coherence",
        "candidate_variance",
        "degeneracy_ratio_topk",
        "degenerate",
        "delta_vs_raw_logit_entropy",
        "delta_vs_raw_gap_state_to_candidates",
        "delta_vs_raw_candidate_coherence",
        "delta_vs_raw_degeneracy_ratio_topk",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in csv_fields})

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
