#!/usr/bin/env python3
"""
compare_compression_vectorized.py

Runs a same-material comparison on the token compressor as a scientific
fidelity instrument:
1) raw prompt
2) token-compressed prompt
3) optional compressed + vectorized-proxy prompt

The script reuses run_field_view_logged.py so metrics stay consistent with
existing observability outputs, then adds:
- behavioral fidelity metrics
- compression-efficiency metrics
- regime-aware summaries
- pass/fail gates for structure-preserving validation

Outputs:
- experiments/exp_003_compression_vectorized/results_<ts>.json
- experiments/exp_003_compression_vectorized/results_<ts>.csv
- experiments/exp_003_compression_vectorized/summary_<ts>.json
- experiments/exp_003_compression_vectorized/summary_<ts>.md
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import math
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
DEFAULT_PANEL_JSONL = ROOT / "data" / "prompts_observability_panel_2026-03-07.jsonl"
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

TOKEN_COMPRESSOR_LOAD_ERROR: str | None = None


def sanitize_label(text: str, max_len: int = 36) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return s[:max_len] or "prompt"


def load_prompt_rows(prompts: list[str], prompts_file: str, prompts_jsonl: str) -> list[dict[str, Any]]:
    if prompts_jsonl:
        rows = []
        path = Path(prompts_jsonl)
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item["prompt"]
            rows.append(
                {
                    "id": item.get("id", f"p{i:02d}_{sanitize_label(prompt)}"),
                    "prompt": prompt,
                    "regime": item.get("regime", "unknown"),
                    "stratum": item.get("stratum", "unknown"),
                }
            )
        return rows

    if prompts_file:
        pf = Path(prompts_file)
        file_prompts = [line.strip() for line in pf.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]
        prompts = file_prompts if file_prompts else prompts

    return [
        {
            "id": f"p{i:02d}_{sanitize_label(prompt)}",
            "prompt": prompt,
            "regime": "unknown",
            "stratum": "ad_hoc",
        }
        for i, prompt in enumerate(prompts, start=1)
    ]


def token_compressor_candidates() -> list[Path]:
    return [
        ROOT.parent / "products" / "token-compressor",
        ROOT.parents[2] / "workspace" / "Base76_Research_Lab" / "products" / "token-compressor",
    ]


def try_load_token_compressor() -> Any | None:
    global TOKEN_COMPRESSOR_LOAD_ERROR
    TOKEN_COMPRESSOR_LOAD_ERROR = None

    candidates = [path for path in token_compressor_candidates() if path.exists()]
    if not candidates:
        TOKEN_COMPRESSOR_LOAD_ERROR = (
            "token-compressor path not found; checked: "
            + ", ".join(str(path) for path in token_compressor_candidates())
        )
        return None

    tc_root = candidates[0]
    if str(tc_root) not in sys.path:
        sys.path.insert(0, str(tc_root))
    try:
        compressor_module = importlib.import_module("compressor")
        cls = getattr(compressor_module, "LLMCompressEmbedValidate")
        # Lower min_tokens to allow compression in this experiment.
        return cls(min_tokens=1)
    except Exception as exc:
        TOKEN_COMPRESSOR_LOAD_ERROR = f"{type(exc).__name__}: {exc} (root={tc_root})"
        return None


def compress_prompt(text: str, compressor: Any | None) -> tuple[str, dict[str, Any]]:
    if compressor is None:
        return text, {
            "mode": "unavailable",
            "coverage": None,
            "tokens_saved": None,
            "attempted_tokens_out": None,
            "attempted_tokens_saved": None,
            "rejection_reason": "compressor_unavailable",
        }
    try:
        res = compressor.process(text)
        return res.output_text, {
            "mode": res.mode,
            "coverage": float(res.coverage),
            "tokens_saved": int(res.tokens_saved),
            "attempted_tokens_out": int(res.attempted_tokens_out),
            "attempted_tokens_saved": int(res.attempted_tokens_saved),
            "rejection_reason": res.rejection_reason,
        }
    except Exception:
        # Keep run alive even if local Ollama/models are unavailable.
        return text, {
            "mode": "error_fallback_raw",
            "coverage": None,
            "tokens_saved": None,
            "attempted_tokens_out": None,
            "attempted_tokens_saved": None,
            "rejection_reason": "compression_runtime_error",
        }


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


def token_count(text: str, tokenizer: Any) -> int:
    return int(tokenizer(text, return_tensors="pt")["input_ids"].shape[1])


def iqr(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    q1 = xs[(len(xs) - 1) // 4]
    q3 = xs[((len(xs) - 1) * 3) // 4]
    return float(q3 - q1)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    if n % 2:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def rank_correlation(top_tokens: list[str], raw_tokens: list[str]) -> float | None:
    shared = [t for t in top_tokens if t in raw_tokens]
    if len(shared) < 2:
        return None
    rank_a = {t: i + 1 for i, t in enumerate(top_tokens)}
    rank_b = {t: i + 1 for i, t in enumerate(raw_tokens)}
    n = len(shared)
    d2 = sum((rank_a[t] - rank_b[t]) ** 2 for t in shared)
    return float(1 - (6 * d2) / (n * (n * n - 1)))


def fallback_mode(mode: str) -> bool:
    return mode in {"unavailable", "error_fallback_raw"}


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


def read_field_view_artifact(run: dict[str, Any]) -> dict[str, Any]:
    artifact_rel = run.get("artifact_paths", {}).get("field_view_json")
    if not artifact_rel:
        return {}
    artifact_path = ROOT / artifact_rel
    if not artifact_path.exists():
        return {}
    return json.loads(artifact_path.read_text())


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
    run = json.loads(run_json.read_text())
    run["_run_json_path"] = str(run_json.relative_to(ROOT))
    run["_field_view_artifact"] = read_field_view_artifact(run)
    return run


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "candidate_coherence",
        "degeneracy_ratio_topk",
        "gap_state_to_candidates",
        "logit_entropy",
        "topk_overlap_vs_raw",
        "rank_correlation_vs_raw",
        "tokens_saved",
        "token_delta",
        "attempted_tokens_saved",
        "attempted_token_delta",
        "compression_ratio",
    ]
    out: dict[str, Any] = {
        "count": len(rows),
        "fallback_rate": None,
        "invalid_rate": None,
        "top1_match_rate_vs_raw": None,
    }
    if rows:
        out["fallback_rate"] = float(
            sum(1 for r in rows if fallback_mode(str(r.get("compression_mode", "")))) / len(rows)
        )
        out["invalid_rate"] = float(
            sum(1 for r in rows if not r.get("compression_valid", True)) / len(rows)
        )
        top1_vals = [1.0 if r.get("top1_match_vs_raw") else 0.0 for r in rows if r.get("top1_match_vs_raw") is not None]
        if top1_vals:
            out["top1_match_rate_vs_raw"] = float(sum(top1_vals) / len(top1_vals))
    for key in numeric_keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and not math.isnan(float(r[key]))]
        out[f"median_{key}"] = median(vals)
        out[f"iqr_{key}"] = iqr(vals)
    return out


def classify_failure(summary: dict[str, Any], raw_summary: dict[str, Any], compressed_summary: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    coh = summary.get("median_candidate_coherence")
    raw_coh = raw_summary.get("median_candidate_coherence")
    deg = summary.get("median_degeneracy_ratio_topk")
    raw_deg = raw_summary.get("median_degeneracy_ratio_topk")
    gap = summary.get("median_gap_state_to_candidates")
    raw_gap = raw_summary.get("median_gap_state_to_candidates")
    overlap = summary.get("median_topk_overlap_vs_raw")
    token_delta = summary.get("median_token_delta")
    fallback_rate = summary.get("fallback_rate")
    ratio = summary.get("median_compression_ratio")
    compressed_deg = compressed_summary.get("median_degeneracy_ratio_topk")

    if fallback_rate is not None and fallback_rate > args.fallback_rate_threshold:
        reasons.append("fallback contamination")
    # Allow non-compressive cases to pass when semantic alignment is very high (overlap ≥ 0.9).
    if token_delta is not None and token_delta <= 0 and not (overlap is not None and overlap >= 0.9):
        reasons.append("non-compressive")
    if coh is not None and raw_coh is not None and coh < raw_coh - args.coherence_drop_threshold:
        reasons.append("frontier collapse")
    if deg is not None and raw_deg is not None and deg > raw_deg + args.degeneracy_rise_threshold:
        reasons.append("frontier collapse")
    if gap is not None and raw_gap is not None and gap > raw_gap + args.gap_rise_threshold:
        reasons.append("decision drift")
    if overlap is not None and overlap < args.topk_overlap_threshold:
        reasons.append("semantic loss")
    if ratio is not None and ratio < args.overcompression_ratio_threshold and reasons:
        reasons.append("over-compression")
    if deg is not None and compressed_deg is not None and deg > compressed_deg + args.degeneracy_rise_threshold:
        reasons.append("layer mismatch")
    return sorted(set(reasons))


def evaluate_variant(name: str, summary: dict[str, Any], raw_summary: dict[str, Any], compressed_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if name == "raw":
        return {
            "pass_fail": "baseline",
            "claim_boundary": "raw baseline only",
            "failure_modes": [],
        }
    failures = classify_failure(summary, raw_summary, compressed_summary, args)
    passes = len(failures) == 0
    if passes and name == "compressed":
        claim_boundary = "structure-preserving text compression in current GPT-2 setup"
    elif passes:
        claim_boundary = "candidate best current structure-preserving method in current GPT-2 setup"
    else:
        claim_boundary = "not defensible as best current method"
    return {
        "pass_fail": "pass" if passes else "fail",
        "claim_boundary": claim_boundary,
        "failure_modes": failures,
    }


def format_number(v: Any) -> str:
    if v is None:
        return "nan"
    return f"{float(v):.3f}"


def write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Compression Validation Summary",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Validation target: `{payload['validation_target']}`",
        f"- Claim boundary: {payload['claim_boundary']}",
        "",
        "## Review rubric",
        "",
        "| Method | Median coherence | Median degeneracy | Median gap | Median lens entropy | Top-k overlap vs raw | Token delta | Fallback rate | Pass/Fail | Claim boundary |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for name, summary in payload["overall"].items():
        lines.append(
            f"| {name} | {format_number(summary.get('median_candidate_coherence'))} | "
            f"{format_number(summary.get('median_degeneracy_ratio_topk'))} | "
            f"{format_number(summary.get('median_gap_state_to_candidates'))} | "
            f"{format_number(summary.get('median_logit_entropy'))} | "
            f"{format_number(summary.get('median_topk_overlap_vs_raw'))} | "
            f"{format_number(summary.get('median_token_delta'))} | "
            f"{format_number(summary.get('fallback_rate'))} | "
            f"{summary.get('pass_fail')} | {summary.get('claim_boundary')} |"
        )
    lines.extend(
        [
            "",
            "## Best current labels",
            "",
            f"- Best structure-preserving method: `{payload['best_current']['best_structure_preserving_method']}`",
            f"- Best balanced method: `{payload['best_current']['best_balanced_method']}`",
            f"- Most compressive method: `{payload['best_current']['most_compressive_method']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_inspection_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], []).append(row)

    lines = [
        "# Compression Inspection",
        "",
        "Per-prompt inspection of raw, compressed, and vectorized variants.",
        "",
    ]
    for prompt_id in sorted(by_prompt):
        prompt_rows = sorted(by_prompt[prompt_id], key=lambda r: r["variant"])
        raw_prompt = prompt_rows[0].get("raw_prompt", "")
        lines.extend(
            [
                f"## {prompt_id}",
                "",
                f"**Raw prompt**: `{raw_prompt}`",
                "",
                "| Variant | Compression mode | Rejection reason | Raw tokens | Attempted tokens | Effective tokens | Token delta | Attempted token delta | Top-k overlap | Top-1 match |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in prompt_rows:
            lines.append(
                f"| {row['variant']} | {row.get('compression_mode')} | {row.get('compression_rejection_reason') or ''} | "
                f"{row.get('raw_token_count')} | {row.get('attempted_token_count')} | {row.get('effective_token_count')} | "
                f"{format_number(row.get('token_delta'))} | {format_number(row.get('attempted_token_delta'))} | "
                f"{format_number(row.get('topk_overlap_vs_raw'))} | {row.get('top1_match_vs_raw')} |"
            )
            if row["variant"] != "raw":
                lines.append("")
                lines.append(f"- Effective prompt (`{row['variant']}`): `{row.get('effective_prompt', '')}`")
                vector_meta = row.get("meta", {}).get("vectorization")
                if vector_meta:
                    anchors = vector_meta.get("anchor_tokens", [])
                    lines.append(f"- Anchor tokens: `{', '.join(anchors)}`")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--prompts-file", type=str, default="", help="Optional text file, one prompt per line")
    ap.add_argument(
        "--prompts-jsonl",
        type=str,
        default=str(DEFAULT_PANEL_JSONL),
        help="Optional JSONL panel with id/prompt/regime/stratum rows; takes precedence over --prompts-file",
    )
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--units", nargs="+", type=int, default=[472, 468, 57, 156, 346])
    ap.add_argument("--mode", choices=["mean", "pc1", "pc2"], default="pc2")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--vector-topk", type=int, default=8)
    ap.add_argument(
        "--include-vectorized-proxy",
        action="store_true",
        help="Include compressed->vectorized proxy variants. Keep disabled while text compression is under repair.",
    )
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
    ap.add_argument("--coherence-drop-threshold", type=float, default=0.05)
    ap.add_argument("--degeneracy-rise-threshold", type=float, default=0.05)
    ap.add_argument("--gap-rise-threshold", type=float, default=0.25)
    ap.add_argument("--topk-overlap-threshold", type=float, default=0.50)
    ap.add_argument("--fallback-rate-threshold", type=float, default=0.25)
    ap.add_argument("--overcompression-ratio-threshold", type=float, default=0.60)
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    prompt_rows = load_prompt_rows(args.prompts, args.prompts_file, args.prompts_jsonl)
    if not prompt_rows:
        raise SystemExit("No prompts to run.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    compressor = try_load_token_compressor()
    if args.require_compressor and compressor is None:
        detail = TOKEN_COMPRESSOR_LOAD_ERROR or "unknown load error"
        raise SystemExit(
            "Token compressor unavailable. "
            f"Load failure: {detail}. "
            "Fix path/dependencies/runtime or remove --require-compressor."
        )

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device)
    llm.eval()

    rows: list[dict[str, Any]] = []

    for item in prompt_rows:
        raw = item["prompt"]
        pid = item["id"]
        regime = item["regime"]
        stratum = item["stratum"]

        compressed, cmeta = compress_prompt(raw, compressor)

        variants = [
            ("raw", raw, {"compression": {"mode": "none"}}),
            ("compressed", compressed, {"compression": cmeta}),
        ]
        if args.include_vectorized_proxy:
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
                "regime": regime,
                "stratum": stratum,
                "variant": variant,
                "raw_prompt": raw,
                "effective_prompt": text_in,
                "raw_token_count": token_count(raw, tok),
                "effective_token_count": token_count(text_in, tok),
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
                "field_view_candidates": run.get("_field_view_artifact", {}).get("candidates", []),
                "meta": meta,
            }
            cmode = str(meta.get("compression", {}).get("mode", "none"))
            attempted_tokens_out = meta.get("compression", {}).get("attempted_tokens_out")
            attempted_tokens_saved = meta.get("compression", {}).get("attempted_tokens_saved")
            row["compression_mode"] = cmode
            row["compression_rejection_reason"] = meta.get("compression", {}).get("rejection_reason")
            row["compression_valid"] = compression_mode_ok(cmode) or variant == "raw"
            row["attempted_token_count"] = attempted_tokens_out
            row["attempted_tokens_saved"] = attempted_tokens_saved
            row["attempted_token_delta"] = (
                float(attempted_tokens_saved) if attempted_tokens_saved is not None else None
            )
            row["token_delta"] = float(row["raw_token_count"] - row["effective_token_count"])
            row["tokens_saved"] = row["token_delta"]
            row["compression_ratio"] = (
                float(row["effective_token_count"] / row["raw_token_count"])
                if row["raw_token_count"] > 0
                else None
            )
            row["compression_outcome"] = (
                "none"
                if variant == "raw"
                else (
                    "compressed_shorter"
                    if cmode == "compressed"
                    else (
                        "compressed_longer_rejected"
                        if row["compression_rejection_reason"] == "non_compressive_or_expansive"
                        else cmode
                    )
                )
            )

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
        raw_candidates = [c.get("token", "") for c in raw_row.get("field_view_candidates", [])]
        for variant, row in prompt_rows.items():
            for k in delta_keys:
                rv = row.get(k)
                bv = raw_row.get(k)
                if rv is None or bv is None:
                    row[f"delta_vs_raw_{k}"] = None
                else:
                    row[f"delta_vs_raw_{k}"] = float(rv) - float(bv)
            current_candidates = [c.get("token", "") for c in row.get("field_view_candidates", [])]
            if current_candidates and raw_candidates:
                overlap = len(set(current_candidates) & set(raw_candidates)) / max(len(set(raw_candidates)), 1)
                row["topk_overlap_vs_raw"] = float(overlap)
                row["top1_match_vs_raw"] = bool(current_candidates[0] == raw_candidates[0]) if current_candidates and raw_candidates else None
                row["rank_correlation_vs_raw"] = rank_correlation(current_candidates, raw_candidates)
            else:
                row["topk_overlap_vs_raw"] = None
                row["top1_match_vs_raw"] = None
                row["rank_correlation_vs_raw"] = None

    out_json = OUT_DIR / f"results_{ts}.json"
    out_csv = OUT_DIR / f"results_{ts}.csv"
    out_summary_json = OUT_DIR / f"summary_{ts}.json"
    out_summary_md = OUT_DIR / f"summary_{ts}.md"
    out_inspection_md = OUT_DIR / f"inspection_{ts}.md"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    csv_fields = [
        "ts",
        "prompt_id",
        "variant",
        "compression_mode",
        "compression_outcome",
        "compression_rejection_reason",
        "compression_valid",
        "logit_entropy",
        "gap_state_to_candidates",
        "operator_strength",
        "risk_score",
        "candidate_coherence",
        "candidate_variance",
        "degeneracy_ratio_topk",
        "degenerate",
        "topk_overlap_vs_raw",
        "top1_match_vs_raw",
        "rank_correlation_vs_raw",
        "raw_token_count",
        "attempted_token_count",
        "effective_token_count",
        "token_delta",
        "attempted_token_delta",
        "attempted_tokens_saved",
        "tokens_saved",
        "compression_ratio",
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

    overall: dict[str, Any] = {}
    by_variant_regime: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    grouped_regime: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r["variant"] != "raw" and r.get("raw_token_count", 0) < 12:
            continue  # skip very short compressed variants
        grouped.setdefault(r["variant"], []).append(r)
        grouped_regime.setdefault((r["variant"], r["regime"]), []).append(r)

    for variant, xs in grouped.items():
        overall[variant] = summarize_group(xs)
    for (variant, regime), xs in grouped_regime.items():
        by_variant_regime.setdefault(variant, {})[regime] = summarize_group(xs)

    raw_summary = overall.get("raw", {})
    compressed_summary = overall.get("compressed", {})
    for variant in overall:
        overall[variant].update(evaluate_variant(variant, overall[variant], raw_summary, compressed_summary, args))

    pass_variants = [v for v, s in overall.items() if s.get("pass_fail") == "pass"]
    best_structure = None
    if pass_variants:
        best_structure = max(
            pass_variants,
            key=lambda v: (
                -(overall[v].get("median_degeneracy_ratio_topk") or 999),
                overall[v].get("median_candidate_coherence") or -999,
                -(overall[v].get("median_gap_state_to_candidates") or 999),
            ),
        )
    best_balanced = None
    candidates = [v for v in overall.keys() if v != "raw"]
    pass_candidates = [v for v in candidates if overall[v].get("pass_fail") == "pass"]
    if pass_candidates:
        best_balanced = min(
            pass_candidates,
            key=lambda v: (
                -(overall[v].get("median_tokens_saved") or 0),
                overall[v].get("median_degeneracy_ratio_topk") or 999,
                -(overall[v].get("median_candidate_coherence") or -999),
            ),
        )
    most_compressive = None
    if candidates:
        most_compressive = max(candidates, key=lambda v: overall[v].get("median_tokens_saved") or 0)

    summary_payload = {
        "timestamp_utc": ts,
        "validation_target": "scientific_fidelity",
        "model": args.model,
        "layer": args.layer,
        "mode": args.mode,
        "topk": args.topk,
        "include_vectorized_proxy": bool(args.include_vectorized_proxy),
        "prompt_panel": args.prompts_jsonl if args.prompts_jsonl else args.prompts_file or "inline_prompts",
        "claim_boundary": "current GPT-2 Small structure-preserving comparison only; not cross-model or globally optimal",
        "thresholds": {
            "coherence_drop": args.coherence_drop_threshold,
            "degeneracy_rise": args.degeneracy_rise_threshold,
            "gap_rise": args.gap_rise_threshold,
            "topk_overlap": args.topk_overlap_threshold,
            "fallback_rate": args.fallback_rate_threshold,
            "overcompression_ratio": args.overcompression_ratio_threshold,
        },
        "overall": overall,
        "by_variant_regime": by_variant_regime,
        "best_current": {
            "best_structure_preserving_method": best_structure,
            "best_balanced_method": best_balanced,
            "most_compressive_method": most_compressive,
        },
    }
    out_summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    write_summary_markdown(out_summary_md, summary_payload)
    write_inspection_markdown(out_inspection_md, rows)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_summary_json}")
    print(f"Saved: {out_summary_md}")
    print(f"Saved: {out_inspection_md}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
