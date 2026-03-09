#!/usr/bin/env python3
"""
wu_projection_check.py

Checks direct cosine alignment between residual state vector h (chosen layer)
and unembedding matrix W_U rows.

Purpose:
- verify whether hallucination-like states align strongly with prompt-irrelevant tokens
- compare cosine-topk vs logits-topk for the same prompt/layer
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "exp_003_compression_vectorized" / "wu_projection_checks"


def decode_token(tok: AutoTokenizer, idx: int) -> str:
    t = tok.decode([idx]).strip()
    return t if t else "<space>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device)
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    h = out.hidden_states[args.layer][0, -1, :].detach()  # (d_model,)
    logits = out.logits[0, -1, :].detach()  # (vocab,)

    wu = model.get_output_embeddings().weight.detach()  # (vocab, d_model)

    # Cosine between state vector and each W_U row
    h_norm = h / (torch.norm(h) + 1e-9)
    wu_norm = wu / (torch.norm(wu, dim=1, keepdim=True) + 1e-9)
    cos = wu_norm @ h_norm

    topk = min(args.topk, cos.shape[0])
    cos_vals, cos_idx = torch.topk(cos, k=topk)
    log_vals, log_idx = torch.topk(logits, k=topk)

    cosine_top = [
        {
            "token": decode_token(tok, int(i)),
            "token_id": int(i),
            "cosine": float(v),
        }
        for v, i in zip(cos_vals.cpu(), cos_idx.cpu())
    ]
    logits_top = [
        {
            "token": decode_token(tok, int(i)),
            "token_id": int(i),
            "logit": float(v),
        }
        for v, i in zip(log_vals.cpu(), log_idx.cpu())
    ]

    overlap = sorted(set(x["token_id"] for x in cosine_top).intersection(x["token_id"] for x in logits_top))

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = args.name if args.name else f"wu_projection_{ts}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.json"

    result = {
        "timestamp_utc": ts,
        "prompt": args.prompt,
        "model": args.model,
        "layer": args.layer,
        "topk": topk,
        "state_norm": float(torch.norm(h)),
        "cosine_topk": cosine_top,
        "logits_topk": logits_top,
        "overlap_token_ids": overlap,
        "overlap_count": len(overlap),
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"Saved: {out_path}")
    print(f"Overlap cosine/logits top-{topk}: {len(overlap)}")
    print("Cosine top tokens:", [x["token"] for x in cosine_top])
    print("Logits top tokens:", [x["token"] for x in logits_top])


if __name__ == "__main__":
    main()
