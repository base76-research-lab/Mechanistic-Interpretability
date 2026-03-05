#!/usr/bin/env python3
"""
field_view.py — inspect the Field (possible future-states) before collapse

For a prompt and position (default: last token):
- take the residual (layer 5 by default) and project into a chosen SAE subspace (default: antonym cluster PC2)
- compute logit entropy H
- take top-k logit candidates, project their W_U vectors into the same subspace -> a "cloud" of possible collapses
- save everything to JSON and print a short summary

Exempel:
python3 scripts/field_view.py --prompt "the opposite of hot is" --units 472 468 57 156 346 --mode pc2 --topk 8
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAE_STATE = ROOT / "experiments/exp_001_sae_v3/sae_weights.pt"
OUT_JSON = ROOT / "experiments/exp_001_sae_v3/field_view.json"


def load_basis(units: List[int], mode: str, sae_state: Path) -> Tuple[torch.Tensor, List[float]]:
    state = torch.load(sae_state, map_location="cpu")
    dec = state["decoder.weight"]  # (d_model, dict)
    vecs = dec[:, units]  # (d_model, k)
    if mode == "mean":
        basis = vecs.mean(dim=1, keepdim=True)
        return basis, [1.0]
    x = vecs - vecs.mean(dim=1, keepdim=True)
    u, s, _ = torch.linalg.svd(x, full_matrices=False)
    if mode == "pc1":
        basis = u[:, :1] * s[:1]
        var = (s ** 2) / (s ** 2).sum()
        return basis, [float(var[0])]
    # pc2
    basis = u[:, :2] * s[:2]
    var = (s ** 2) / (s ** 2).sum()
    return basis, [float(var[0]), float(var[1])]


def project(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.matmul(vec, basis)


def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    return float(-(probs * torch.log(probs + 1e-9)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--units", nargs="+", type=int, default=[472, 468, 57, 156, 346])
    ap.add_argument("--mode", choices=["mean", "pc1", "pc2"], default="pc2")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--name", type=str, default="field_view")
    ap.add_argument("--model", type=str, default="gpt2", help="HF model id")
    ap.add_argument("--layer", type=int, default=5, help="Index i hidden_states att projicera")
    ap.add_argument(
        "--sae_state",
        type=str,
        default=str(DEFAULT_SAE_STATE),
        help="Path to SAE weights (expects decoder.weight)",
    )
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device)
    model.eval()

    sae_state_path = Path(args.sae_state)
    basis, var = load_basis(args.units, mode=args.mode, sae_state=sae_state_path)
    basis = basis.to(args.device)  # (d_model, k)

    data = tok(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**data, output_hidden_states=True)
        h = out.hidden_states[args.layer][0, -1, :]  # residual at last token
        logits = out.logits[0, -1, :]

    coords = project(h, basis)  # (k,)
    H = entropy_from_logits(logits)
    topv, topi = torch.topk(logits, k=args.topk)

    # Projektion av kandidat-collapses (W_U)
    w_u = model.get_output_embeddings().weight  # (vocab, d_model)
    cand = []
    for logit, idx in zip(topv, topi):
        w = w_u[idx]
        c = project(w, basis)
        cand.append({
            "token": tok.decode([idx]).strip() or "<space>",
            "logit": float(logit),
            "coords": [float(x) for x in c],
        })

    # Risk/koherens
    state_norm = float(torch.norm(coords))
    cand_mean = torch.tensor([c["coords"] for c in cand]).mean(dim=0)
    gap = float(torch.norm(coords - cand_mean))
    cand_spread = float(torch.tensor([c["coords"] for c in cand]).norm(dim=1).mean())
    entropy_norm = H / 10.0  # grov normalisering
    gap_norm = gap / (1.0 + cand_spread)
    risk = min(1.0, 0.5 * entropy_norm + 0.5 * gap_norm)

    result = {
        "prompt": args.prompt,
        "units": args.units,
        "mode": args.mode,
        "explained_var": var,
        "topk": args.topk,
        "field_coords": [float(x) for x in coords],
        "operator_strength": state_norm,
        "logit_entropy": H,
        "state_norm": state_norm,
        "candidate_mean": [float(x) for x in cand_mean],
        "candidate_spread_mean": cand_spread,
        "gap_state_to_candidates": gap,
        "risk_score": risk,
        "candidates": cand,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out_path = OUT_JSON if args.name == "field_view" else OUT_JSON.with_name(f"{args.name}.json")
    out_path.write_text(json.dumps(result, indent=2))

    print(f"Prompt: {args.prompt}")
    print(f"Field coords: {[float(x) for x in coords]}, |coords|={state_norm:.3f}, H={H:.3f}, risk={risk:.3f}")
    print("Top candidates (token, logit, coords):")
    for c in cand:
        print(f"  {c['token']!r:>8s}  logit={c['logit']:.2f}  coords={c['coords']}")
    print("Saved →", out_path)


if __name__ == "__main__":
    main()
