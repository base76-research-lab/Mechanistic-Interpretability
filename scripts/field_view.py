#!/usr/bin/env python3
"""
field_view.py — titt in i Field (möjliga future-states) innan kollaps

För en prompt och position (default sista token):
- Beräkna residualen (layer 5) och projicera på valda SAE-subspace (default antonym-kluster PC2)
- Beräkna logit-entropi H
- Ta top-k logitkandidater, projicera deras W_U-vektorer i samma subspace → “moln” av möjliga kollapser
- Spara allt i JSON + skriv en snabb textöversikt

Exempel:
python3 scripts/field_view.py --prompt "the opposite of hot is" --units 472 468 57 156 346 --mode pc2 --topk 8
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_vector_injection import apply_injection, build_prompt_vector, resolve_token_index

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAE_STATE = ROOT / "experiments/exp_001_sae_v3/sae_weights.pt"
OUT_JSON = ROOT / "experiments/exp_001_sae_v3/field_view.json"


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z


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


def load_sae(sae_state: Path, d_model: int) -> SparseAutoencoder:
    state = torch.load(sae_state, map_location="cpu")
    d_hidden = state["encoder.weight"].shape[0]
    sae = SparseAutoencoder(d_model, d_hidden)
    sae.load_state_dict(state)
    sae.eval()
    return sae


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
        help="Path till SAE-vikter (decoder.weight)",
    )
    ap.add_argument(
        "--use-sae-reconstruction",
        action="store_true",
        help="Project SAE reconstruction instead of the raw hidden vector before lens/candidate metrics.",
    )
    ap.add_argument(
        "--prompt-vector-mode",
        choices=["none", "residual_mean", "sae_recon", "basis_recon"],
        default="none",
        help="Build a prompt-level vector and inject it before field/lens metrics.",
    )
    ap.add_argument("--prompt-vector-source-layer", type=int, default=5)
    ap.add_argument("--prompt-vector-token-span", choices=["all", "last_n"], default="all")
    ap.add_argument("--prompt-vector-last-n", type=int, default=4)
    ap.add_argument("--inject-at-layer", type=int, default=5)
    ap.add_argument("--inject-at-token-index", type=int, default=-1)
    ap.add_argument("--inject-alpha", type=float, default=0.0)
    ap.add_argument("--inject-mode", choices=["add", "mix"], default="add")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    if not hasattr(cfg, 'pad_token_id') or cfg.pad_token_id is None:
        cfg.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, config=cfg, dtype=torch.float32).to(args.device)
    model.eval()

    sae_state_path = Path(args.sae_state)
    basis, var = load_basis(args.units, mode=args.mode, sae_state=sae_state_path)
    basis = basis.to(args.device)  # (d_model, k)
    sae = load_sae(sae_state_path, model.get_output_embeddings().weight.shape[1]).to(args.device)

    data = tok(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**data, output_hidden_states=True)
        seq_len = int(data["input_ids"].shape[1])
        token_index = resolve_token_index(seq_len, -1)
        inject_token_index = resolve_token_index(seq_len, args.inject_at_token_index)
        prompt_vector, prompt_meta = build_prompt_vector(
            hidden_states=out.hidden_states,
            source_layer=args.prompt_vector_source_layer,
            seq_len=seq_len,
            span_mode=args.prompt_vector_token_span,
            last_n=args.prompt_vector_last_n,
            vector_mode=args.prompt_vector_mode,
            sae=sae,
            basis=basis,
        )
        h_raw = out.hidden_states[args.layer][0, token_index, :]  # residual på sista token
        injection_delta_norm = None
        if prompt_vector is not None and args.layer == args.inject_at_layer and token_index == inject_token_index:
            h_raw, injection_delta_norm = apply_injection(
                h_raw,
                prompt_vector,
                inject_mode=args.inject_mode,
                inject_alpha=args.inject_alpha,
            )
        if args.use_sae_reconstruction:
            h, _ = sae(h_raw.unsqueeze(0))
            h = h.squeeze(0)
            final_ln = getattr(getattr(model, "transformer", None), "ln_f", None)
            lens_input = h.unsqueeze(0)
            if final_ln is not None:
                lens_input = final_ln(lens_input)
            logits = model.get_output_embeddings()(lens_input).squeeze(0)
        else:
            h = h_raw
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
        "used_sae_reconstruction": bool(args.use_sae_reconstruction),
        "prompt_vector_meta": None if prompt_meta is None else {
            "source_layer": prompt_meta.source_layer,
            "span_start": prompt_meta.span_start,
            "span_end": prompt_meta.span_end,
            "token_count": prompt_meta.token_count,
            "vector_mode": prompt_meta.vector_mode,
            "prompt_vector_norm": prompt_meta.prompt_vector_norm,
        },
        "injection": {
            "layer": args.inject_at_layer if prompt_vector is not None else None,
            "token_index": inject_token_index if prompt_vector is not None else None,
            "alpha": args.inject_alpha,
            "mode": args.inject_mode,
            "delta_norm": injection_delta_norm,
        },
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
