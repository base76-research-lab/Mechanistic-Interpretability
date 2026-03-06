#!/usr/bin/env python3
"""
opposite_probe.py — residual-inspektion för "opposite"-subspacet

Gör tre saker:
1) PCA på decoder-vektorer för ett valt feature-kluster (default antonym-kluster)
2) Kosinus-toppar mot embeddings (W_E) och utmatningsvikter (W_U/c_proj) → visar vilka ord/kanaler riktningen pekar på
3) Per-token residual-profil för en prompt: projektion på PC1/PC2, residualnorm och logitentropi

Sparar resultat till experiments/exp_001_sae_v3/opposite_probe.json och skriver kort rapport till stdout.

Exempel:
python3 scripts/opposite_probe.py --prompt "the opposite of hot is" --units 472 468 57 156 346
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SAE_STATE = ROOT / "experiments/exp_001_sae_v3/sae_weights.pt"
OUT_JSON = ROOT / "experiments/exp_001_sae_v3/opposite_probe.json"


def load_decoder(units):
    state = torch.load(SAE_STATE, map_location="cpu")
    dec = state["decoder.weight"]  # (d_model, dict)
    vecs = dec[:, units]  # (d_model, k)
    return vecs


def pca(vecs, k=2):
    x = vecs - vecs.mean(dim=1, keepdim=True)
    u, s, _ = torch.linalg.svd(x, full_matrices=False)
    comps = u[:, :k]  # (d_model, k)
    var = (s ** 2) / (s ** 2).sum()
    return comps, var[:k]


def top_cos(matrix, vec, labels, k=8):
    # matrix: (n, d), vec: (d)
    sim = torch.nn.functional.cosine_similarity(matrix, vec.unsqueeze(0), dim=1)
    topv, topi = torch.topk(sim, k=k)
    return [(labels[i], float(topv[j])) for j, i in enumerate(topi.tolist())]


def residual_profile(model, tok, prompt, proj):
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[5][0]  # (seq, d_model) efter block 5
        logits = out.logits[0]  # (seq, vocab)

    proj_vals = hs @ proj  # (seq, k)
    norms = hs.norm(dim=1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

    tokens = inputs["input_ids"][0]
    rows = []
    for i, tid in enumerate(tokens.tolist()):
        rows.append({
            "pos": i,
            "token": tok.decode([tid]).replace("\n", "\\n"),
            "proj_pc1": float(proj_vals[i, 0]),
            "proj_pc2": float(proj_vals[i, 1]) if proj_vals.shape[1] > 1 else 0.0,
            "res_norm": float(norms[i]),
            "logit_entropy": float(entropy[i]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="*", type=int, default=[472, 468, 57, 156, 346], help="SAE-unit IDs för klustret")
    ap.add_argument("--prompt", type=str, default="the opposite of hot is")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    vecs = load_decoder(args.units)
    comps, var = pca(vecs, k=2)
    mean_vec = vecs.mean(dim=1)

    wte = model.transformer.wte.weight  # (vocab, d_model)
    cproj = model.transformer.h[5].mlp.c_proj.weight  # (d_model, d_model)

    vocab_labels = [tok.decode([i]).strip() or "<space>" for i in range(wte.size(0))]
    top_tokens_pc1 = top_cos(wte, comps[:, 0], vocab_labels, k=args.topk)
    top_tokens_pc2 = top_cos(wte, comps[:, 1], vocab_labels, k=args.topk)
    top_tokens_mean = top_cos(wte, mean_vec, vocab_labels, k=args.topk)

    top_cproj_pc1 = top_cos(cproj, comps[:, 0], [f"c_proj_row_{i}" for i in range(cproj.size(0))], k=args.topk)
    top_cproj_pc2 = top_cos(cproj, comps[:, 1], [f"c_proj_row_{i}" for i in range(cproj.size(0))], k=args.topk)

    profile = residual_profile(model, tok, args.prompt, comps)

    result = {
        "units": args.units,
        "explained_var": [float(v) for v in var],
        "top_tokens_pc1": top_tokens_pc1,
        "top_tokens_pc2": top_tokens_pc2,
        "top_tokens_mean": top_tokens_mean,
        "top_cproj_pc1": top_cproj_pc1,
        "top_cproj_pc2": top_cproj_pc2,
        "prompt": args.prompt,
        "profile": profile,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("Units:", args.units)
    print("Explained var PC1/PC2:", [f"{v:.3f}" for v in var])
    print("Top tokens PC1:", top_tokens_pc1[:5])
    print("Top tokens PC2:", top_tokens_pc2[:5])
    print("Top tokens mean:", top_tokens_mean[:5])
    print("Saved →", OUT_JSON)


if __name__ == "__main__":
    main()
