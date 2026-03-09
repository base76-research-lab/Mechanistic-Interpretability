#!/usr/bin/env python3
"""
state_rollout.py — rollout av latent feature-state med SAE-subspace

Idé: behandla modellen som en state-prediktor i feature-space.
För en given prompt:
  - Projektera sista token-residualen (lager 5) på ett valt SAE-kluster (mean/PC1/PC2)
  - Stega fram N steg genom att greedily välja nästa token och logga det nya state
  - Rapporterar coords per steg + logit-entropi + valt token; sparar JSON

Exempel:
python3 scripts/state_rollout.py --prompt "the opposite of hot is" \
  --units 472 468 57 156 346 --steps 4 --mode pc2
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SAE_STATE = ROOT / "experiments/exp_001_sae_v3/sae_weights.pt"
OUT_JSON = ROOT / "experiments/exp_001_sae_v3/state_rollout.json"


def load_basis(units: List[int], mode: str = "pc2") -> Tuple[torch.Tensor, List[float]]:
    """
    Returnerar basis (d_model x k) och förklarad varians (om PCA).
    mode:
      mean  -> 1-vektor (medel)
      pc1   -> 1-vektor (PC1)
      pc2   -> 2-vektor (PC1, PC2)
    """
    state = torch.load(SAE_STATE, map_location="cpu")
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


def project(h: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    # h: (d_model,), basis: (d_model, k)
    return torch.matmul(h, basis)


def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    return float(-(probs * torch.log(probs + 1e-9)).sum())


def rollout(prompt: str, units: List[int], mode: str, steps: int, device: str = "cpu"):
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    basis, var = load_basis(units, mode=mode)
    basis = basis.to(device)  # (d_model, k)

    ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    records = []

    for step in range(steps + 1):
        inputs = tok.decode(ids)
        data = tok(inputs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**data, output_hidden_states=True)
            h = out.hidden_states[5][0, -1, :]  # sista token, lager 5
            logits = out.logits[0, -1, :]
        coords = project(h, basis)  # (k,)
        ent = entropy_from_logits(logits)
        topk = torch.topk(logits, k=5)
        top_tokens = [(tok.decode([i]).strip() or "<space>", float(logits[i])) for i in topk.indices]

        rec = {
            "step": step,
            "context": inputs,
            "coords": [float(c) for c in coords],
            "logit_entropy": ent,
            "top_tokens": top_tokens,
        }
        records.append(rec)

        if step == steps:
            break
        # greedy next token
        next_id = int(torch.argmax(logits))
        ids.append(next_id)

    return {
        "prompt": prompt,
        "units": units,
        "mode": mode,
        "steps": steps,
        "explained_var": var,
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--units", nargs="+", type=int, default=[472, 468, 57, 156, 346])
    ap.add_argument("--mode", choices=["mean", "pc1", "pc2"], default="pc2")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    result = rollout(args.prompt, args.units, args.mode, args.steps, device=args.device)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    print(f"Prompt: {args.prompt}")
    print(f"Mode: {args.mode}, units={args.units}, steps={args.steps}")
    for r in result["records"]:
        print(f" step {r['step']}: coords={r['coords']}, H={r['logit_entropy']:.3f}, top1={r['top_tokens'][0]}")
    print("Saved →", OUT_JSON)


if __name__ == "__main__":
    main()
