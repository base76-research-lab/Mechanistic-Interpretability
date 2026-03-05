#!/usr/bin/env python3
"""
patch_subspace.py — layer-svep för antonym-subspace

- Bygger en subspace-vektor från SAE-decoder (medel eller PC1) för angivet kluster
- Patcher vektorn på sista token i prompten på valda lager
- Valfritt: maskera specificerade attention-heads (per lager) för att se var effekten tas upp
- Rapporterar P(target) före/efter + delta per lager och skriver resultat till JSON

Exempel:
python3 scripts/patch_subspace.py --prompt "the opposite of hot is" --targets cold dark \
    --units 472 468 57 156 346 --layers 3 4 5 6 7 8 9 --mode pc1 --scale 10

Head-mask syntax:
  --mask-heads "5:0,1;6:2,3"   # mask head 0,1 i lager 5 och head 2,3 i lager 6
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SAE_STATE = ROOT / "experiments/exp_001_sae_v3/sae_weights.pt"
OUT_JSON = ROOT / "experiments/exp_001_sae_v3/subspace_patch.json"


def parse_mask(mask_str: str) -> Dict[int, List[int]]:
    """
    "5:0,1;6:2,3" -> {5:[0,1], 6:[2,3]}
    """
    res: Dict[int, List[int]] = {}
    if not mask_str:
        return res
    for block in mask_str.split(";"):
        if not block.strip():
            continue
        layer_s, heads_s = block.split(":")
        res[int(layer_s)] = [int(h) for h in heads_s.split(",") if h.strip() != ""]
    return res


def load_subspace_vec(units: List[int], mode: str = "mean") -> torch.Tensor:
    state = torch.load(SAE_STATE, map_location="cpu")
    dec = state["decoder.weight"]  # (d_model, dict)
    vecs = dec[:, units]  # (d_model, k)
    if mode == "pc1":
        x = vecs - vecs.mean(dim=1, keepdim=True)
        u, s, _ = torch.linalg.svd(x, full_matrices=False)
        return u[:, 0] * s[0]
    return vecs.mean(dim=1)


def head_mask_hook(layer_idx: int, heads: List[int], n_head: int, head_dim: int):
    def hook(module, inputs, output):
        # output: (batch, seq, hidden)
        if not heads:
            return output
        y = output[0] if isinstance(output, tuple) else output
        b, t, h = y.shape
        y_reshaped = y.view(b, t, n_head, head_dim)
        for h_id in heads:
            if 0 <= h_id < n_head:
                y_reshaped[:, :, h_id, :] = 0
        y = y_reshaped.view(b, t, h)
        if isinstance(output, tuple):
            rest = output[1:]
            return (y, *rest)
        return y

    return hook


def block_patch_hook(patch_vec: torch.Tensor):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        hidden = hidden.clone()
        hidden[:, -1, :] += patch_vec
        if rest is None:
            return hidden
        return (hidden, *rest)

    return hook


def run_layer(model, tok, prompt, target_ids, layer_idx, patch_vec, head_masks, device="cpu"):
    # register hooks
    handles = []
    # head mask for this layer?
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head
    if layer_idx in head_masks:
        handles.append(
            model.transformer.h[layer_idx].attn.register_forward_hook(
                head_mask_hook(layer_idx, head_masks[layer_idx], n_head, head_dim)
            )
        )
    # patch hook
    handles.append(model.transformer.h[layer_idx].register_forward_hook(block_patch_hook(patch_vec)))

    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    for h in handles:
        h.remove()
    probs = torch.softmax(logits, -1)
    return {tid: float(probs[tid]) for tid in target_ids}


def run_layer_baseline(model, tok, prompt, target_ids, head_masks, device="cpu"):
    handles = []
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head
    for layer_idx, heads in head_masks.items():
        handles.append(
            model.transformer.h[layer_idx].attn.register_forward_hook(
                head_mask_hook(layer_idx, heads, n_head, head_dim)
            )
        )
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    for h in handles:
        h.remove()
    probs = torch.softmax(logits, -1)
    return {tid: float(probs[tid]) for tid in target_ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--targets", nargs="+", required=True, help="målord (tokens) att mäta logit-shift för")
    ap.add_argument("--units", nargs="+", type=int, default=[472, 468, 57, 156, 346])
    ap.add_argument("--layers", nargs="+", type=int, default=[3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--mode", choices=["mean", "pc1"], default="pc1")
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--mask-heads", type=str, default="", help='t.ex. "5:0,1;6:2,3"')
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = args.device
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    target_ids = [tok.encode(t.strip())[0] for t in args.targets]
    head_masks = parse_mask(args.mask_heads)

    sub_vec = load_subspace_vec(args.units, mode=args.mode).to(device) * args.scale

    # baseline (with any head masks applied)
    base = run_layer_baseline(model, tok, args.prompt, target_ids, head_masks, device=device)

    results = []
    for layer in args.layers:
        probs = run_layer(model, tok, args.prompt, target_ids, layer, sub_vec, head_masks, device=device)
        entry = {"layer": layer}
        for tid, base_p in base.items():
            entry[f"target_{tok.decode([tid]).strip()}_base"] = base_p
            entry[f"target_{tok.decode([tid]).strip()}_patch"] = probs[tid]
            entry[f"delta_{tok.decode([tid]).strip()}"] = probs[tid] - base_p
        results.append(entry)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": args.prompt,
        "targets": args.targets,
        "units": args.units,
        "mode": args.mode,
        "scale": args.scale,
        "layers": args.layers,
        "head_masks": head_masks,
        "results": results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Prompt: {args.prompt}")
    print(f"Units: {args.units} ({args.mode}), scale={args.scale}")
    if head_masks:
        print("Head masks:", head_masks)
    print("Baseline probs:", {k: base[k] for k in base})
    for r in results:
        deltas = {k: v for k, v in r.items() if k.startswith("delta_")}
        print(f"Layer {r['layer']:2d}  deltas {deltas}")
    print("Saved →", OUT_JSON)


if __name__ == "__main__":
    main()
