#!/usr/bin/env python3
"""
patch_demo.py
- Laddar GPT-2 small och SAE v3
- Patcher en SAE-unit på valfritt lager (default layer5 unit132) på sista token i prompten
- Rapporterar P(target) före/efter samt top-5 delta

Usage:
python3 scripts/patch_demo.py --unit 132 --scale 5.0 --prompt "king is to queen as man is to" --target " woman"
"""
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAE_STATE = ROOT/'experiments/exp_001_sae_v3/sae_weights.pt'


def load_sae_vec(unit, scale):
    state = torch.load(SAE_STATE)
    dec = state['decoder.weight']  # (768, dict)
    return dec[:, unit] * scale


def run_patch(prompt, target, unit, scale, layer_idx=5, device='cpu'):
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device).eval()
    patch_vec = load_sae_vec(unit, scale).to(device)

    inputs = tok(prompt, return_tensors='pt').to(device)
    target_id = tok.encode(target.strip())[0]

    def hook_fn(module, inputs, output):
        # output may be hidden or (hidden, attn) depending on config
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
        else:
            return (hidden, *rest)

    def forward(patch=False):
        handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn if patch else (lambda m,i,o:o))
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        handle.remove()
        return torch.softmax(logits, -1)

    p_orig = forward(False)
    p_patch = forward(True)
    delta = p_patch - p_orig
    top_vals, top_idx = torch.topk(delta, 5)
    return {
        'p_orig': float(p_orig[target_id]),
        'p_patch': float(p_patch[target_id]),
        'delta_top': [(tok.decode([i]).strip(), float(v)) for v,i in zip(top_vals, top_idx)]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', type=int, default=132)
    ap.add_argument('--scale', type=float, default=5.0)
    ap.add_argument('--layer', type=int, default=5)
    ap.add_argument('--prompt', type=str, default='king is to queen as man is to')
    ap.add_argument('--target', type=str, default=' woman')
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    res = run_patch(args.prompt, args.target, args.unit, args.scale, layer_idx=args.layer, device=args.device)
    print(f"Prompt: {args.prompt}")
    print(f"Target: {args.target.strip()}")
    print(f"Unit {args.unit} scale {args.scale} → P(target): {res['p_orig']:.6f} → {res['p_patch']:.6f}")
    print("Top5 delta:")
    for t,v in res['delta_top']:
        print(f"  {t!r}: {v:+.6f}")

if __name__ == '__main__':
    main()
