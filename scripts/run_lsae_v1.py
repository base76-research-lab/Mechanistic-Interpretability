#!/usr/bin/env python3
"""
run_lsae_v1.py

Minimal L-SAE v1 training:
- Reconstruction loss + L1 sparsity
- Optional logit-lens supervision on next-token prediction from reconstructed hidden state

Outputs:
- activations.pt (hidden states and targets)
- sae_weights.pt
- metrics.json
- top_features.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "prompts.txt"
OUT_DIR = ROOT / "experiments" / "exp_001_sae_v4_lsae_v1"


def load_prompts(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


@torch.no_grad()
def collect_hidden_and_targets(model, tokenizer, prompts, layer_idx: int, device: str = "cpu"):
    """Collect hidden states at a layer and next-token targets."""
    model.to(device)
    model.eval()
    hs_list = []
    targets_list = []
    tok_ids = []
    for p in prompts:
        tokens = tokenizer(p, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = model(**tokens, output_hidden_states=True)
        hidden = out.hidden_states[layer_idx][0]  # (seq, d_model)
        input_ids = tokens["input_ids"][0]        # (seq,)
        if hidden.size(0) < 2:
            continue
        hs_list.append(hidden[:-1].cpu())        # drop last, no target
        targets_list.append(input_ids[1:].cpu()) # predict next token
        tok_ids.append(input_ids.cpu())
    if not hs_list:
        raise RuntimeError("No sequences with length >=2 were collected.")
    hs = torch.cat(hs_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    tok_cat = torch.cat(tok_ids, dim=0)
    return hs, targets, tok_cat


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z


def train_lsae(
    hs: torch.Tensor,
    targets: torch.Tensor,
    ln_f: nn.Module,
    lm_head: nn.Module,
    *,
    d_hidden: int,
    steps: int,
    lr: float,
    l1: float,
    lens_weight: float,
    device: str,
):
    sae = SparseAutoencoder(hs.size(1), d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    hs = hs.to(device)
    targets = targets.to(device)
    loss_hist = []
    ce_loss = nn.CrossEntropyLoss()

    for step in range(steps):
        opt.zero_grad()
        recon, z = sae(hs)
        mse = torch.mean((recon - hs) ** 2)
        l1_term = l1 * torch.mean(torch.abs(z))
        loss = mse + l1_term

        if lens_weight > 0:
            with torch.no_grad():
                recon_ln = ln_f(recon)
            logits = lm_head(recon_ln)
            lens_ce = ce_loss(logits, targets)
            loss = loss + lens_weight * lens_ce
        else:
            lens_ce = torch.tensor(0.0, device=device)

        loss.backward()
        opt.step()

        if step % max(1, steps // 10) == 0:
            loss_hist.append(
                (
                    step,
                    float(mse.item()),
                    float(l1_term.item()),
                    float(lens_ce.item()),
                    float(loss.item()),
                )
            )
    sae.cpu()
    return sae, loss_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--dict-size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--lens-weight", type=float, default=1e-2, help="Weight for logit-lens CE term")
    ap.add_argument("--layernorm", action="store_true", help="Apply LayerNorm to activations before SAE")
    ap.add_argument("--suffix", type=str, default="", help="Optional suffix for output dir")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--prompts", type=str, default=str(DATA))
    args = ap.parse_args()

    out_dir = OUT_DIR if not args.suffix else OUT_DIR.with_name(OUT_DIR.name + args.suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.prompts))
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, output_hidden_states=True)

    hs, targets, tok_cat = collect_hidden_and_targets(model, tokenizer, prompts, layer_idx=args.layer, device=args.device)
    if args.layernorm:
        hs = (hs - hs.mean(dim=0, keepdim=True)) / (hs.std(dim=0, keepdim=True) + 1e-6)
    torch.save({"acts": hs, "targets": targets, "token_ids": tok_cat}, out_dir / "activations.pt")
    print(f"Activations shape: {tuple(hs.shape)} saved to {out_dir}/activations.pt")

    ln_f = getattr(model.transformer, "ln_f")
    lm_head = model.get_output_embeddings()
    sae, loss_hist = train_lsae(
        hs,
        targets,
        ln_f,
        lm_head,
        d_hidden=args.dict_size,
        steps=args.steps,
        lr=args.lr,
        l1=args.l1,
        lens_weight=args.lens_weight,
        device=args.device,
    )
    torch.save(sae.state_dict(), out_dir / "sae_weights.pt")

    with torch.no_grad():
        z_full = torch.relu(sae.encoder(hs))
    mean_abs_z = float(z_full.abs().mean())
    frac_zero = float((z_full < 1e-6).float().mean())
    frac_small = float((z_full < 1e-3).float().mean())

    metrics = {
        "model": args.model,
        "layer": args.layer,
        "dict_size": args.dict_size,
        "steps": args.steps,
        "lr": args.lr,
        "l1": args.l1,
        "lens_weight": args.lens_weight,
        "loss_history": loss_hist,
        "sparsity": {
            "mean_abs_z": mean_abs_z,
            "frac_zero_lt1e-6": frac_zero,
            "frac_small_lt1e-3": frac_small,
        },
        "activations_shape": list(hs.shape),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("L-SAE v1 training done")
    print("Last recorded step:", loss_hist[-1] if loss_hist else "n/a")
    print("Sparsity: mean|z|={:.4f}, zero<1e-6={:.3f}, small<1e-3={:.3f}".format(mean_abs_z, frac_zero, frac_small))

    # top features (top 10 units, top 5 tokens each)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    unit_means = z_full.mean(dim=0)
    top_units = torch.topk(unit_means, k=min(10, z_full.size(1))).indices
    top_features = []
    for u in top_units:
        vals = z_full[:, u]
        topk = torch.topk(vals, k=min(5, vals.numel()))
        entries = []
        for score_idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            token_id = tok_cat[score_idx].item()
            token_str = tokenizer.decode([token_id]).strip()
            entries.append({"token_id": token_id, "token": token_str, "value": score})
        top_features.append({"unit": int(u), "mean_activation": float(unit_means[u]), "top_tokens": entries})
    (out_dir / "top_features.json").write_text(json.dumps(top_features, indent=2))
    print(f"Top features saved to {out_dir}/top_features.json")


if __name__ == "__main__":
    main()
