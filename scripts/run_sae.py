#!/usr/bin/env python3
"""
run_sae.py — exp_001_sae (generic model runner)

- Loads an arbitrary HF model (default: gpt2)
- Extracts hidden_states[layer] for prompts in data/prompts.txt
- Trains a small Sparse Autoencoder (SAE) on the activations
- Saves:
  * experiments/exp_001_sae*/activations.pt
  * .../sae_weights.pt
  * .../metrics.json
- Prints a short summary to stdout; the write-up lives in reports/exp_001_sae.md

Example:
python3 scripts/run_sae.py --model microsoft/phi-2 --layer 12 --dict-size 512 --steps 400 --lr 1e-3 --l1 1e-3 --suffix _phi2
"""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "prompts.txt"
OUT_DIR = ROOT / "experiments" / "exp_001_sae"


def load_prompts(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def collect_activations(model, tokenizer, prompts, layer_idx: int, device: str = "cpu"):
    model.to(device)
    model.eval()
    acts = []
    tok_ids = []
    with torch.no_grad():
        for p in prompts:
            tokens = tokenizer(p, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model(**tokens, output_hidden_states=True)
            hs = out.hidden_states[layer_idx]  # (1, seq, hidden)
            acts.append(hs.squeeze(0))  # (seq, hidden)
        tok_ids.append(tokens["input_ids"].squeeze(0).cpu())  # (seq,)
    acts_cat = torch.cat(acts, dim=0)  # (total_tokens, hidden)
    tok_cat = torch.cat(tok_ids, dim=0)  # (total_tokens,)
    return acts_cat.cpu(), tok_cat


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z


def train_sae(acts, d_hidden=512, steps=500, lr=1e-3, l1=5e-3, device="cpu"):
    sae = SparseAutoencoder(acts.size(1), d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    loss_hist = []
    acts = acts.to(device)
    for step in range(steps):
        opt.zero_grad()
        recon, z = sae(acts)
        mse = torch.mean((recon - acts) ** 2)
        l1_term = l1 * torch.mean(torch.abs(z))
        loss = mse + l1_term
        loss.backward()
        opt.step()
        if step % max(1, steps // 10) == 0:
            loss_hist.append((step, float(mse.item()), float(l1_term.item())))
    sae.cpu()
    return sae, loss_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2", help="HF model id")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--dict-size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--layernorm", action="store_true", help="Apply LayerNorm to activations before SAE")
    ap.add_argument("--suffix", type=str, default="", help="Optional suffix for output dir (e.g., _v2)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--prompts", type=str, default=str(DATA), help="Path to prompts.txt")
    args = ap.parse_args()

    out_dir = OUT_DIR if not args.suffix else OUT_DIR.with_name(OUT_DIR.name + args.suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompts)
    prompts = load_prompts(prompt_path)
    print(f"Loaded {len(prompts)} prompts from {prompt_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, output_hidden_states=True)

    acts, tok_cat = collect_activations(model, tokenizer, prompts, layer_idx=args.layer, device=args.device)
    if args.layernorm:
        acts = (acts - acts.mean(dim=0, keepdim=True)) / (acts.std(dim=0, keepdim=True) + 1e-6)
    torch.save({"acts": acts, "token_ids": tok_cat}, out_dir / "activations.pt")
    print(f"Activations shape: {tuple(acts.shape)} saved to {out_dir}/activations.pt")

    sae, loss_hist = train_sae(acts, d_hidden=args.dict_size, steps=args.steps, lr=args.lr, l1=args.l1, device=args.device)
    torch.save(sae.state_dict(), out_dir / "sae_weights.pt")

    # sparsity metrics
    with torch.no_grad():
        z_full = torch.relu(sae.encoder(acts))
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
        "loss_history": loss_hist,
        "sparsity": {
            "mean_abs_z": mean_abs_z,
            "frac_zero_lt1e-6": frac_zero,
            "frac_small_lt1e-3": frac_small,
        },
        "activations_shape": list(acts.shape),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("SAE training done")
    print("Last mse/l1:", loss_hist[-1])
    print("Sparsity: mean|z|={:.4f}, zero<1e-6={:.3f}, small<1e-3={:.3f}".format(mean_abs_z, frac_zero, frac_small))

    # top features (top 10 units, top 5 tokens each)
    top_features = []
    with torch.no_grad():
        z_all = torch.relu(sae.encoder(acts))
    unit_means = z_all.mean(dim=0)
    top_units = torch.topk(unit_means, k=min(10, z_all.size(1))).indices
    for u in top_units:
        vals = z_all[:, u]
        topk = torch.topk(vals, k=min(5, vals.numel()))
        entries = []
        for score_idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            token_id = tok_cat[score_idx].item()
            token_str = tokenizer.decode([token_id]).strip()
            entries.append({"token_id": token_id, "token": token_str, "value": score})
        top_features.append({"unit": int(u), "mean_activation": float(unit_means[u]), "top_tokens": entries})
    with open(out_dir / "top_features.json", "w") as f:
        json.dump(top_features, f, indent=2)

    print(f"Top features saved to {out_dir}/top_features.json")


if __name__ == "__main__":
    main()
