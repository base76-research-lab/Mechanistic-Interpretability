from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def entropy_from_logits(logits: torch.Tensor) -> float:
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        return float(-(probs * torch.log(probs + 1e-9)).sum().item())


def gap_top2(logits: torch.Tensor) -> float:
    with torch.no_grad():
        topv, _ = torch.topk(logits, k=2)
        return float(topv[0] - topv[1])


def decode_topk(tokenizer, logits: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
    topv, topi = torch.topk(logits, k=k)
    out = []
    for v, i in zip(topv, topi):
        tid = int(i.item())
        out.append(
            {"token": tokenizer.decode([tid]).strip() or "<space>", "token_id": tid, "logit": float(v.item())}
        )
    return out


def load_panel(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


class SparseAutoencoder(torch.nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = torch.nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z


def load_sae(path: Path, d_model: int) -> SparseAutoencoder:
    state = torch.load(path, map_location="cpu")
    d_hidden = state["encoder.weight"].shape[0]
    sae = SparseAutoencoder(d_model, d_hidden)
    sae.load_state_dict(state)
    sae.eval()
    return sae


def run_trace(
    prompt_jsonl: Path,
    model_name: str,
    layers: List[int],
    device: str,
    out_dir: Path,
    run_name: str,
    topk: int,
    store_projections: bool,
    sae_state: Path | None,
    sae_topk: int,
) -> Path:
    device_t = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, output_hidden_states=True
    ).to(device_t)
    model.eval()

    sae = None
    if sae_state is not None:
        sae = load_sae(sae_state, model.config.n_embd).to(device_t)

    panel = load_panel(prompt_jsonl)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"

    mlp_cache: Dict[int, torch.Tensor] = {}

    def make_mlp_hook(layer_idx: int):
        def hook(_module, _inp, out):
            mlp_cache[layer_idx] = out.detach()
        return hook

    handles = []
    for li in layers:
        handles.append(model.transformer.h[li].mlp.register_forward_hook(make_mlp_hook(li)))

    with trace_path.open("w") as f_out:
        for sample in panel:
            mlp_cache.clear()
            prompt = sample["prompt"]
            regime = sample.get("regime")
            inputs = tokenizer(prompt, return_tensors="pt").to(device_t)
            with torch.no_grad():
                out = model(**inputs)
            hidden_states = out.hidden_states
            attentions = out.attentions

            seq_len = inputs["input_ids"].shape[1]
            prev_entropy_per_layer: Dict[int, float] = {}

            pca_coords = {}
            if store_projections:
                for layer_idx in layers:
                    hs = hidden_states[layer_idx].squeeze(0)  # (seq, hidden)
                    hs_center = hs - hs.mean(dim=0, keepdim=True)
                    U, S, V = torch.pca_lowrank(hs_center, q=min(2, hs_center.shape[1]))
                    coords = torch.matmul(hs_center, V[:, :2])
                    pca_coords[layer_idx] = coords  # (seq, 2)

            for layer_idx in layers:
                hs = hidden_states[layer_idx]
                mlp_out = mlp_cache.get(layer_idx)
                attn = attentions[layer_idx]
                coords_layer = pca_coords.get(layer_idx)
                sae_z = None
                if sae is not None:
                    with torch.no_grad():
                        _, z_full = sae(hs.squeeze(0))  # (seq, d_hidden)
                    sae_z = z_full
                for tok_idx in range(seq_len):
                    h_vec = hs[0, tok_idx, :]
                    lens_logits = model.lm_head(h_vec)
                    ent = entropy_from_logits(lens_logits)
                    gap = gap_top2(lens_logits)
                    topk_tokens = decode_topk(tokenizer, lens_logits, k=topk)

                    prev_ent = prev_entropy_per_layer.get(layer_idx)
                    d_ent = None if prev_ent is None else ent - prev_ent
                    prev_entropy_per_layer[layer_idx] = ent

                    attn_weights = attn[0, :, tok_idx, :]
                    attn_entropy = float(-(attn_weights * (attn_weights + 1e-9).log()).sum().item())

                    sae_top = None
                    if sae_z is not None:
                        z_tok = sae_z[tok_idx]
                        topv, topi = torch.topk(z_tok, k=min(sae_topk, z_tok.numel()))
                        sae_top = [
                            {"feature": int(i.item()), "activation": float(v.item())}
                            for v, i in zip(topv, topi)
                        ]

                    rec: Dict[str, Any] = {
                        "prompt_id": sample.get("id"),
                        "regime": regime,
                        "layer": layer_idx,
                        "token_index": tok_idx,
                        "token": tokenizer.decode([int(inputs["input_ids"][0, tok_idx])]),
                        "entropy": ent,
                        "gap_top2": gap,
                        "entropy_delta_vs_prev_token": d_ent,
                        "topk": topk_tokens,
                        "hidden_sha": sha256_tensor(h_vec),
                        "mlp_sha": sha256_tensor(mlp_out[0, tok_idx, :]) if mlp_out is not None else None,
                        "attn_entropy": attn_entropy,
                    }
                    if coords_layer is not None:
                        rec["pca_x"] = float(coords_layer[tok_idx, 0].item())
                        rec["pca_y"] = float(coords_layer[tok_idx, 1].item())
                    if sae_top is not None:
                        rec["sae_top"] = sae_top
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for h in handles:
        h.remove()

    return trace_path


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    default_panel = root / "data" / "prompts_observability_panel_2026-03-07.jsonl"
    default_out = root / "experiments" / "exp_004_unified_observability_stack"
    ap.add_argument("--prompt-jsonl", type=str, default=str(default_panel))
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--layers", nargs="+", type=int, default=[1, 6, 9, 11])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default=str(default_out))
    ap.add_argument("--run-name", type=str, default="")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--store-projections", action="store_true")
    ap.add_argument("--sae-state", type=str, default=None, help="optional SAE weights for feature activations")
    ap.add_argument("--sae-topk", type=int, default=8)
    args = ap.parse_args()

    run_name = args.run_name or f"transformer_oscilloscope_{Path(args.prompt_jsonl).stem}"
    out_dir = Path(args.out_dir) / run_name
    trace_path = run_trace(
        prompt_jsonl=Path(args.prompt_jsonl),
        model_name=args.model,
        layers=[int(l) for l in args.layers],
        device=args.device,
        out_dir=out_dir,
        run_name=run_name,
        topk=args.topk,
        store_projections=args.store_projections,
        sae_state=Path(args.sae_state) if args.sae_state else None,
        sae_topk=args.sae_topk,
    )
    print(f"Saved trace: {trace_path}")


if __name__ == "__main__":
    main()
