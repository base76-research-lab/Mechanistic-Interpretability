from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


GENERIC_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "i",
    "he",
    "she",
    "it",
    "they",
    "that",
    "this",
    "these",
    "those",
    "we",
    "you",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
}

GENERIC_SPECIAL = {"<space>"}


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


def is_punctuation_token(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return True
    return all(not ch.isalnum() and ch != "_" for ch in stripped)


def is_generic_token(token: str) -> bool:
    lower = token.strip().lower()
    if lower in GENERIC_SPECIAL or lower in GENERIC_WORDS:
        return True
    return is_punctuation_token(token)


def mean_pairwise_cosine(matrix: torch.Tensor) -> float:
    if matrix.shape[0] < 2:
        return float("nan")
    norms = torch.linalg.norm(matrix, dim=1, keepdim=True).clamp_min(1e-12)
    normalized = matrix / norms
    sim = normalized @ normalized.T
    upper = sim[torch.triu_indices(sim.shape[0], sim.shape[1], offset=1).unbind()]
    return float(upper.mean().item())


def trace_covariance(matrix: torch.Tensor) -> float:
    if matrix.shape[0] < 2:
        return 0.0
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (matrix.shape[0] - 1)
    return float(torch.trace(cov).item())


def project(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.matmul(vec, basis)


def project_candidates(
    model: AutoModelForCausalLM,
    tokenizer,
    lens_logits: torch.Tensor,
    basis: torch.Tensor | None,
    topk: int,
) -> list[dict[str, Any]]:
    topv, topi = torch.topk(lens_logits, k=topk)
    if basis is None:
        return [
            {
                "token": tokenizer.decode([int(idx.item())]).strip() or "<space>",
                "token_id": int(idx.item()),
                "logit": float(logit.item()),
                "coords": [],
            }
            for logit, idx in zip(topv, topi)
        ]
    w_u = model.get_output_embeddings().weight
    rows = []
    for logit, idx in zip(topv, topi):
        token_id = int(idx.item())
        coords = project(w_u[idx], basis)
        rows.append(
            {
                "token": tokenizer.decode([token_id]).strip() or "<space>",
                "token_id": token_id,
                "logit": float(logit.item()),
                "coords": [float(x.item()) for x in coords],
            }
        )
    return rows


def candidate_metrics_from_coords(field_coords: torch.Tensor | None, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates or not candidates[0]["coords"]:
        return {
            "candidate_coherence": None,
            "candidate_variance": None,
            "degeneracy_ratio_topk": None,
            "state_to_centroid_distance": None,
            "generic_count": None,
        }
    coords = torch.tensor([c["coords"] for c in candidates], dtype=torch.float32)
    candidate_centroid = coords.mean(dim=0)
    generic_count = sum(1 for c in candidates if is_generic_token(c["token"]))
    state_to_centroid = None
    if field_coords is not None:
        state_to_centroid = float(torch.linalg.norm(field_coords - candidate_centroid).item())
    return {
        "candidate_coherence": mean_pairwise_cosine(coords),
        "candidate_variance": trace_covariance(coords),
        "degeneracy_ratio_topk": generic_count / len(candidates),
        "state_to_centroid_distance": state_to_centroid,
        "generic_count": generic_count,
    }


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


def load_basis(units: List[int], mode: str, sae_state: Path) -> tuple[torch.Tensor, List[float]]:
    state = torch.load(sae_state, map_location="cpu")
    dec = state["decoder.weight"]
    vecs = dec[:, units]
    if mode == "mean":
        basis = vecs.mean(dim=1, keepdim=True)
        return basis, [1.0]
    x = vecs - vecs.mean(dim=1, keepdim=True)
    u, s, _ = torch.linalg.svd(x, full_matrices=False)
    if mode == "pc1":
        basis = u[:, :1] * s[:1]
        var = (s ** 2) / (s ** 2).sum()
        return basis, [float(var[0])]
    basis = u[:, :2] * s[:2]
    var = (s ** 2) / (s ** 2).sum()
    return basis, [float(var[0]), float(var[1])]


def project(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.matmul(vec, basis)


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
    units: List[int],
    basis_mode: str,
) -> Path:
    device_t = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, output_hidden_states=True
    ).to(device_t)
    model.eval()

    sae = None
    basis = None
    basis_var = None
    if sae_state is not None:
        sae = load_sae(sae_state, model.config.n_embd).to(device_t)
        basis, basis_var = load_basis(units, basis_mode, sae_state)
        basis = basis.to(device_t)

    panel = load_panel(prompt_jsonl)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    metadata_path = out_dir / "metadata.json"

    mlp_cache: Dict[int, torch.Tensor] = {}
    num_blocks = len(model.transformer.h)

    def make_mlp_hook(hidden_state_idx: int):
        def hook(_module, _inp, out):
            mlp_cache[hidden_state_idx] = out.detach()
        return hook

    handles = []
    for li in layers:
        block_idx = li - 1
        if 0 <= block_idx < num_blocks:
            handles.append(model.transformer.h[block_idx].mlp.register_forward_hook(make_mlp_hook(li)))

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
                attn = None
                attn_idx = layer_idx - 1
                if 0 <= attn_idx < len(attentions):
                    attn = attentions[attn_idx]
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
                    candidate_rows = project_candidates(model, tokenizer, lens_logits, basis, topk)
                    field_coords = project(h_vec, basis) if basis is not None else None
                    candidate_metrics = candidate_metrics_from_coords(field_coords, candidate_rows)

                    prev_ent = prev_entropy_per_layer.get(layer_idx)
                    d_ent = None if prev_ent is None else ent - prev_ent
                    prev_entropy_per_layer[layer_idx] = ent

                    attn_entropy = None
                    if attn is not None:
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
                        "lens_entropy": ent,
                        "gap_top2": gap,
                        "entropy_delta_vs_prev_token": d_ent,
                        "topk": topk_tokens,
                        "lens_topk": topk_tokens,
                        "hidden_sha": sha256_tensor(h_vec),
                        "mlp_sha": sha256_tensor(mlp_out[0, tok_idx, :]) if mlp_out is not None else None,
                        "attn_entropy": attn_entropy,
                    }
                    if coords_layer is not None:
                        rec["pca_x"] = float(coords_layer[tok_idx, 0].item())
                        rec["pca_y"] = float(coords_layer[tok_idx, 1].item())
                    if basis is not None:
                        rec["subspace_coords"] = [float(x.item()) for x in field_coords]
                        rec["subspace_operator_strength"] = float(torch.norm(field_coords).item())
                        rec["basis_mode"] = basis_mode
                        rec["basis_explained_var"] = basis_var
                        rec["candidates"] = candidate_rows
                        rec["frontier_coherence"] = candidate_metrics["candidate_coherence"]
                        rec["frontier_degeneracy"] = candidate_metrics["degeneracy_ratio_topk"]
                        rec["gap_state_to_candidates"] = candidate_metrics["state_to_centroid_distance"]
                        rec["candidate_variance"] = candidate_metrics["candidate_variance"]
                    if sae_top is not None:
                        rec["sae_top"] = sae_top
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for h in handles:
        h.remove()

    metadata = {
        "run_name": run_name,
        "model": model_name,
        "layers": layers,
        "device": device,
        "panel": str(prompt_jsonl),
        "record_count": sum(1 for _ in trace_path.open()),
        "store_projections": store_projections,
        "sae_state": str(sae_state) if sae_state is not None else None,
        "sae_topk": sae_topk,
        "units": units if sae_state is not None else None,
        "basis_mode": basis_mode if sae_state is not None else None,
        "trace_file": str(trace_path),
        "observer_class": "read_only",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
    ap.add_argument("--units", nargs="+", type=int, default=[472, 468, 57, 156, 346])
    ap.add_argument("--basis-mode", choices=["mean", "pc1", "pc2"], default="pc2")
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
        units=args.units,
        basis_mode=args.basis_mode,
    )
    print(f"Saved trace: {trace_path}")


if __name__ == "__main__":
    main()
