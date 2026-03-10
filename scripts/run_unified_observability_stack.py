#!/usr/bin/env python3
"""
run_unified_observability_stack.py

Unified microscopy stack v1:
- SAE feature telemetry
- logit lens baseline
- frontier metrics
- aligned recorder output

Outputs:
- JSONL trace file with one record per prompt/layer/token-step
- JSON metadata file for reproducibility and review
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from candidate_front_metrics import compute_metrics as compute_candidate_front_metrics
from prompt_vector_injection import apply_injection, build_prompt_vector, resolve_token_index

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PANEL = ROOT / "data" / "prompts_observability_panel_2026-03-07.jsonl"
DEFAULT_SAE_STATE = ROOT / "experiments" / "exp_001_sae_v3" / "sae_weights.pt"
DEFAULT_OUT_DIR = ROOT / "experiments" / "exp_004_unified_observability_stack"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_panel(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu(self.encoder(x))
        recon = self.decoder(z)
        return recon, z


def load_sae(sae_state_path: Path, d_model: int) -> SparseAutoencoder:
    state = torch.load(sae_state_path, map_location="cpu")
    d_hidden = state["encoder.weight"].shape[0]
    sae = SparseAutoencoder(d_model, d_hidden)
    sae.load_state_dict(state)
    sae.eval()
    return sae


def load_basis(units: list[int], mode: str, sae_state_path: Path) -> tuple[torch.Tensor, list[float]]:
    state = torch.load(sae_state_path, map_location="cpu")
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


def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    return float(-(probs * torch.log(probs + 1e-9)).sum())


def get_layer_vector(hidden_states: tuple[torch.Tensor, ...], layer: int, token_index: int) -> torch.Tensor:
    return hidden_states[layer][0, token_index, :]


def apply_logit_lens(model: AutoModelForCausalLM, hidden_vec: torch.Tensor) -> torch.Tensor:
    x = hidden_vec.unsqueeze(0)
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "ln_f"):
        x = transformer.ln_f(x)
    lm_head = model.get_output_embeddings()
    logits = lm_head(x).squeeze(0)
    return logits


def project(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.matmul(vec, basis)


def decode_topk(tokenizer: AutoTokenizer, logits: torch.Tensor, topk: int) -> list[dict[str, Any]]:
    topv, topi = torch.topk(logits, k=topk)
    rows = []
    for logit, idx in zip(topv, topi):
        token_id = int(idx.item())
        rows.append(
            {
                "token_id": token_id,
                "token": tokenizer.decode([token_id]).strip() or "<space>",
                "logit": float(logit.item()),
            }
        )
    return rows


def project_candidates(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, lens_logits: torch.Tensor, basis: torch.Tensor, topk: int
) -> list[dict[str, Any]]:
    topv, topi = torch.topk(lens_logits, k=topk)
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
                "coords": [float(x) for x in coords.detach().cpu().tolist()],
            }
        )
    return rows


def top_sae_features(sae: SparseAutoencoder, hidden_vec: torch.Tensor, topn: int) -> list[dict[str, Any]]:
    with torch.no_grad():
        z = torch.relu(sae.encoder(hidden_vec.unsqueeze(0))).squeeze(0)
    topv, topi = torch.topk(z, k=min(topn, z.numel()))
    return [
        {"feature": int(idx.item()), "activation": float(val.item())}
        for idx, val in zip(topi, topv)
    ]


def make_trace_record(
    *,
    sample: dict[str, Any],
    model_name: str,
    layer: int,
    token_index: int,
    intervention_state: str,
    hidden_vec: torch.Tensor,
    field_coords: torch.Tensor,
    lens_logits: torch.Tensor,
    lens_topk: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top_features: list[dict[str, Any]],
    basis_var: list[float],
    prompt_vector_meta: dict[str, Any] | None,
    injection_target_layer: int | None,
    injection_target_token_index: int | None,
    injection_delta_norm: float | None,
) -> dict[str, Any]:
    field_view_like = {
        "prompt": sample["prompt"],
        "topk": len(candidates),
        "field_coords": [float(x) for x in field_coords.detach().cpu().tolist()],
        "operator_strength": float(torch.norm(field_coords).item()),
        "state_norm": float(torch.norm(field_coords).item()),
        "logit_entropy": entropy_from_logits(lens_logits),
        "gap_state_to_candidates": None,
        "candidates": candidates,
    }
    candidate_metrics = compute_candidate_front_metrics(field_view_like, Path(sample["id"]))
    record = {
        "prompt_id": sample["id"],
        "prompt": sample["prompt"],
        "regime": sample["regime"],
        "stratum": sample["stratum"],
        "model": model_name,
        "layer": layer,
        "token_index": token_index,
        "intervention_state": intervention_state,
        "residual_norm": float(torch.norm(hidden_vec).item()),
        "subspace_coords": [float(x) for x in field_coords.detach().cpu().tolist()],
        "subspace_operator_strength": float(torch.norm(field_coords).item()),
        "sae_top_features": top_features,
        "lens_entropy": field_view_like["logit_entropy"],
        "lens_topk": lens_topk,
        "frontier_coherence": candidate_metrics["candidate_coherence"],
        "frontier_degeneracy": candidate_metrics["degeneracy_ratio_topk"],
        "gap_state_to_candidates": candidate_metrics["state_to_centroid_distance"],
        "candidate_variance": candidate_metrics["candidate_variance"],
        "basis_explained_var": basis_var,
        "prompt_vector_meta": prompt_vector_meta,
        "injection_target_layer": injection_target_layer,
        "injection_target_token_index": injection_target_token_index,
        "injection_delta_norm": injection_delta_norm,
    }
    return record


def add_drift(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        key = f"{row['prompt_id']}::{row['intervention_state']}"
        grouped.setdefault(key, []).append(row)
    out = []
    for _, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["layer"])
        prev = None
        for row in rows:
            current = dict(row)
            if prev is None:
                current["feature_drift_vs_prev_layer"] = None
                current["lens_entropy_delta_vs_prev_layer"] = None
                current["frontier_gap_delta_vs_prev_layer"] = None
                current["operator_strength_delta_vs_prev_layer"] = None
            else:
                prev_feats = {f["feature"]: f["activation"] for f in prev["sae_top_features"]}
                curr_feats = {f["feature"]: f["activation"] for f in current["sae_top_features"]}
                shared = set(prev_feats) | set(curr_feats)
                current["feature_drift_vs_prev_layer"] = float(
                    sum(abs(curr_feats.get(k, 0.0) - prev_feats.get(k, 0.0)) for k in shared)
                )
                current["lens_entropy_delta_vs_prev_layer"] = float(
                    current["lens_entropy"] - prev["lens_entropy"]
                )
                current["frontier_gap_delta_vs_prev_layer"] = float(
                    current["gap_state_to_candidates"] - prev["gap_state_to_candidates"]
                )
                current["operator_strength_delta_vs_prev_layer"] = float(
                    current["subspace_operator_strength"] - prev["subspace_operator_strength"]
                )
            out.append(current)
            prev = current
    return out


def add_trace_aggregates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        key = f"{row['prompt_id']}::{row['intervention_state']}"
        grouped.setdefault(key, []).append(row)
    out = []
    for _, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["layer"])
        dts = 0.0
        for row in rows:
            delta = row.get("lens_entropy_delta_vs_prev_layer")
            if delta is not None:
                dts += abs(float(delta))
        for row in rows:
            current = dict(row)
            current["decision_trajectory_smoothness"] = float(dts)
            out.append(current)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-jsonl", type=str, default=str(DEFAULT_PANEL))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 5, 6, 9, 12])
    parser.add_argument("--sae-state", type=str, default=str(DEFAULT_SAE_STATE))
    parser.add_argument("--units", nargs="+", type=int, default=[472, 468, 57, 156, 346])
    parser.add_argument("--basis-mode", choices=["mean", "pc1", "pc2"], default="pc2")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--top-features", type=int, default=8)
    parser.add_argument("--token-position", choices=["last"], default="last")
    parser.add_argument("--intervention-state", type=str, default="baseline")
    parser.add_argument(
        "--use-sae-reconstruction",
        action="store_true",
        help="If set, replace hidden_vec with SAE reconstruction before metrics (enables L-SAE+R style intervention).",
    )
    parser.add_argument(
        "--prompt-vector-mode",
        choices=["none", "residual_mean", "sae_recon", "basis_recon"],
        default="none",
        help="Build a prompt-level vector and inject it at a fixed layer/token before metrics.",
    )
    parser.add_argument(
        "--prompt-vector-source-layer",
        type=int,
        default=5,
        help="Source layer for building the prompt vector.",
    )
    parser.add_argument(
        "--prompt-vector-token-span",
        choices=["all", "last_n"],
        default="all",
        help="Token span used to build the prompt vector.",
    )
    parser.add_argument(
        "--prompt-vector-last-n",
        type=int,
        default=4,
        help="If token span is last_n, average over this many final prompt tokens.",
    )
    parser.add_argument(
        "--inject-at-layer",
        type=int,
        default=5,
        help="Layer index where the prompt vector is injected.",
    )
    parser.add_argument(
        "--inject-at-token-index",
        type=int,
        default=-1,
        help="Target token index for injection. Negative values are relative to sequence end.",
    )
    parser.add_argument(
        "--inject-alpha",
        type=float,
        default=0.0,
        help="Injection strength. Zero disables the prompt vector even if a mode is selected.",
    )
    parser.add_argument(
        "--inject-mode",
        choices=["add", "mix"],
        default="add",
        help="How to combine the prompt vector with the target hidden state.",
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    panel_path = Path(args.prompt_jsonl)
    sae_state_path = Path(args.sae_state)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = args.run_name or f"stack_{args.model.replace('/', '_')}_{ts}"
    out_dir = DEFAULT_OUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = out_dir / "trace.jsonl"
    metadata_json = out_dir / "metadata.json"

    panel = load_panel(panel_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)
    model.eval()

    d_model = model.get_output_embeddings().weight.shape[1]
    sae = load_sae(sae_state_path, d_model).to(args.device)
    basis, basis_var = load_basis(args.units, args.basis_mode, sae_state_path)
    basis = basis.to(args.device)

    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in panel:
            tokens = tokenizer(sample["prompt"], return_tensors="pt").to(args.device)
            seq_len = int(tokens["input_ids"].shape[1])
            token_index = seq_len - 1
            inject_token_index = resolve_token_index(seq_len, args.inject_at_token_index)

            # First forward to build prompt vector; reuse hidden_states for vector construction
            base_out = model(**tokens, output_hidden_states=True)
            prompt_vector, prompt_meta = build_prompt_vector(
                hidden_states=base_out.hidden_states,
                source_layer=args.prompt_vector_source_layer,
                seq_len=seq_len,
                span_mode=args.prompt_vector_token_span,
                last_n=args.prompt_vector_last_n,
                vector_mode=args.prompt_vector_mode,
                sae=sae,
                basis=basis,
            )

            # If we have a prompt vector and alpha>0, do a causal forward pass with injected state
            injection_delta_norm_global = None
            if prompt_vector is not None and args.inject_alpha != 0.0:
                def hook_factory():
                    def hook(module, inputs, output):
                        hidden = output[0]
                        hidden_mod = hidden.clone()
                        target_vec = hidden_mod[0, inject_token_index, :]
                        injected, delta_norm = apply_injection(
                            target_vec,
                            prompt_vector,
                            inject_mode=args.inject_mode,
                            inject_alpha=args.inject_alpha,
                        )
                        hidden_mod[0, inject_token_index, :] = injected
                        nonlocal injection_delta_norm_global
                        injection_delta_norm_global = delta_norm
                        return (hidden_mod, *output[1:])
                    return hook

                handle = model.transformer.h[args.inject_at_layer].register_forward_hook(hook_factory())
                out = model(**tokens, output_hidden_states=True)
                handle.remove()
            else:
                out = base_out
            for layer in args.layers:
                hidden_vec_raw = get_layer_vector(out.hidden_states, layer, token_index)

                # Optionally route through SAE reconstruction to activate L-SAE+R behavior
                if args.use_sae_reconstruction:
                    with torch.no_grad():
                        recon_vec, z_vec = sae(hidden_vec_raw.unsqueeze(0))
                        hidden_vec = recon_vec.squeeze(0)
                else:
                    hidden_vec = hidden_vec_raw

                field_coords = project(hidden_vec, basis)
                lens_logits = apply_logit_lens(model, hidden_vec)
                lens_topk = decode_topk(tokenizer, lens_logits, args.topk)
                candidates = project_candidates(model, tokenizer, lens_logits, basis, args.topk)
                top_features = top_sae_features(sae, hidden_vec, args.top_features)
                record = make_trace_record(
                    sample=sample,
                    model_name=args.model,
                    layer=layer,
                    token_index=token_index,
                    intervention_state=args.intervention_state,
                    hidden_vec=hidden_vec,
                    field_coords=field_coords,
                    lens_logits=lens_logits,
                    lens_topk=lens_topk,
                    candidates=candidates,
                    top_features=top_features,
                    basis_var=basis_var,
                    prompt_vector_meta=None if prompt_meta is None else {
                        "source_layer": prompt_meta.source_layer,
                        "span_start": prompt_meta.span_start,
                        "span_end": prompt_meta.span_end,
                        "token_count": prompt_meta.token_count,
                        "vector_mode": prompt_meta.vector_mode,
                        "prompt_vector_norm": prompt_meta.prompt_vector_norm,
                    },
                    injection_target_layer=args.inject_at_layer if prompt_vector is not None else None,
                    injection_target_token_index=inject_token_index if prompt_vector is not None else None,
                    injection_delta_norm=injection_delta_norm_global if prompt_vector is not None else None,
                )
                records.append(record)

    records = add_drift(records)
    records = add_trace_aggregates(records)
    with open(trace_jsonl, "w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")

    metadata = {
        "run_name": run_name,
        "timestamp_utc": ts,
        "model": args.model,
        "layers": args.layers,
        "panel": safe_path_label(panel_path),
        "panel_sha256": sha256(panel_path),
        "sae_state": safe_path_label(sae_state_path),
        "sae_state_sha256": sha256(sae_state_path),
        "basis_mode": args.basis_mode,
        "units": args.units,
        "topk": args.topk,
        "top_features": args.top_features,
        "intervention_state": args.intervention_state,
        "use_sae_reconstruction": args.use_sae_reconstruction,
        "prompt_vector_mode": args.prompt_vector_mode,
        "prompt_vector_source_layer": args.prompt_vector_source_layer,
        "prompt_vector_token_span": args.prompt_vector_token_span,
        "prompt_vector_last_n": args.prompt_vector_last_n,
        "inject_at_layer": args.inject_at_layer,
        "inject_at_token_index": args.inject_at_token_index,
        "inject_alpha": args.inject_alpha,
        "inject_mode": args.inject_mode,
        "record_count": len(records),
        "schema": {
            "grain": "one record per prompt_id, layer, token_index, intervention_state",
            "trace_file": safe_path_label(trace_jsonl),
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))

    print(f"Saved trace: {trace_jsonl.relative_to(ROOT)}")
    print(f"Saved metadata: {metadata_json.relative_to(ROOT)}")
    print(f"Records: {len(records)}")


if __name__ == "__main__":
    main()
