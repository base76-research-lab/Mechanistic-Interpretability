#!/usr/bin/env python3
"""
run_field_view_logged.py

Wrapper som kör field_view.py och loggar allt som behövs för spårbarhet:
- timestamp + scenario-namn
- git head/status (om repo finns)
- SHA256 för SAE-viktfil och output
- Prompt, units, mode, topk, device
- Extraherar metrik från field_view-output (risk, H, gap, coords, operator_strength)
- Lagrar:
    artifacts/<ts>/<name>.json    (rå-output från field_view)
    runs/<ts>.json                (strukturerad logg)
    runs/<ts>.md                  (snabb textöversikt)

Användning:
    python3 scripts/run_field_view_logged.py \\
        --scenario math_det \\
        --prompt "2 + 2 =" \\
        --mode pc2 --topk 10 --device cpu

Default-units: antonym-klustret (472/468/57/156/346), mode=pc2, topk=8.
"""

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from candidate_front_metrics import compute_metrics as compute_candidate_front_metrics

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments" / "exp_001_sae_v3"
ARTIFACT_DIR = EXP_DIR / "artifacts"
RUNS_DIR = EXP_DIR / "runs"
DEFAULT_SAE_STATE = EXP_DIR / "sae_weights.pt"
FIELD_VIEW_SCRIPT = ROOT / "scripts" / "field_view.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(cwd: Path) -> dict:
    info = {"head": None, "status": None}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=cwd, text=True).strip()
        info["head"] = head
        info["status"] = status
    except Exception:
        pass
    return info


def safe_path_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="Prompt att köra")
    ap.add_argument("--scenario", required=True, help="Kort etikett, t.ex. math_det / analogy_reason / hallucination")
    ap.add_argument("--units", nargs="+", type=int, default=[472, 468, 57, 156, 346])
    ap.add_argument("--mode", choices=["mean", "pc1", "pc2"], default="pc2")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="gpt2", help="HF model-id eller provider-spec id")
    ap.add_argument("--layer", type=int, default=5, help="hidden_states-index att projicera")
    ap.add_argument("--sae_state", default=str(DEFAULT_SAE_STATE), help="Path till SAE-viktfil")
    ap.add_argument(
        "--use-sae-reconstruction",
        action="store_true",
        help="Route the hidden vector through SAE reconstruction before field/lens metrics.",
    )
    ap.add_argument("--prompt-vector-mode", choices=["none", "residual_mean", "sae_recon", "basis_recon"], default="none")
    ap.add_argument("--prompt-vector-source-layer", type=int, default=5)
    ap.add_argument("--prompt-vector-token-span", choices=["all", "last_n"], default="all")
    ap.add_argument("--prompt-vector-last-n", type=int, default=4)
    ap.add_argument("--inject-at-layer", type=int, default=5)
    ap.add_argument("--inject-at-token-index", type=int, default=-1)
    ap.add_argument("--inject-alpha", type=float, default=0.0)
    ap.add_argument("--inject-mode", choices=["add", "mix"], default="add")
    ap.add_argument("--degenerate-threshold", type=float, default=0.7, help="Threshold for candidate_front.degenerate")
    args = ap.parse_args()

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"field_view_{args.scenario}__{ts}"

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Kör field_view
    cmd = [
        "python3",
        str(FIELD_VIEW_SCRIPT),
        "--prompt",
        args.prompt,
        "--units",
        *[str(u) for u in args.units],
        "--mode",
        args.mode,
        "--topk",
        str(args.topk),
        "--name",
        name,
        "--model",
        args.model,
        "--layer",
        str(args.layer),
        "--sae_state",
        args.sae_state,
        "--device",
        args.device,
    ]
    if args.use_sae_reconstruction:
        cmd.append("--use-sae-reconstruction")
    if args.prompt_vector_mode != "none":
        cmd.extend(
            [
                "--prompt-vector-mode",
                args.prompt_vector_mode,
                "--prompt-vector-source-layer",
                str(args.prompt_vector_source_layer),
                "--prompt-vector-token-span",
                args.prompt_vector_token_span,
                "--prompt-vector-last-n",
                str(args.prompt_vector_last_n),
                "--inject-at-layer",
                str(args.inject_at_layer),
                "--inject-at-token-index",
                str(args.inject_at_token_index),
                "--inject-alpha",
                str(args.inject_alpha),
                "--inject-mode",
                args.inject_mode,
            ]
        )
    print("Kör:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)

    # Flytta output till artifacts/<ts>/
    src_json = EXP_DIR / f"{name}.json"
    dest_dir = ARTIFACT_DIR / ts
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_json = dest_dir / f"{name}.json"
    shutil.move(src_json, dest_json)

    # Läs metrik
    data = json.loads(dest_json.read_text())
    candidate_front = compute_candidate_front_metrics(data, dest_json)
    candidate_front["degenerate"] = candidate_front["degeneracy_ratio_topk"] > args.degenerate_threshold
    candidate_metrics_path = dest_dir / "candidate_front_metrics.json"
    candidate_metrics_path.write_text(json.dumps(candidate_front, indent=2))

    sae_state_path = Path(args.sae_state)
    operator_strength = data.get("operator_strength", data.get("state_norm"))
    run_record = {
        "timestamp_utc": ts,
        "scenario": args.scenario,
        "prompt": args.prompt,
        "model": args.model,
        "layer": args.layer,
        "units": args.units,
        "mode": args.mode,
        "topk": args.topk,
        "device": args.device,
        "use_sae_reconstruction": bool(args.use_sae_reconstruction),
        "prompt_vector": {
            "mode": args.prompt_vector_mode,
            "source_layer": args.prompt_vector_source_layer,
            "token_span": args.prompt_vector_token_span,
            "last_n": args.prompt_vector_last_n,
            "inject_at_layer": args.inject_at_layer,
            "inject_at_token_index": args.inject_at_token_index,
            "inject_alpha": args.inject_alpha,
            "inject_mode": args.inject_mode,
        },
        "script": str(FIELD_VIEW_SCRIPT.relative_to(ROOT)),
        "script_sha256": sha256(FIELD_VIEW_SCRIPT),
        "sae_weights": safe_path_label(sae_state_path),
        "sae_weights_sha256": sha256(sae_state_path) if sae_state_path.exists() else None,
        "git": git_info(ROOT),
        "metrics": {
            "logit_entropy": data.get("logit_entropy"),
            "gap_state_to_candidates": data.get("gap_state_to_candidates"),
            "operator_strength": operator_strength,
            "risk_score": data.get("risk_score"),
            "field_coords": data.get("field_coords"),
            "candidate_spread_mean": data.get("candidate_spread_mean"),
            "prompt_vector_norm": (data.get("prompt_vector_meta") or {}).get("prompt_vector_norm"),
            "injection_delta_norm": (data.get("injection") or {}).get("delta_norm"),
        },
        "candidate_front": {
            "coherence": candidate_front.get("candidate_coherence"),
            "variance": candidate_front.get("candidate_variance"),
            "centroid": candidate_front.get("candidate_centroid"),
            "state_to_centroid_distance": candidate_front.get("state_to_centroid_distance"),
            "degeneracy_ratio_topk": candidate_front.get("degeneracy_ratio_topk"),
            "degenerate": candidate_front.get("degenerate"),
            "generic_tokens": candidate_front.get("generic_tokens"),
            "generic_count": candidate_front.get("generic_count"),
            "candidate_count": candidate_front.get("candidate_count"),
        },
        "artifact_paths": {
            "field_view_json": str(dest_json.relative_to(ROOT)),
            "candidate_front_metrics_json": str(candidate_metrics_path.relative_to(ROOT)),
        },
        "checksums": {
            "field_view_json_sha256": sha256(dest_json),
            "candidate_front_metrics_json_sha256": sha256(candidate_metrics_path),
        },
    }

    # Skriv run JSON
    run_json_path = RUNS_DIR / f"{ts}.json"
    run_json_path.write_text(json.dumps(run_record, indent=2))

    # Skriv kort MD
    def fmt(value):
        if value is None:
            return "nan"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "nan"

    md_lines = [
        f"# Field View run — {ts}",
        "",
        f"- Scenario: `{args.scenario}`",
        f"- Prompt: {args.prompt}",
        f"- Model: {args.model}, layer={args.layer}",
        f"- Units: {args.units}, mode={args.mode}, topk={args.topk}, device={args.device}",
        f"- Risk: {fmt(data.get('risk_score'))}, H={fmt(data.get('logit_entropy'))}, "
        f"|coords|={fmt(operator_strength)}, gap={fmt(data.get('gap_state_to_candidates'))}",
        f"- Candidate front: coherence={fmt(candidate_front.get('candidate_coherence'))}, "
        f"variance={fmt(candidate_front.get('candidate_variance'))}, "
        f"degeneracy={fmt(candidate_front.get('degeneracy_ratio_topk'))}, "
        f"degenerate={str(bool(candidate_front.get('degenerate'))).lower()}",
        "",
        "Artifacts:",
        f"- JSON: {dest_json.relative_to(ROOT)}",
        f"- Candidate metrics JSON: {candidate_metrics_path.relative_to(ROOT)}",
    ]
    run_md_path = RUNS_DIR / f"{ts}.md"
    run_md_path.write_text("\n".join(md_lines))

    print("Klart.")
    print("Run JSON:", run_json_path.relative_to(ROOT))
    print("Run MD:  ", run_md_path.relative_to(ROOT))
    print("Artifact:", dest_json.relative_to(ROOT))
    print("Candidate metrics:", candidate_metrics_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
