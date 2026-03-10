#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PANEL = ROOT / "data" / "prompts_observability_panel_2026-03-07.jsonl"
DEFAULT_SAE = ROOT / "experiments" / "exp_001_sae_v3" / "sae_weights.pt"
DEFAULT_OUT_ROOT = ROOT / "experiments" / "exp_005_trajectory_block"
DEFAULT_LAYERS = [3, 5, 6, 9, 12]
DEFAULT_UNITS = [472, 468, 57, 156, 346]


def run_cmd(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-jsonl", default=str(DEFAULT_PANEL))
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--sae-state", default=str(DEFAULT_SAE))
    ap.add_argument("--units", nargs="+", type=int, default=DEFAULT_UNITS)
    ap.add_argument("--basis-mode", choices=["mean", "pc1", "pc2"], default="pc2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--run-name", default="")
    ap.add_argument("--baseline-trace", default="")
    ap.add_argument("--recon-trace", default="")
    args = ap.parse_args()

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"trajectory_block_{args.model}_{ts}"
    block_dir = Path(args.out_root) / run_name
    block_dir.mkdir(parents=True, exist_ok=True)

    readonly_name = f"{run_name}_readonly"
    baseline_name = f"{run_name}_baseline"
    recon_name = f"{run_name}_writeback"

    readonly_cmd = [
        "python3",
        "-m",
        "transformer_oscilloscope.cli",
        "trace",
        "--prompt-jsonl",
        args.panel_jsonl,
        "--model",
        args.model,
        "--layers",
        *[str(x) for x in args.layers],
        "--device",
        args.device,
        "--out-dir",
        str(block_dir),
        "--run-name",
        readonly_name,
        "--store-projections",
        "--sae-state",
        args.sae_state,
        "--sae-topk",
        "8",
        "--basis-mode",
        args.basis_mode,
        "--units",
        *[str(x) for x in args.units],
    ]
    run_cmd(readonly_cmd)

    baseline_trace = args.baseline_trace
    if not baseline_trace:
        baseline_cmd = [
            "python3",
            "scripts/run_unified_observability_stack.py",
            "--prompt-jsonl",
            args.panel_jsonl,
            "--model",
            args.model,
            "--layers",
            *[str(x) for x in args.layers],
            "--sae-state",
            args.sae_state,
            "--basis-mode",
            args.basis_mode,
            "--units",
            *[str(x) for x in args.units],
            "--intervention-state",
            "unified_baseline",
            "--run-name",
            baseline_name,
            "--device",
            args.device,
        ]
        run_cmd(baseline_cmd)
        baseline_trace = str(ROOT / "experiments" / "exp_004_unified_observability_stack" / baseline_name / "trace.jsonl")

    recon_trace = args.recon_trace
    if not recon_trace:
        recon_cmd = [
            "python3",
            "scripts/run_unified_observability_stack.py",
            "--prompt-jsonl",
            args.panel_jsonl,
            "--model",
            args.model,
            "--layers",
            *[str(x) for x in args.layers],
            "--sae-state",
            args.sae_state,
            "--basis-mode",
            args.basis_mode,
            "--units",
            *[str(x) for x in args.units],
            "--intervention-state",
            "writeback_intervention",
            "--use-sae-reconstruction",
            "--run-name",
            recon_name,
            "--device",
            args.device,
        ]
        run_cmd(recon_cmd)
        recon_trace = str(ROOT / "experiments" / "exp_004_unified_observability_stack" / recon_name / "trace.jsonl")

    readonly_trace = block_dir / readonly_name / "trace.jsonl"
    analyze_cmd = [
        "python3",
        "scripts/analyze_trajectory_block.py",
        "--panel-jsonl",
        args.panel_jsonl,
        "--readonly-trace",
        str(readonly_trace),
        "--baseline-trace",
        baseline_trace,
        "--recon-trace",
        recon_trace,
        "--out-dir",
        str(block_dir / "analysis"),
    ]
    run_cmd(analyze_cmd)

    manifest = {
        "run_name": run_name,
        "panel_jsonl": args.panel_jsonl,
        "model": args.model,
        "layers": args.layers,
        "sae_state": args.sae_state,
        "units": args.units,
        "basis_mode": args.basis_mode,
        "device": args.device,
        "readonly_trace": str(readonly_trace),
        "baseline_trace": baseline_trace,
        "recon_trace": recon_trace,
        "analysis_dir": str(block_dir / "analysis"),
    }
    (block_dir / "block_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved trajectory block manifest to {block_dir / 'block_manifest.json'}")


if __name__ == "__main__":
    main()
