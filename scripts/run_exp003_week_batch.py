#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COMPARE = ROOT / "scripts" / "compare_compression_vectorized.py"
OUT_DIR = ROOT / "experiments" / "exp_003_compression_vectorized" / "week_of_2026-03-09"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-jsonl", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--units", nargs="+", type=int, default=[472, 468, 57, 156, 346])
    ap.add_argument("--mode", choices=["mean", "pc1", "pc2"], default="pc2")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--vector-topk", type=int, default=8)
    ap.add_argument("--vector-methods", nargs="+", choices=["mean", "attn_weighted", "pca1"], default=["mean", "attn_weighted", "pca1"])
    ap.add_argument("--sae-state", default=str(ROOT / "experiments" / "exp_001_sae_v3" / "sae_weights.pt"))
    ap.add_argument("--use-sae-reconstruction", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--require-compressor", action="store_true")
    ap.add_argument("--exclude-invalid-compression", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    prompt_path = Path(args.prompt_jsonl)
    entries = [json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise SystemExit("No prompts found in JSONL.")

    prompt_map = {entry["prompt"]: {"id": entry["id"], "stratum": entry["stratum"]} for entry in entries}
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry["prompt"] + "\n")
        temp_prompt_file = Path(handle.name)

    before = {p.name for p in (ROOT / "experiments" / "exp_003_compression_vectorized").glob("results_*.json")}
    cmd = [
        sys.executable,
        str(COMPARE),
        "--prompts-file",
        str(temp_prompt_file),
        "--model",
        args.model,
        "--layer",
        str(args.layer),
        "--units",
        *[str(u) for u in args.units],
        "--mode",
        args.mode,
        "--topk",
        str(args.topk),
        "--vector-topk",
        str(args.vector_topk),
        "--vector-methods",
        *args.vector_methods,
        "--sae-state",
        args.sae_state,
        "--device",
        args.device,
    ]
    if args.use_sae_reconstruction:
        cmd.append("--use-sae-reconstruction")
    if args.require_compressor:
        cmd.append("--require-compressor")
    if args.exclude_invalid_compression:
        cmd.append("--exclude-invalid-compression")
    subprocess.run(cmd, cwd=ROOT, check=True)

    out_base = ROOT / "experiments" / "exp_003_compression_vectorized"
    created = sorted(set(p.name for p in out_base.glob("results_*.json")) - before)
    if not created:
        raise SystemExit("No compare results were created.")
    latest_json = out_base / created[-1]
    latest_csv = latest_json.with_suffix(".csv")
    rows = json.loads(latest_json.read_text(encoding="utf-8"))
    for row in rows:
        meta = prompt_map.get(row["raw_prompt"], {})
        row["prompt_source_id"] = meta.get("id", row["prompt_id"])
        row["prompt_stratum"] = meta.get("stratum", "unknown")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    annotated_json = OUT_DIR / f"annotated_results_{stamp}.json"
    annotated_csv = OUT_DIR / f"annotated_results_{stamp}.csv"
    annotated_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys() if key != "meta"})
    with annotated_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean_row = {key: value for key, value in row.items() if key in fieldnames}
            writer.writerow(clean_row)

    print(f"Source compare JSON: {latest_json}")
    print(f"Annotated JSON: {annotated_json}")
    print(f"Annotated CSV: {annotated_csv}")


if __name__ == "__main__":
    main()
