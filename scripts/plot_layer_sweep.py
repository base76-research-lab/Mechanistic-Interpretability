#!/usr/bin/env python3
"""
plot_layer_sweep.py

Reads a set of field_view JSON files (from run_field_view_logged.py) and plots
H, gap, risk, and operator strength vs. layer.

Usage:
  python3 scripts/plot_layer_sweep.py \\
    --out experiments/exp_001_sae_phi2/layer_sweep.png \\
    math:8:artifacts/.../field_view_math_L8.json \\
    math:16:artifacts/.../field_view_math_L16.json \\
    math:24:artifacts/.../field_view_math_L24.json

Format per argument:  <label>:<layer>:<path>
  label  = scenario (e.g. math, analogy, halluc)
  layer  = integer (layer index)
  path   = path to a field_view JSON file

Plots one panel per label and also prints a summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


@dataclass
class Record:
    label: str
    layer: int
    path: Path
    entropy: float
    gap: float
    risk: float
    strength: float  # |coords|


def parse_arg(arg: str) -> tuple[str, int, Path]:
    """
    Expect format label:layer:path. Fallback: try to extract layer from filename (L(\d+)).
    """
    if ":" not in arg:
        raise ValueError(f"Invalid argument '{arg}'. Expected label:layer:path")
    parts = arg.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid argument '{arg}'. Expected label:layer:path")
    label, layer_str, path_str = parts
    layer: int
    if layer_str.isdigit():
        layer = int(layer_str)
    else:
        m = re.search(r"[Ll](\d+)", layer_str)
        if not m:
            raise ValueError(f"Could not parse layer from '{layer_str}'")
        layer = int(m.group(1))
    return label, layer, Path(path_str)


def load_record(label: str, layer: int, path: Path) -> Record:
    data = json.loads(path.read_text())
    return Record(
        label=label,
        layer=layer,
        path=path,
        entropy=float(data["logit_entropy"]),
        gap=float(data["gap_state_to_candidates"]),
        risk=float(data["risk_score"]),
        strength=float(data.get("operator_strength", 0.0)),
    )


def plot(records: List[Record], out: Path):
    labels = sorted({r.label for r in records})
    fig, axes = plt.subplots(len(labels), 1, figsize=(6, 3 * len(labels)), sharex=True)
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        rs = sorted([r for r in records if r.label == label], key=lambda r: r.layer)
        layers = [r.layer for r in rs]
        ax.plot(layers, [r.entropy for r in rs], "o-", label="H (entropy)")
        ax.plot(layers, [r.gap for r in rs], "s--", label="gap")
        ax.plot(layers, [r.risk for r in rs], "d-.", label="risk")
        ax.plot(layers, [r.strength for r in rs], "x:", label="|coords|")
        ax.set_title(f"{label}")
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Plot saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="label:layer:path to JSON (>=2 layers per label recommended)")
    parser.add_argument("--out", type=Path, default=Path("layer_sweep.png"))
    args = parser.parse_args()

    recs: List[Record] = []
    for a in args.inputs:
        label, layer, path = parse_arg(a)
        if not path.exists():
            raise FileNotFoundError(path)
        recs.append(load_record(label, layer, path))

    # Summarize to stdout
    recs_sorted = sorted(recs, key=lambda r: (r.label, r.layer))
    print("label\tlayer\tH\tgap\trisk\t|coords|\tpath")
    for r in recs_sorted:
        print(f"{r.label}\t{r.layer}\t{r.entropy:.3f}\t{r.gap:.3f}\t{r.risk:.3f}\t{r.strength:.3f}\t{r.path}")

    plot(recs, args.out)


if __name__ == "__main__":
    main()
