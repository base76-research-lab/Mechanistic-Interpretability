#!/usr/bin/env python3
"""
plot_unified_stack_traces.py

Create a simple multi-panel visualization layer for unified stack traces:
- operator strength by layer
- lens entropy by layer
- frontier gap by layer
- feature drift by layer
- decision trajectory smoothness by prompt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "anchored": "#22c55e",
    "reasoning": "#38bdf8",
    "transition": "#f59e0b",
    "hallucination_prone": "#ef4444",
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def plot_metric(ax, rows: list[dict], metric: str, title: str, ylabel: str) -> None:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["prompt_id"], []).append(row)
    for prompt_id, prompt_rows in grouped.items():
        prompt_rows = sorted(prompt_rows, key=lambda r: r["layer"])
        regime = prompt_rows[0]["regime"]
        color = COLORS.get(regime, "#94a3b8")
        ax.plot(
            [r["layer"] for r in prompt_rows],
            [r.get(metric) for r in prompt_rows],
            marker="o",
            color=color,
            alpha=0.8,
            linewidth=1.8,
        )
        ax.text(prompt_rows[-1]["layer"], prompt_rows[-1].get(metric), prompt_id, color=color, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle="--")


def plot_trace_metric(ax, rows: list[dict], metric: str, title: str, ylabel: str) -> None:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["prompt_id"], []).append(row)
    prompt_ids = []
    values = []
    colors = []
    for prompt_id, prompt_rows in grouped.items():
        prompt_rows = sorted(prompt_rows, key=lambda r: r["layer"])
        prompt_ids.append(prompt_id)
        values.append(prompt_rows[-1].get(metric))
        colors.append(COLORS.get(prompt_rows[0]["regime"], "#94a3b8"))
    xs = np.arange(len(prompt_ids))
    ax.bar(xs, values, color=colors, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(prompt_ids, rotation=35, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle="--", axis="y")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.trace_jsonl))
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    plot_metric(axes[0, 0], rows, "subspace_operator_strength", "Operator Strength by Layer", "Operator strength")
    plot_metric(axes[0, 1], rows, "lens_entropy", "Lens Entropy by Layer", "Entropy")
    plot_metric(axes[1, 0], rows, "gap_state_to_candidates", "Frontier Gap by Layer", "Gap to candidates")
    plot_metric(axes[1, 1], rows, "feature_drift_vs_prev_layer", "Feature Drift by Layer", "Feature drift")
    plot_metric(
        axes[2, 0],
        rows,
        "lens_entropy_delta_vs_prev_layer",
        "Lens Entropy Delta by Layer",
        "Entropy delta",
    )
    plot_trace_metric(
        axes[2, 1],
        rows,
        "decision_trajectory_smoothness",
        "Decision Trajectory Smoothness by Prompt",
        "DTS",
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
