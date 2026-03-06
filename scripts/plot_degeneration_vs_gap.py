#!/usr/bin/env python3
"""
plot_degeneration_vs_gap.py

Create diagnostic scatter plots:
- x: gap_state_to_candidates
- y1: degeneracy_ratio_topk
- y2: candidate_coherence
- color: risk_score

Input sources:
1) run logs: experiments/exp_001_sae_v3/runs/*.json
2) candidate metrics JSON list: experiments/exp_001_sae_v3/candidate_front_metrics.json

Usage:
  python3 scripts/plot_degeneration_vs_gap.py

  python3 scripts/plot_degeneration_vs_gap.py \
    --runs-glob "experiments/exp_001_sae_v3/runs/*.json" \
    --out-degeneration reports/figures/degeneration_vs_gap.png \
    --out-coherence reports/figures/coherence_vs_gap.png
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_GLOB = str(ROOT / "experiments" / "exp_001_sae_v3" / "runs" / "*.json")
DEFAULT_METRICS_JSON = ROOT / "experiments" / "exp_001_sae_v3" / "candidate_front_metrics.json"
DEFAULT_OUT_DEGEN = ROOT / "reports" / "figures" / "degeneration_vs_gap.png"
DEFAULT_OUT_COH = ROOT / "reports" / "figures" / "coherence_vs_gap.png"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def collect_from_runs(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in sorted(glob.glob(pattern)):
        path = Path(match)
        data = load_json(path)
        metrics = data.get("metrics", {})
        front = data.get("candidate_front", {})
        scenario = str(data.get("scenario", path.stem))

        gap = metrics.get("gap_state_to_candidates")
        risk = metrics.get("risk_score")
        deg = front.get("degeneracy_ratio_topk")
        if gap is None or deg is None:
            continue

        rows.append(
            {
                "label": scenario,
                "gap": float(gap),
                "degeneracy": float(deg),
                "coherence": float(front.get("coherence")) if front.get("coherence") is not None else None,
                "risk": float(risk) if risk is not None else None,
            }
        )
    return rows


def collect_from_metrics_json(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    data = load_json(path)
    if not isinstance(data, list):
        return rows
    for item in data:
        file_label = Path(str(item.get("file", "unknown"))).stem
        gap = item.get("gap_state_to_candidates")
        deg = item.get("degeneracy_ratio_topk")
        risk = item.get("risk_score")
        if gap is None or deg is None:
            continue
        rows.append(
            {
                "label": file_label,
                "gap": float(gap),
                "degeneracy": float(deg),
                "coherence": float(item.get("candidate_coherence"))
                if item.get("candidate_coherence") is not None
                else None,
                "risk": float(risk) if risk is not None else None,
            }
        )
    return rows


def plot_degeneration(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to plot.")

    xs = [row["gap"] for row in rows]
    ys = [row["degeneracy"] for row in rows]
    risks = [row["risk"] if row["risk"] is not None else 0.0 for row in rows]

    plt.figure(figsize=(7.5, 5.0), dpi=160)
    scatter = plt.scatter(xs, ys, c=risks, cmap="viridis", s=85, edgecolors="black", linewidths=0.4)
    plt.colorbar(scatter, label="risk_score")

    for row in rows:
        plt.annotate(
            row["label"],
            (row["gap"], row["degeneracy"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    plt.title("Candidate Frontier: Degeneracy vs Gap")
    plt.xlabel("gap_state_to_candidates")
    plt.ylabel("degeneracy_ratio_topk")
    plt.grid(alpha=0.25)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_coherence(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    filtered = [row for row in rows if row.get("coherence") is not None]
    if not filtered:
        raise ValueError("No rows with coherence to plot.")

    xs = [row["gap"] for row in filtered]
    ys = [row["coherence"] for row in filtered]
    risks = [row["risk"] if row["risk"] is not None else 0.0 for row in filtered]

    plt.figure(figsize=(7.5, 5.0), dpi=160)
    scatter = plt.scatter(xs, ys, c=risks, cmap="viridis", s=85, edgecolors="black", linewidths=0.4)
    plt.colorbar(scatter, label="risk_score")

    for row in filtered:
        plt.annotate(
            row["label"],
            (row["gap"], row["coherence"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    plt.title("Candidate Frontier: Coherence vs Gap")
    plt.xlabel("gap_state_to_candidates")
    plt.ylabel("candidate_coherence")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-glob", type=str, default=DEFAULT_RUNS_GLOB, help="Glob for run JSON files")
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=str(DEFAULT_METRICS_JSON),
        help="Fallback list JSON from candidate_front_metrics.py",
    )
    parser.add_argument("--out-degeneration", type=str, default=str(DEFAULT_OUT_DEGEN), help="Output PNG path for degeneration plot")
    parser.add_argument("--out-coherence", type=str, default=str(DEFAULT_OUT_COH), help="Output PNG path for coherence plot")
    args = parser.parse_args()

    rows = collect_from_runs(args.runs_glob)
    # If run logs are sparse (e.g. only one post-integration run), prefer the batch metrics list.
    if len(rows) < 2:
        rows = collect_from_metrics_json(Path(args.metrics_json))

    plot_degeneration(rows, Path(args.out_degeneration))
    plot_coherence(rows, Path(args.out_coherence))
    print(f"Saved: {args.out_degeneration}")
    print(f"Saved: {args.out_coherence}")
    print(f"Points: {len(rows)}")


if __name__ == "__main__":
    main()
