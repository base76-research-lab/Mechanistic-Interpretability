#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "experiments" / "exp_003_compression_vectorized" / "results_20260305T135202Z.json"
DEFAULT_OUT = ROOT / "reports" / "figures" / "vector_mode_degeneracy_vs_coherence.png"

COLOR_MAP = {
    "raw": "#1f77b4",
    "compressed": "#2ca02c",
    "mean": "#d62728",
    "attn_weighted": "#ff7f0e",
    "pca1": "#9467bd",
}


def infer_mode(variant: str) -> str:
    if variant == "raw":
        return "raw"
    if variant == "compressed":
        return "compressed"
    if "_attn_weighted" in variant:
        return "attn_weighted"
    if "_pca1" in variant:
        return "pca1"
    if "_mean" in variant or "vectorized_proxy" in variant:
        return "mean"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default=str(DEFAULT_RESULTS))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    src = Path(args.results)
    out = Path(args.out)
    rows = json.loads(src.read_text())

    points = []
    for r in rows:
        deg = r.get("degeneracy_ratio_topk")
        coh = r.get("candidate_coherence")
        if deg is None or coh is None:
            continue
        variant = str(r.get("variant", ""))
        mode = infer_mode(variant)
        points.append(
            {
                "prompt_id": str(r.get("prompt_id", "p")),
                "variant": variant,
                "mode": mode,
                "x": float(deg),
                "y": float(coh),
            }
        )

    if not points:
        raise SystemExit("No points to plot.")

    plt.figure(figsize=(8.2, 5.4), dpi=170)

    # Visual zones
    plt.axvspan(0.7, 1.0, ymin=0.0, ymax=0.3, alpha=0.10, color="red")
    plt.axvspan(0.0, 0.3, ymin=0.7, ymax=1.0, alpha=0.08, color="green")

    for mode in sorted({p["mode"] for p in points}):
        mode_points = [p for p in points if p["mode"] == mode]
        xs = [p["x"] for p in mode_points]
        ys = [p["y"] for p in mode_points]
        color = COLOR_MAP.get(mode, "#7f7f7f")
        plt.scatter(xs, ys, label=mode, s=95, c=color, edgecolors="black", linewidths=0.45)

    for p in points:
        lbl = f"{p['prompt_id']}:{p['mode']}"
        plt.annotate(lbl, (p["x"], p["y"]), textcoords="offset points", xytext=(5, 6), fontsize=7)

    plt.title("Degeneracy vs Coherence by Vector Mode")
    plt.xlabel("degeneracy_ratio_topk")
    plt.ylabel("candidate_coherence")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-1.02, 1.02)
    plt.grid(alpha=0.26)
    plt.legend(title="vector_mode", loc="lower left", fontsize=8)
    plt.text(0.71, -0.93, "Hallucination zone", color="darkred", fontsize=8)
    plt.text(0.02, 0.90, "Reasoning zone", color="darkgreen", fontsize=8)
    plt.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
