from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_trace(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def heatmap_metric(df: pd.DataFrame, metric: str, out: Path, title: str) -> None:
    piv = df.pivot_table(index="layer", columns="token_index", values=metric, aggfunc="mean")
    plt.figure(figsize=(10, 4))
    plt.imshow(piv.values, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label=metric)
    plt.yticks(range(len(piv.index)), piv.index)
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Layer")
    plt.xlabel("Token index")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def pca_scatter(df: pd.DataFrame, out: Path, title: str) -> None:
    if "pca_x" not in df.columns or "pca_y" not in df.columns:
        return
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(df["pca_x"], df["pca_y"], c=df["token_index"], cmap="viridis", s=30, alpha=0.8)
    plt.colorbar(sc, label="token index")
    plt.title(title)
    plt.xlabel("pca_x")
    plt.ylabel("pca_y")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def run_viz(trace_path: Path, out_dir: Path) -> None:
    df = load_trace(trace_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for prompt_id, dfp in df.groupby("prompt_id"):
        slug = str(prompt_id) if prompt_id is not None else "unknown"
        heatmap_metric(dfp, "entropy", out_dir / f"{slug}_entropy.png", f"{slug} — entropy")
        heatmap_metric(dfp, "gap_top2", out_dir / f"{slug}_gap.png", f"{slug} — gap_top2")
        pca_scatter(dfp, out_dir / f"{slug}_pca.png", f"{slug} — PCA")

    heatmap_metric(df, "entropy", out_dir / "all_entropy.png", "All prompts — entropy")
    heatmap_metric(df, "gap_top2", out_dir / "all_gap.png", "All prompts — gap_top2")
    if "pca_x" in df.columns and "pca_y" in df.columns:
        pca_scatter(df, out_dir / "all_pca.png", "All prompts — PCA")

    # CSV summaries mean/var per prompt & layer
    summary = []
    for (pid, layer), dfg in df.groupby(["prompt_id", "layer"]):
        summary.append(
            {
                "prompt_id": pid,
                "layer": layer,
                "entropy_mean": dfg["entropy"].mean(),
                "entropy_var": dfg["entropy"].var(ddof=0),
                "gap_mean": dfg["gap_top2"].mean(),
                "gap_var": dfg["gap_top2"].var(ddof=0),
                "attn_entropy_mean": dfg["attn_entropy"].mean(),
            }
        )
    pd.DataFrame(summary).to_csv(out_dir / "summary_by_prompt_layer.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    run_viz(Path(args.trace), Path(args.out_dir))
    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
