#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_trace(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def add_subspace_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "subspace_coords" not in out.columns:
        raise ValueError("trace is missing subspace_coords")
    out["subspace_x"] = out["subspace_coords"].apply(lambda x: float(x[0]))
    out["subspace_y"] = out["subspace_coords"].apply(lambda x: float(x[1]) if len(x) > 1 else 0.0)
    return out


def select_last_token(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("prompt_id")["token_index"].transform("max") == df["token_index"]
    return df[idx].copy()


def path_length(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    if len(df) < 2:
        return 0.0
    dx = df[x_col].diff()
    dy = df[y_col].diff()
    step = np.sqrt(dx.pow(2) + dy.pow(2))
    return float(step.fillna(0.0).sum())


def compare_traces(
    baseline_path: Path,
    observer_path: Path,
    out_dir: Path,
    baseline_label: str,
    observer_label: str,
) -> None:
    baseline = add_subspace_columns(load_trace(baseline_path))
    observer = add_subspace_columns(load_trace(observer_path))

    baseline = select_last_token(baseline)
    observer = select_last_token(observer)

    keep = [
        "prompt_id",
        "layer",
        "token_index",
        "subspace_x",
        "subspace_y",
        "subspace_operator_strength",
    ]
    merged = baseline[keep].merge(
        observer[keep],
        on=["prompt_id", "layer", "token_index"],
        suffixes=("_baseline", "_observer"),
        how="inner",
    )
    merged["point_distance"] = np.sqrt(
        (merged["subspace_x_observer"] - merged["subspace_x_baseline"]) ** 2
        + (merged["subspace_y_observer"] - merged["subspace_y_baseline"]) ** 2
    )

    per_prompt = []
    for prompt_id, group in merged.groupby("prompt_id"):
        group = group.sort_values("layer")
        baseline_path_len = path_length(group, "subspace_x_baseline", "subspace_y_baseline")
        observer_path_len = path_length(group, "subspace_x_observer", "subspace_y_observer")
        per_prompt.append(
            {
                "prompt_id": prompt_id,
                "layer_count": int(group["layer"].nunique()),
                "mean_point_distance": float(group["point_distance"].mean()),
                "max_point_distance": float(group["point_distance"].max()),
                "baseline_path_length": baseline_path_len,
                "observer_path_length": observer_path_len,
                "path_length_delta": observer_path_len - baseline_path_len,
            }
        )
    per_prompt_df = pd.DataFrame(per_prompt).sort_values("mean_point_distance", ascending=False)

    summary = {
        "baseline_trace": str(baseline_path),
        "observer_trace": str(observer_path),
        "baseline_label": baseline_label,
        "observer_label": observer_label,
        "shared_point_count": int(len(merged)),
        "prompt_count": int(merged["prompt_id"].nunique()),
        "layer_count": int(merged["layer"].nunique()),
        "mean_point_distance": float(merged["point_distance"].mean()),
        "max_point_distance": float(merged["point_distance"].max()),
        "mean_baseline_operator_strength": float(merged["subspace_operator_strength_baseline"].mean()),
        "mean_observer_operator_strength": float(merged["subspace_operator_strength_observer"].mean()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "merged_points.csv", index=False)
    per_prompt_df.to_csv(out_dir / "per_prompt_summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    prompts = sorted(merged["prompt_id"].unique())
    cols = 3
    rows = int(np.ceil(len(prompts) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    for ax, prompt_id in zip(axes.flatten(), prompts):
        group = merged[merged["prompt_id"] == prompt_id].sort_values("layer")
        ax.plot(group["subspace_x_baseline"], group["subspace_y_baseline"], marker="o", label=baseline_label)
        ax.plot(group["subspace_x_observer"], group["subspace_y_observer"], marker="o", label=observer_label)
        for _, row in group.iterrows():
            ax.text(row["subspace_x_baseline"], row["subspace_y_baseline"], str(int(row["layer"])), fontsize=7)
        ax.set_title(prompt_id)
        ax.set_xlabel("subspace_x")
        ax.set_ylabel("subspace_y")
        ax.grid(alpha=0.25)
    for ax in axes.flatten()[len(prompts):]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Trajectory comparison: {baseline_label} vs {observer_label}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "trajectory_overlay.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-trace", required=True)
    ap.add_argument("--observer-trace", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--observer-label", default="observer")
    args = ap.parse_args()

    compare_traces(
        baseline_path=Path(args.baseline_trace),
        observer_path=Path(args.observer_trace),
        out_dir=Path(args.out_dir),
        baseline_label=args.baseline_label,
        observer_label=args.observer_label,
    )


if __name__ == "__main__":
    main()
