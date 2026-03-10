#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_panel(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def load_trace(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def prepare_readonly(df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["prompt_id", "layer", "token_index"])
    last_idx = out.groupby(["prompt_id", "layer"])["token_index"].transform("max") == out["token_index"]
    out = out[last_idx].copy()
    out["intervention_state"] = "readonly_observer"
    if "entropy" in out.columns and "lens_entropy" not in out.columns:
        out["lens_entropy"] = out["entropy"]
    if "frontier_coherence" not in out.columns:
        out["frontier_coherence"] = np.nan
    if "frontier_degeneracy" not in out.columns:
        out["frontier_degeneracy"] = np.nan
    if "gap_state_to_candidates" not in out.columns:
        out["gap_state_to_candidates"] = np.nan
    if "candidate_variance" not in out.columns:
        out["candidate_variance"] = np.nan
    out = out.merge(panel_df[["id", "prompt", "stratum"]], left_on="prompt_id", right_on="id", how="left")
    out["prompt"] = out["prompt_x"].fillna(out["prompt_y"]) if "prompt_x" in out.columns else out["prompt"]
    if "prompt_x" in out.columns:
        out = out.drop(columns=["id", "prompt_x", "prompt_y"])
    else:
        out = out.drop(columns=["id"])
    return out


def prepare_unified(df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["prompt_id", "layer", "token_index"])
    if "prompt" not in out.columns or "stratum" not in out.columns:
        out = out.merge(panel_df[["id", "prompt", "stratum"]], left_on="prompt_id", right_on="id", how="left")
        out = out.drop(columns=["id"])
    return out


def add_geometry_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subspace_x"] = out["subspace_coords"].apply(lambda v: float(v[0]))
    out["subspace_y"] = out["subspace_coords"].apply(lambda v: float(v[1]) if len(v) > 1 else 0.0)
    return out


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def max_step_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.linalg.norm(diffs, axis=1).max())


def trajectory_curvature(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            continue
        cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        total += float(np.arccos(cos))
    return total


def prompt_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (prompt_id, intervention_state), group in df.groupby(["prompt_id", "intervention_state"]):
        group = group.sort_values("layer")
        points = group[["subspace_x", "subspace_y"]].to_numpy(dtype=float)
        layers = group["layer"].to_numpy(dtype=float)
        if len(points) >= 2:
            step_vectors = np.diff(points, axis=0)
            step_lengths = np.linalg.norm(step_vectors, axis=1)
            layer_deltas = np.diff(layers)
            phase_velocity = step_lengths / np.where(layer_deltas == 0.0, 1.0, layer_deltas)
            mean_step_distance = float(step_lengths.mean())
            std_step_distance = float(step_lengths.std(ddof=0))
            mean_phase_velocity = float(phase_velocity.mean())
            max_phase_velocity = float(phase_velocity.max())
        else:
            mean_step_distance = 0.0
            std_step_distance = 0.0
            mean_phase_velocity = 0.0
            max_phase_velocity = 0.0
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": group["prompt"].iloc[0],
                "regime": group["regime"].iloc[0],
                "stratum": group["stratum"].iloc[0] if "stratum" in group.columns else None,
                "intervention_state": intervention_state,
                "layer_count": int(group["layer"].nunique()),
                "path_length": path_length(points),
                "max_step_distance": max_step_distance(points),
                "mean_step_distance": mean_step_distance,
                "std_step_distance": std_step_distance,
                "mean_phase_velocity": mean_phase_velocity,
                "max_phase_velocity": max_phase_velocity,
                "trajectory_curvature": trajectory_curvature(points),
                "endpoint_x": float(points[-1, 0]),
                "endpoint_y": float(points[-1, 1]),
                "mean_entropy": float(group["lens_entropy"].mean()),
                "max_entropy": float(group["lens_entropy"].max()),
                "mean_gap": float(group["gap_state_to_candidates"].mean()),
                "max_gap": float(group["gap_state_to_candidates"].max()),
                "mean_coherence": float(group["frontier_coherence"].mean()),
                "mean_degeneracy": float(group["frontier_degeneracy"].mean()),
                "mean_operator_strength": float(group["subspace_operator_strength"].mean()),
                "decision_trajectory_smoothness": float(group.get("decision_trajectory_smoothness", pd.Series([np.nan])).iloc[-1])
                if "decision_trajectory_smoothness" in group.columns
                else float(group["lens_entropy"].diff().abs().fillna(0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def binary_auc(scores: pd.Series, labels: pd.Series) -> float:
    labels = labels.astype(int)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = scores.rank(method="average")
    sum_ranks_pos = float(ranks[labels == 1].sum())
    return (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def regime_summary_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for regime, group in df.groupby("regime"):
        row: dict[str, Any] = {"regime": regime, "prompt_count": int(len(group))}
        for metric in metrics:
            row[f"{metric}_median"] = float(group[metric].median())
            q1 = float(group[metric].quantile(0.25))
            q3 = float(group[metric].quantile(0.75))
            row[f"{metric}_iqr"] = q3 - q1
        rows.append(row)
    return pd.DataFrame(rows).sort_values("regime")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_detection(prompt_df: pd.DataFrame, out_path: Path) -> None:
    order = ["anchored", "reasoning", "transition", "hallucination_prone", "control"]
    data = [prompt_df.loc[prompt_df["regime"] == regime, "geometry_detection_score"].to_numpy() for regime in order if regime in set(prompt_df["regime"])]
    entropy = [prompt_df.loc[prompt_df["regime"] == regime, "mean_entropy"].to_numpy() for regime in order if regime in set(prompt_df["regime"])]
    labels = [regime for regime in order if regime in set(prompt_df["regime"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot(data, tick_labels=labels)
    axes[0].set_title("Geometry Detection Score by Regime")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].boxplot(entropy, tick_labels=labels)
    axes[1].set_title("Entropy Baseline by Regime")
    axes[1].tick_params(axis="x", rotation=25)
    for ax in axes:
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bifurcation(layer_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for comparison, group in layer_df.groupby("comparison"):
        group = group.sort_values("layer")
        axes[0].plot(group["layer"], group["subspace_centroid_distance"], marker="o", label=comparison)
        axes[1].plot(group["layer"], group["composite_divergence"], marker="o", label=comparison)
        axes[2].plot(group["layer"], group["phase_velocity_delta"], marker="o", label=comparison)
    axes[0].set_title("Hallucination vs Regime Centroid Distance")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Distance")
    axes[1].set_title("Composite Divergence by Layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Composite divergence")
    axes[2].set_title("Phase Velocity Delta by Layer")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Velocity delta")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stability(fingerprint_df: pd.DataFrame, out_path: Path) -> None:
    regimes = fingerprint_df["regime"].tolist()
    metrics = [
        "path_length_median",
        "mean_phase_velocity_median",
        "std_step_distance_median",
        "trajectory_curvature_median",
        "endpoint_variance",
        "within_regime_dispersion",
    ]
    matrix = fingerprint_df[metrics].to_numpy(dtype=float).copy()
    for col in range(matrix.shape[1]):
        col_vals = matrix[:, col]
        std = np.std(col_vals)
        if std > 0:
            matrix[:, col] = (col_vals - np.mean(col_vals)) / std
        else:
            matrix[:, col] = 0.0
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels(regimes)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_title("Regime Stability Fingerprints (normalized)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_detection(readonly_prompt_df: pd.DataFrame, baseline_prompt_df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    prompt_df = readonly_prompt_df.copy()
    prompt_df["geometry_detection_score"] = (
        zscore(prompt_df["mean_gap"].fillna(prompt_df["mean_gap"].median()))
        + zscore((-prompt_df["mean_coherence"]).fillna((-prompt_df["mean_coherence"]).median()))
        + zscore(prompt_df["mean_degeneracy"].fillna(prompt_df["mean_degeneracy"].median()))
        + zscore(prompt_df["path_length"])
        + zscore(prompt_df["max_step_distance"])
        + zscore(prompt_df["trajectory_curvature"])
        + zscore(prompt_df["mean_phase_velocity"])
    ) / 7.0
    prompt_df["is_hallucination"] = (prompt_df["regime"] == "hallucination_prone").astype(int)
    pair_aucs = {}
    for other in ["anchored", "reasoning", "transition"]:
        subset = prompt_df[prompt_df["regime"].isin(["hallucination_prone", other])].copy()
        pair_aucs[f"geometry_vs_{other}"] = binary_auc(subset["geometry_detection_score"], subset["is_hallucination"])
        pair_aucs[f"entropy_vs_{other}"] = binary_auc(subset["mean_entropy"], subset["is_hallucination"])
    overall_auc = binary_auc(prompt_df["geometry_detection_score"], prompt_df["is_hallucination"])
    entropy_auc = binary_auc(prompt_df["mean_entropy"], prompt_df["is_hallucination"])
    verdict = "uncertain"
    if overall_auc > entropy_auc + 0.05:
        verdict = "useful"
    elif overall_auc <= entropy_auc:
        verdict = "not_better_than_entropy"

    regime_summary = regime_summary_table(
        prompt_df,
        [
            "geometry_detection_score",
            "mean_entropy",
            "mean_gap",
            "mean_coherence",
            "mean_degeneracy",
            "path_length",
            "trajectory_curvature",
            "mean_phase_velocity",
        ],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_df.to_csv(out_dir / "per_prompt_summary.csv", index=False)
    regime_summary.to_csv(out_dir / "regime_summary.csv", index=False)
    plot_detection(prompt_df, out_dir / "detection_regime_separation.png")
    summary = {
        "overall_geometry_auc": overall_auc,
        "overall_entropy_auc": entropy_auc,
        "pairwise_auc": pair_aucs,
        "verdict": verdict,
        "readonly_prompt_count": int(len(prompt_df)),
        "baseline_prompt_count": int(len(baseline_prompt_df)),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def analyze_bifurcation(readonly_df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    comparisons = []
    step_rows = []
    hallu = readonly_df[readonly_df["regime"] == "hallucination_prone"]
    for other in ["anchored", "reasoning", "transition"]:
        other_df = readonly_df[readonly_df["regime"] == other]
        prev_h_centroid: np.ndarray | None = None
        prev_o_centroid: np.ndarray | None = None
        prev_layer: int | None = None
        for layer in sorted(readonly_df["layer"].unique()):
            h_layer = hallu[hallu["layer"] == layer]
            o_layer = other_df[other_df["layer"] == layer]
            if h_layer.empty or o_layer.empty:
                continue
            h_centroid = h_layer[["subspace_x", "subspace_y"]].mean().to_numpy(dtype=float)
            o_centroid = o_layer[["subspace_x", "subspace_y"]].mean().to_numpy(dtype=float)
            if prev_h_centroid is not None and prev_o_centroid is not None and prev_layer is not None:
                layer_gap = max(1.0, float(layer - prev_layer))
                h_velocity = float(np.linalg.norm(h_centroid - prev_h_centroid) / layer_gap)
                o_velocity = float(np.linalg.norm(o_centroid - prev_o_centroid) / layer_gap)
                phase_velocity_delta = h_velocity - o_velocity
            else:
                phase_velocity_delta = 0.0
            comparisons.append(
                {
                    "comparison": f"hallucination_prone_vs_{other}",
                    "layer": int(layer),
                    "subspace_centroid_distance": float(np.linalg.norm(h_centroid - o_centroid)),
                    "entropy_delta": float(h_layer["lens_entropy"].mean() - o_layer["lens_entropy"].mean()),
                    "gap_delta": float(h_layer["gap_state_to_candidates"].mean() - o_layer["gap_state_to_candidates"].mean()),
                    "coherence_delta": float(h_layer["frontier_coherence"].mean() - o_layer["frontier_coherence"].mean()),
                    "degeneracy_delta": float(h_layer["frontier_degeneracy"].mean() - o_layer["frontier_degeneracy"].mean()),
                    "phase_velocity_delta": phase_velocity_delta,
                }
            )
            prev_h_centroid = h_centroid
            prev_o_centroid = o_centroid
            prev_layer = int(layer)
        for prompt_id, group in readonly_df[readonly_df["regime"].isin(["hallucination_prone", other])].groupby("prompt_id"):
            group = group.sort_values("layer")
            pts = group[["subspace_x", "subspace_y"]].to_numpy(dtype=float)
            layers = group["layer"].to_numpy(dtype=float)
            regime = group["regime"].iloc[0]
            for i in range(len(group) - 1):
                start_layer = int(layers[i])
                end_layer = int(layers[i + 1])
                dist = float(np.linalg.norm(pts[i + 1] - pts[i]))
                velocity = dist / max(1.0, layers[i + 1] - layers[i])
                step_rows.append(
                    {
                        "comparison_family": other,
                        "prompt_id": prompt_id,
                        "regime": regime,
                        "layer_start": start_layer,
                        "layer_end": end_layer,
                        "step_distance": dist,
                        "phase_velocity": velocity,
                    }
                )
    layer_df = pd.DataFrame(comparisons)
    layer_df["composite_divergence"] = (
        zscore(layer_df["subspace_centroid_distance"])
        + zscore(layer_df["gap_delta"].abs())
        + zscore(layer_df["coherence_delta"].abs())
        + zscore(layer_df["degeneracy_delta"].abs())
    ) / 4.0
    layer_scores = layer_df.groupby("layer")["composite_divergence"].mean().reset_index()
    top_layers = layer_scores.sort_values("composite_divergence", ascending=False)["layer"].tolist()
    conclusion = "single-layer candidate"
    if 6 in top_layers[:2] and 9 in top_layers[:3]:
        conclusion = "L6-L9 transition zone"
    elif top_layers and top_layers[0] == 6:
        conclusion = "Layer 6 remains strongest local bifurcation candidate"
    first_stable = int(layer_scores.sort_values("layer").sort_values(["composite_divergence", "layer"], ascending=[False, True]).iloc[0]["layer"])
    step_df = pd.DataFrame(step_rows)
    step_summary = (
        step_df.groupby(["regime", "layer_start", "layer_end"])[["step_distance", "phase_velocity"]]
        .mean()
        .reset_index()
    )
    largest_hallu_step = (
        step_summary[step_summary["regime"] == "hallucination_prone"]
        .sort_values("step_distance", ascending=False)
        .iloc[0]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_df.to_csv(out_dir / "layer_pairwise_summary.csv", index=False)
    layer_scores.to_csv(out_dir / "layer_scores.csv", index=False)
    step_summary.to_csv(out_dir / "step_summary.csv", index=False)
    plot_bifurcation(layer_df, out_dir / "layer_bifurcation.png")
    summary = {
        "conclusion": conclusion,
        "first_strongest_divergence_layer": first_stable,
        "top_layers": top_layers,
        "largest_hallucination_expansion_step": {
            "layer_start": int(largest_hallu_step["layer_start"]),
            "layer_end": int(largest_hallu_step["layer_end"]),
            "step_distance": float(largest_hallu_step["step_distance"]),
            "phase_velocity": float(largest_hallu_step["phase_velocity"]),
        },
        "layer_scores": [
            {"layer": int(row.layer), "composite_divergence": float(row.composite_divergence)}
            for row in layer_scores.itertuples()
        ],
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def analyze_stability(readonly_prompt_df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    rows = []
    for regime, group in readonly_prompt_df.groupby("regime"):
        endpoints = group[["endpoint_x", "endpoint_y"]].to_numpy(dtype=float)
        centroid = endpoints.mean(axis=0)
        dists = np.linalg.norm(endpoints - centroid, axis=1)
        endpoint_var = float(np.trace(np.cov(endpoints, rowvar=False))) if len(endpoints) > 1 else 0.0
        rows.append(
            {
                "regime": regime,
                "prompt_count": int(len(group)),
                "path_length_median": float(group["path_length"].median()),
                "path_length_iqr": float(group["path_length"].quantile(0.75) - group["path_length"].quantile(0.25)),
                "mean_phase_velocity_median": float(group["mean_phase_velocity"].median()),
                "std_step_distance_median": float(group["std_step_distance"].median()),
                "trajectory_curvature_median": float(group["trajectory_curvature"].median()),
                "trajectory_curvature_iqr": float(group["trajectory_curvature"].quantile(0.75) - group["trajectory_curvature"].quantile(0.25)),
                "endpoint_variance": endpoint_var,
                "within_regime_dispersion": float(dists.mean()) if len(dists) else 0.0,
            }
        )
    fingerprint_df = pd.DataFrame(rows).sort_values("regime")
    numeric = fingerprint_df[
        [
            "path_length_median",
            "mean_phase_velocity_median",
            "std_step_distance_median",
            "trajectory_curvature_median",
            "endpoint_variance",
            "within_regime_dispersion",
        ]
    ]
    normalized = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, np.nan)
    min_pairwise_distance = float("nan")
    if len(normalized) > 1:
        distances = []
        arr = normalized.fillna(0.0).to_numpy(dtype=float)
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                distances.append(float(np.linalg.norm(arr[i] - arr[j])))
        min_pairwise_distance = min(distances) if distances else float("nan")
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_df.to_csv(out_dir / "regime_fingerprint.csv", index=False)
    plot_stability(fingerprint_df, out_dir / "regime_stability_fingerprint.png")
    summary = {
        "regime_count": int(len(fingerprint_df)),
        "min_pairwise_fingerprint_distance": min_pairwise_distance,
        "fingerprint_status": "regime_level_signal" if np.isfinite(min_pairwise_distance) and min_pairwise_distance > 1.0 else "prompt_sensitive",
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def write_synthesis(
    detection_summary: dict[str, Any],
    bifurcation_summary: dict[str, Any],
    stability_summary: dict[str, Any],
    out_dir: Path,
) -> None:
    summary = {
        "supported_now_in_gpt2": [],
        "exploratory_only": [],
        "blocked_claims": [],
    }
    if detection_summary["verdict"] == "useful":
        summary["supported_now_in_gpt2"].append("geometry_detection_score separates hallucination-prone prompts better than entropy on the current panel")
    else:
        summary["exploratory_only"].append("pre-hallucination detection signal exists but is not yet clearly better than entropy alone")
    summary["supported_now_in_gpt2"].append(
        f"layer divergence is best described as {bifurcation_summary['conclusion']} in the current GPT-2 panel"
    )
    summary["supported_now_in_gpt2"].append(
        f"largest hallucination expansion currently occurs at {bifurcation_summary['largest_hallucination_expansion_step']['layer_start']}->{bifurcation_summary['largest_hallucination_expansion_step']['layer_end']}"
    )
    if stability_summary["fingerprint_status"] == "regime_level_signal":
        summary["supported_now_in_gpt2"].append("regime stability fingerprints are distinct at regime level in the current setup")
    else:
        summary["exploratory_only"].append("regime stability fingerprints remain partly prompt-sensitive")
    summary["blocked_claims"].extend(
        [
            "cross-model generalization",
            "production-ready early warning",
            "attractor-level claims",
        ]
    )
    write_json(out_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-jsonl", default="data/prompts_observability_panel_2026-03-07.jsonl")
    parser.add_argument("--readonly-trace", required=True)
    parser.add_argument("--baseline-trace", required=True)
    parser.add_argument("--recon-trace", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    panel_df = load_panel(Path(args.panel_jsonl))
    readonly_df = add_geometry_columns(prepare_readonly(load_trace(Path(args.readonly_trace)), panel_df))
    baseline_df = add_geometry_columns(prepare_unified(load_trace(Path(args.baseline_trace)), panel_df))
    recon_df = add_geometry_columns(prepare_unified(load_trace(Path(args.recon_trace)), panel_df))

    readonly_prompt_df = prompt_features(readonly_df)
    baseline_prompt_df = prompt_features(baseline_df)
    recon_prompt_df = prompt_features(recon_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block_metadata = {
        "panel": args.panel_jsonl,
        "readonly_trace": args.readonly_trace,
        "baseline_trace": args.baseline_trace,
        "recon_trace": args.recon_trace,
        "run_family": "trajectory_block",
    }
    write_json(out_dir / "block_metadata.json", block_metadata)
    readonly_prompt_df.to_csv(out_dir / "readonly_prompt_features.csv", index=False)
    baseline_prompt_df.to_csv(out_dir / "baseline_prompt_features.csv", index=False)
    recon_prompt_df.to_csv(out_dir / "recon_prompt_features.csv", index=False)

    detection_summary = analyze_detection(readonly_prompt_df, baseline_prompt_df, out_dir / "detection")
    bifurcation_summary = analyze_bifurcation(readonly_df, out_dir / "bifurcation")
    stability_summary = analyze_stability(readonly_prompt_df, out_dir / "stability")
    write_synthesis(detection_summary, bifurcation_summary, stability_summary, out_dir / "synthesis")
    print(f"Saved trajectory block analysis to {out_dir}")


if __name__ == "__main__":
    main()
