#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_countercase(profiles: pd.DataFrame, out_path: Path) -> None:
    colors = {
        "hallucination_prone": "#b91c1c",
        "transition": "#c2410c",
        "reasoning": "#1d4ed8",
        "anchored": "#166534",
        "control": "#6b7280",
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for regime, group in profiles.groupby("regime"):
        ax.scatter(
            group["lead_tokens"],
            group["post_onset_persistence"],
            label=regime,
            color=colors.get(regime, "#111827"),
            s=90,
            alpha=0.85,
        )
        for row in group.itertuples():
            ax.annotate(row.prompt_id, (row.lead_tokens, row.post_onset_persistence), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Lead tokens")
    ax.set_ylabel("Post-onset persistence")
    ax.set_title("Transition Counter-Case Profiles")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_profiles(per_token_df: pd.DataFrame, detected_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for detected in detected_df.itertuples(index=False):
        onset_token = int(detected.onset_token)
        prompt_tokens = per_token_df[per_token_df["prompt_id"] == detected.prompt_id].sort_values("token_index")
        onset_row = prompt_tokens[prompt_tokens["token_index"] == onset_token].iloc[0]
        tail = prompt_tokens[prompt_tokens["token_index"] >= onset_token]
        rows.append(
            {
                "prompt_id": detected.prompt_id,
                "prompt": detected.prompt,
                "regime": detected.regime,
                "stratum": detected.stratum,
                "threshold_label": detected.threshold_label,
                "onset_token": onset_token,
                "lead_tokens": float(detected.lead_tokens),
                "relative_lead": float(detected.relative_lead),
                "post_onset_persistence": float(detected.post_onset_persistence),
                "onset_score_delta": float(onset_row["score_delta"]),
                "onset_entropy_delta": float(onset_row["entropy_delta"]),
                "onset_curvature": float(onset_row["trajectory_curvature"]),
                "onset_phase_velocity": float(onset_row["mean_phase_velocity"]),
                "onset_std_step_distance": float(onset_row["std_step_distance"]),
                "tail_mean_score_delta": float(tail["score_delta"].mean()),
                "tail_min_score_delta": float(tail["score_delta"].min()),
                "tail_mean_entropy_delta": float(tail["entropy_delta"].mean()),
                "tail_mean_curvature": float(tail["trajectory_curvature"].mean()),
                "tail_mean_phase_velocity": float(tail["mean_phase_velocity"].mean()),
                "tail_mean_std_step_distance": float(tail["std_step_distance"].mean()),
                "tail_mean_coherence": float(tail["mean_coherence"].mean()),
                "tail_mean_degeneracy": float(tail["mean_degeneracy"].mean()),
                "last_token_score_delta": float(prompt_tokens[prompt_tokens["token_index"] == prompt_tokens["end_token"]]["score_delta"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead-time-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    lead_time_dir = Path(args.lead_time_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_token_df = pd.read_csv(lead_time_dir / "per_token_summary.csv")
    q90 = pd.read_csv(lead_time_dir / "prompt_lead_time_operational_q90.csv")
    q95 = pd.read_csv(lead_time_dir / "prompt_lead_time_conservative_q95.csv")
    q90["threshold_label"] = "operational_q90"
    q95["threshold_label"] = "conservative_q95"

    detected_q90 = q90[q90["detected"] == True].copy()
    detected_q95 = q95[q95["detected"] == True].copy()
    profiles_q90 = build_profiles(per_token_df, detected_q90)
    profiles_q95 = build_profiles(per_token_df, detected_q95) if not detected_q95.empty else pd.DataFrame()

    regime_summary = (
        profiles_q90.groupby("regime")[
            [
                "lead_tokens",
                "relative_lead",
                "post_onset_persistence",
                "onset_score_delta",
                "onset_entropy_delta",
                "tail_mean_score_delta",
                "tail_mean_entropy_delta",
                "tail_mean_curvature",
                "tail_mean_phase_velocity",
                "tail_mean_std_step_distance",
            ]
        ]
        .median()
        .reset_index()
    )

    plot_countercase(profiles_q90, out_dir / "transition_countercase_scatter.png")
    profiles_q90.to_csv(out_dir / "detected_prompt_profiles_q90.csv", index=False)
    if not profiles_q95.empty:
        profiles_q95.to_csv(out_dir / "detected_prompt_profiles_q95.csv", index=False)
    regime_summary.to_csv(out_dir / "regime_countercase_summary.csv", index=False)

    hallucination_q90 = profiles_q90[profiles_q90["regime"] == "hallucination_prone"]
    transition_q90 = profiles_q90[profiles_q90["regime"] == "transition"]
    reasoning_q90 = profiles_q90[profiles_q90["regime"] == "reasoning"]

    conservative_isolates_hallucination = not profiles_q95.empty and set(profiles_q95["regime"]) == {"hallucination_prone"}
    hallucination_persistence = float(hallucination_q90["post_onset_persistence"].median()) if not hallucination_q90.empty else float("nan")
    transition_persistence = float(transition_q90["post_onset_persistence"].median()) if not transition_q90.empty else float("nan")
    reasoning_persistence = float(reasoning_q90["post_onset_persistence"].median()) if not reasoning_q90.empty else float("nan")

    verdict = "ambiguity_unresolved"
    if conservative_isolates_hallucination:
        verdict = "conservative_threshold_isolates_sparse_hallucination_slice"
    if conservative_isolates_hallucination and pd.notna(hallucination_persistence) and pd.notna(transition_persistence) and hallucination_persistence > transition_persistence:
        verdict = "transition_countercases_differ_by_persistence_but_remain_boundary_cases"

    summary = {
        "verdict": verdict,
        "q90_detected_counts": profiles_q90["regime"].value_counts().to_dict(),
        "q95_detected_counts": profiles_q95["regime"].value_counts().to_dict() if not profiles_q95.empty else {},
        "hallucination_q90_median_persistence": hallucination_persistence,
        "transition_q90_median_persistence": transition_persistence,
        "reasoning_q90_median_persistence": reasoning_persistence,
        "hallucination_q90_median_lead_tokens": float(hallucination_q90["lead_tokens"].median()) if not hallucination_q90.empty else float("nan"),
        "transition_q90_median_lead_tokens": float(transition_q90["lead_tokens"].median()) if not transition_q90.empty else float("nan"),
        "reasoning_q90_median_lead_tokens": float(reasoning_q90["lead_tokens"].median()) if not reasoning_q90.empty else float("nan"),
        "hallucination_q90_median_tail_score_delta": float(hallucination_q90["tail_mean_score_delta"].median()) if not hallucination_q90.empty else float("nan"),
        "transition_q90_median_tail_score_delta": float(transition_q90["tail_mean_score_delta"].median()) if not transition_q90.empty else float("nan"),
        "reasoning_q90_median_tail_score_delta": float(reasoning_q90["tail_mean_score_delta"].median()) if not reasoning_q90.empty else float("nan"),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"Saved transition counter-case analysis to {out_dir}")


if __name__ == "__main__":
    main()
