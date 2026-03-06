#!/usr/bin/env python3
"""
make_figures.py

Genererar små, repo-vänliga figurer för README/rapporter från befintliga JSON-artefakter.
Kör lokalt efter att exp_001_sae_v3 körts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fig_field_view_triage() -> None:
    exp = ROOT / "experiments" / "exp_001_sae_v3"
    files = {
        "math (det)": exp / "field_view_math.json",
        "analogy (reason)": exp / "field_view_analogy_run2.json",
        "hallucination": exp / "field_view_hallucination.json",
    }
    rows = []
    for label, path in files.items():
        if not path.exists():
            continue
        data = load_json(path)
        rows.append(
            {
                "label": label,
                "risk": float(data.get("risk_score", 0.0)),
                "H": float(data.get("logit_entropy", 0.0)),
                "op": float(data.get("operator_strength", 0.0)),
                "gap": float(data.get("gap_state_to_candidates", 0.0)),
            }
        )

    if not rows:
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "field_view_triage.png"

    plt.figure(figsize=(7.2, 4.4), dpi=160)
    plt.title("Field View triage (GPT-2, antonym subspace)")
    plt.xlabel("operator_strength (||field_coords||)")
    plt.ylabel("risk_score (entropy + gap)")
    plt.grid(alpha=0.25)

    for r in rows:
        plt.scatter([r["op"]], [r["risk"]], s=70)
        plt.annotate(
            f'{r["label"]}\nH={r["H"]:.2f}, gap={r["gap"]:.2f}',
            (r["op"], r["risk"]),
            textcoords="offset points",
            xytext=(8, 8),
            ha="left",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    fig_field_view_triage()


if __name__ == "__main__":
    main()

