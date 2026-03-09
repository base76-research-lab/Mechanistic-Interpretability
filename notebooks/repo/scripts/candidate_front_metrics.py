#!/usr/bin/env python3
"""
candidate_front_metrics.py

Compute candidate-front diagnostics from field_view JSON artifacts:
- candidate_coherence: mean pairwise cosine similarity among candidate coords
- candidate_variance: trace(cov(candidate_coords))
- degeneracy_ratio_topk: fraction of candidates that look like generic fallback tokens
- candidate_centroid: mean of candidate coords in subspace
- state_to_centroid_distance: Euclidean distance between field state and candidate centroid
- degenerate: boolean flag from degeneracy_ratio threshold

Usage examples:
  python3 scripts/candidate_front_metrics.py \
    --input experiments/exp_001_sae_v3/field_view_hallucination.json

  python3 scripts/candidate_front_metrics.py \
    --glob "experiments/exp_001_sae_v3/field_view*.json" \
    --out experiments/exp_001_sae_v3/candidate_front_metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


GENERIC_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "i",
    "he",
    "she",
    "it",
    "they",
    "that",
    "this",
    "these",
    "those",
    "we",
    "you",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
}

GENERIC_SPECIAL = {"<space>"}


def is_punctuation_token(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return True
    return re.fullmatch(r"[^\w]+", stripped) is not None


def is_generic_token(token: str) -> bool:
    lower = token.strip().lower()
    if lower in GENERIC_SPECIAL:
        return True
    if lower in GENERIC_WORDS:
        return True
    if is_punctuation_token(token):
        return True
    return False


def mean_pairwise_cosine(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return float("nan")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1e-12, norms)
    normalized = matrix / norms
    sim = normalized @ normalized.T
    upper = sim[np.triu_indices(sim.shape[0], k=1)]
    return float(np.mean(upper))


def trace_covariance(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return 0.0
    cov = np.cov(matrix, rowvar=False)
    return float(np.trace(cov))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compute_metrics(data: dict[str, Any], source: Path) -> dict[str, Any]:
    candidates = data.get("candidates", [])
    coords = np.array([c.get("coords", []) for c in candidates], dtype=float)
    tokens = [str(c.get("token", "")) for c in candidates]

    if coords.ndim != 2 or coords.shape[0] == 0:
        raise ValueError(f"No valid candidates/coords in {source}")

    candidate_coherence = mean_pairwise_cosine(coords)
    candidate_variance = trace_covariance(coords)
    candidate_centroid_vec = np.mean(coords, axis=0)
    candidate_centroid = [float(x) for x in candidate_centroid_vec.tolist()]
    generic_count = sum(1 for token in tokens if is_generic_token(token))
    degeneracy_ratio = generic_count / len(tokens)

    operator_strength = data.get("operator_strength", data.get("state_norm"))
    field_coords = data.get("field_coords")
    state_to_centroid_distance = None
    if isinstance(field_coords, list) and len(field_coords) == coords.shape[1]:
        state = np.array(field_coords, dtype=float)
        state_to_centroid_distance = float(np.linalg.norm(state - candidate_centroid_vec))

    metric = {
        "file": str(source),
        "prompt": data.get("prompt"),
        "topk": int(data.get("topk", len(tokens))),
        "risk_score": float(data.get("risk_score")) if data.get("risk_score") is not None else None,
        "logit_entropy": float(data.get("logit_entropy")) if data.get("logit_entropy") is not None else None,
        "gap_state_to_candidates": float(data.get("gap_state_to_candidates"))
        if data.get("gap_state_to_candidates") is not None
        else None,
        "operator_strength": float(operator_strength) if operator_strength is not None else None,
        "candidate_coherence": candidate_coherence,
        "candidate_variance": candidate_variance,
        "candidate_centroid": candidate_centroid,
        "state_to_centroid_distance": state_to_centroid_distance,
        "degeneracy_ratio_topk": degeneracy_ratio,
        "degenerate": degeneracy_ratio > 0.7,
        "generic_count": generic_count,
        "candidate_count": len(tokens),
        "generic_tokens": [token for token in tokens if is_generic_token(token)],
    }
    return metric


def parse_inputs(single: str | None, pattern: str | None) -> list[Path]:
    files: list[Path] = []
    if single:
        files.append(Path(single))
    if pattern:
        for match in sorted(glob.glob(pattern)):
            files.append(Path(match))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path.resolve())
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    if not deduped:
        raise ValueError("No input files. Provide --input and/or --glob.")
    for path in deduped:
        if not path.exists():
            raise FileNotFoundError(path)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Single field_view JSON file")
    parser.add_argument("--glob", type=str, help='Glob pattern, e.g. "experiments/exp_001_sae_v3/field_view*.json"')
    parser.add_argument("--out", type=str, help="Optional output JSON path")
    parser.add_argument("--degenerate-threshold", type=float, default=0.7, help="Threshold for degenerate flag")
    args = parser.parse_args()

    files = parse_inputs(args.input, args.glob)
    results = []
    for path in files:
        data = load_json(path)
        metric = compute_metrics(data, path)
        metric["degenerate"] = metric["degeneracy_ratio_topk"] > args.degenerate_threshold
        results.append(metric)

    # Stable ordering for tables/plots.
    results = sorted(
        results,
        key=lambda row: (
            -math.inf if row["risk_score"] is None else row["risk_score"],
            -math.inf if row["degeneracy_ratio_topk"] is None else row["degeneracy_ratio_topk"],
            row["file"],
        ),
        reverse=True,
    )

    print("file\trisk\tH\tgap\t|coords|\tcoherence\tvariance\tdegeneracy\tdegenerate\td_state_centroid")

    def fmt(value: Any) -> str:
        if value is None:
            return "nan"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "nan"

    for row in results:
        print(
            f'{row["file"]}\t'
            f'{fmt(row["risk_score"])}\t'
            f'{fmt(row["logit_entropy"])}\t'
            f'{fmt(row["gap_state_to_candidates"])}\t'
            f'{fmt(row["operator_strength"])}\t'
            f'{fmt(row["candidate_coherence"])}\t'
            f'{fmt(row["candidate_variance"])}\t'
            f'{fmt(row["degeneracy_ratio_topk"])}\t'
            f'{str(bool(row["degenerate"])).lower()}\t'
            f'{fmt(row["state_to_centroid_distance"])}'
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
