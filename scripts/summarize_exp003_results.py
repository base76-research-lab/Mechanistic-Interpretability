#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRIC_DIRECTIONS = {
    "delta_vs_raw_gap_state_to_candidates": "lower",
    "delta_vs_raw_candidate_coherence": "higher",
    "delta_vs_raw_degeneracy_ratio_topk": "lower",
    "delta_vs_raw_logit_entropy": "lower",
}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {
        "count": len(values),
        "median": statistics.median(values) if values else 0.0,
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
    }


def method_rank(summary: dict[str, dict]) -> list[dict]:
    methods = list(summary.keys())
    ranked = methods[:]
    ranked.sort(
        key=lambda method: (
            summary[method]["metrics"]["delta_vs_raw_gap_state_to_candidates"]["median"],
            -summary[method]["metrics"]["delta_vs_raw_candidate_coherence"]["median"],
            summary[method]["metrics"]["delta_vs_raw_degeneracy_ratio_topk"]["median"],
            summary[method]["metrics"]["delta_vs_raw_logit_entropy"]["median"],
        )
    )
    return [{"method": method, "rank": index + 1} for index, method in enumerate(ranked)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    filtered = []
    for row in rows:
        variant = row.get("variant", "")
        if not (
            variant.startswith("compressed_vectorized_proxy_")
            or variant == "compressed_vectorized_proxy"
        ):
            continue
        if not row.get("compression_valid", True):
            continue
        filtered.append(row)
    if not filtered:
        raise SystemExit("No valid vectorized rows found.")

    grouped: dict[str, list[dict]] = defaultdict(list)
    grouped_by_stratum: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in filtered:
        method = row["variant"].replace("compressed_vectorized_proxy_", "")
        if method == "compressed_vectorized_proxy":
            method = "mean"
        grouped[method].append(row)
        grouped_by_stratum[method][row.get("prompt_stratum", "unknown")].append(row)

    summary: dict[str, dict] = {}
    for method, method_rows in grouped.items():
        metric_summary = {}
        for metric in METRIC_DIRECTIONS:
            values = [float(row[metric]) for row in method_rows if row.get(metric) is not None]
            metric_summary[metric] = summarize(values)
        per_stratum = {}
        for stratum, stratum_rows in grouped_by_stratum[method].items():
            per_stratum[stratum] = {
                metric: summarize([float(row[metric]) for row in stratum_rows if row.get(metric) is not None])
                for metric in METRIC_DIRECTIONS
            }
        summary[method] = {
            "rows": len(method_rows),
            "metrics": metric_summary,
            "per_stratum": per_stratum,
        }

    ranking = method_rank(summary)
    out_payload = {
        "source_results": args.results,
        "summary": summary,
        "ranking": ranking,
    }
    Path(args.out_json).write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    lines = [
        "# exp_003 Method Comparison",
        "",
        f"Source results: `{args.results}`",
        "",
        "## Ranking",
        "",
    ]
    for item in ranking:
        lines.append(f"- #{item['rank']} `{item['method']}`")
    lines.extend(["", "## Overall Metrics", "", "| Method | gap median | coherence median | degeneracy median | entropy median | rows |", "|---|---:|---:|---:|---:|---:|"])
    for item in ranking:
        method = item["method"]
        metrics = summary[method]["metrics"]
        lines.append(
            f"| {method} | "
            f"{metrics['delta_vs_raw_gap_state_to_candidates']['median']:.4f} | "
            f"{metrics['delta_vs_raw_candidate_coherence']['median']:.4f} | "
            f"{metrics['delta_vs_raw_degeneracy_ratio_topk']['median']:.4f} | "
            f"{metrics['delta_vs_raw_logit_entropy']['median']:.4f} | "
            f"{summary[method]['rows']} |"
        )
    lines.extend(["", "## Prompt-Type Breakdown", ""])
    for item in ranking:
        method = item["method"]
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Stratum | gap median | coherence median | degeneracy median | entropy median |")
        lines.append("|---|---:|---:|---:|---:|")
        for stratum, metrics in sorted(summary[method]["per_stratum"].items()):
            lines.append(
                f"| {stratum} | "
                f"{metrics['delta_vs_raw_gap_state_to_candidates']['median']:.4f} | "
                f"{metrics['delta_vs_raw_candidate_coherence']['median']:.4f} | "
                f"{metrics['delta_vs_raw_degeneracy_ratio_topk']['median']:.4f} | "
                f"{metrics['delta_vs_raw_logit_entropy']['median']:.4f} |"
            )
        lines.append("")
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {args.out_json}")
    print(f"Saved {args.out_md}")


if __name__ == "__main__":
    main()
