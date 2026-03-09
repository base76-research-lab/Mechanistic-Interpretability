#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
COMPARE_SCRIPT = ROOT / "scripts" / "compare_compression_vectorized.py"
DEFAULT_PROMPTS_FILE = ROOT / "data" / "prompts_sanity_2026-03-09.txt"
OUT_DIR = ROOT / "experiments" / "exp_003_compression_vectorized"
SANITY_DIR = OUT_DIR / "sanity_passes"

GO_TO_ROBUST_BATCH = "go_to_robust_batch"
BLOCKED_COMPRESSOR_NO_OP = "blocked_compressor_no_op"
BLOCKED_INVALID_COMPRESSION = "blocked_invalid_compression"
BLOCKED_FALLBACK_CONTAMINATION = "blocked_fallback_contamination"
BLOCKED_STRUCTURE_DEGRADATION = "blocked_structure_degradation"


def now_utc_compact() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def latest_path(pattern: str) -> Path | None:
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def detect_newest_output(before: set[str], suffix_pattern: str) -> Path | None:
    after = set(glob.glob(suffix_pattern))
    created = sorted(after - before)
    if created:
        return Path(created[-1])
    return latest_path(suffix_pattern)


def run_compare_script(args: argparse.Namespace) -> tuple[Path | None, Path | None, str | None]:
    summary_pattern = str(OUT_DIR / "summary_*.json")
    results_pattern = str(OUT_DIR / "results_*.json")
    before_summaries = set(glob.glob(summary_pattern))
    before_results = set(glob.glob(results_pattern))

    cmd = [
        "python3",
        str(COMPARE_SCRIPT),
        "--prompts-file",
        str(args.prompts_file),
        "--prompts-jsonl",
        "",
        "--require-compressor",
        "--exclude-invalid-compression",
        "--device",
        args.device,
    ]

    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    stdout_stderr = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()

    if completed.returncode != 0:
        return None, None, stdout_stderr or "sanity command failed"

    summary_path = detect_newest_output(before_summaries, summary_pattern)
    results_path = detect_newest_output(before_results, results_pattern)
    return summary_path, results_path, None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def classify_stop_go(summary: dict[str, Any], run_error: str | None) -> tuple[str, list[str], str]:
    if run_error:
        if "Token compressor unavailable" in run_error:
            return (
                BLOCKED_COMPRESSOR_NO_OP,
                ["compressor unavailable under `--require-compressor`"],
                "Stop and debug compressor availability",
            )
        return (
            BLOCKED_INVALID_COMPRESSION,
            [run_error],
            "Stop and debug invalid compression path",
        )

    overall = summary.get("overall", {})
    best_current = summary.get("best_current", {})
    non_raw = {k: v for k, v in overall.items() if k != "raw"}
    non_raw_passes = [name for name, row in non_raw.items() if row.get("pass_fail") == "pass"]
    best_structure = best_current.get("best_structure_preserving_method")

    if non_raw_passes and best_structure:
        return (
            GO_TO_ROBUST_BATCH,
            [f"best structure-preserving method: {best_structure}"],
            "Proceed to Tuesday robust batch",
        )

    invalid_heavy = []
    fallback_heavy = []
    no_op_like = []
    structural_failures = []

    for name, row in non_raw.items():
        invalid_rate = row.get("invalid_rate")
        fallback_rate = row.get("fallback_rate")
        token_delta = row.get("median_token_delta")
        compression_ratio = row.get("median_compression_ratio")
        failure_modes = set(row.get("failure_modes", []))

        if invalid_rate is not None and float(invalid_rate) >= 0.5:
            invalid_heavy.append(name)
        if fallback_rate is not None and float(fallback_rate) > float(summary.get("thresholds", {}).get("fallback_rate", 0.25)):
            fallback_heavy.append(name)
        if (
            (token_delta is None or float(token_delta) <= 0.0)
            and (compression_ratio is None or abs(float(compression_ratio) - 1.0) <= 1e-6)
        ):
            no_op_like.append(name)
        if failure_modes.intersection({"frontier collapse", "decision drift", "semantic loss", "layer mismatch", "over-compression"}):
            structural_failures.append(name)

    if invalid_heavy and len(invalid_heavy) == len(non_raw):
        if no_op_like:
            return (
                BLOCKED_COMPRESSOR_NO_OP,
                [f"non-raw variants effectively unusable and compression appears inactive: {', '.join(sorted(no_op_like))}"],
                "Stop and debug compressor availability",
            )
        return (
            BLOCKED_INVALID_COMPRESSION,
            [f"invalid compression dominates non-raw variants: {', '.join(sorted(invalid_heavy))}"],
            "Stop and debug invalid compression path",
        )

    if fallback_heavy and len(fallback_heavy) >= max(1, len(non_raw)):
        return (
            BLOCKED_FALLBACK_CONTAMINATION,
            [f"fallback contamination exceeds threshold in: {', '.join(sorted(fallback_heavy))}"],
            "Stop and inspect fallback contamination",
        )

    if no_op_like and len(no_op_like) == len(non_raw):
        return (
            BLOCKED_COMPRESSOR_NO_OP,
            [f"compression pipeline shows no meaningful engagement in: {', '.join(sorted(no_op_like))}"],
            "Stop and debug compressor availability",
        )

    if structural_failures:
        return (
            BLOCKED_STRUCTURE_DEGRADATION,
            [f"structural degradation dominates failing variants: {', '.join(sorted(structural_failures))}"],
            "Stop and inspect structure degradation before any batch run",
        )

    return (
        BLOCKED_INVALID_COMPRESSION,
        ["no non-raw method passed and no clearer block reason was isolated"],
        "Stop and debug invalid compression path",
    )


def write_internal_note(
    *,
    status: str,
    reasons: list[str],
    next_step: str,
    command_text: str,
    summary_path: Path | None,
    results_path: Path | None,
    run_error: str | None,
) -> tuple[Path, Path]:
    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    ts = now_utc_compact()
    json_path = SANITY_DIR / f"sanity_pass_{ts}.json"
    md_path = SANITY_DIR / f"sanity_pass_{ts}.md"

    payload = {
        "timestamp_utc": ts,
        "status": status,
        "reasons": reasons,
        "next_step": next_step,
        "command_executed": command_text,
        "summary_json_path": str(summary_path.relative_to(ROOT)) if summary_path else None,
        "results_json_path": str(results_path.relative_to(ROOT)) if results_path else None,
        "run_error": run_error,
        "internal_only": True,
        "artifact_status": "agent-drafted",
        "principle_ref": "RCP-2026-01",
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "---",
        "autonomy_level: semi-autonomous",
        "principle_ref: RCP-2026-01",
        "last_human_review: null",
        "artifact_status: agent-drafted",
        "internal_only: true",
        "---",
        "",
        "# Monday Sanity Pass Note",
        "",
        f"- Timestamp: `{ts}`",
        f"- Status: `{status}`",
        f"- Next step: {next_step}",
        "",
        "## Command executed",
        "",
        "```bash",
        command_text,
        "```",
        "",
        "## Generated artifacts",
        "",
        f"- Summary JSON: `{summary_path.relative_to(ROOT)}`" if summary_path else "- Summary JSON: `none`",
        f"- Results JSON: `{results_path.relative_to(ROOT)}`" if results_path else "- Results JSON: `none`",
        f"- Sanity pass JSON: `{json_path.relative_to(ROOT)}`",
        "",
        "## Reason summary",
        "",
    ]
    if reasons:
        lines.extend([f"- {reason}" for reason in reasons])
    else:
        lines.append("- none")

    if run_error:
        lines.extend(["", "## Run error", "", "```text", run_error, "```"])

    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the Monday sanity pass as an internal-only agent workflow.")
    ap.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_FILE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--summary-json", type=Path, default=None, help="Optional existing summary JSON to classify without re-running.")
    ap.add_argument("--results-json", type=Path, default=None, help="Optional matching results JSON when using --summary-json.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    command_text = (
        f'cd "{ROOT}"\n'
        f"python3 scripts/compare_compression_vectorized.py "
        f"--prompts-file {args.prompts_file} "
        f"--prompts-jsonl '' "
        f"--require-compressor --exclude-invalid-compression --device {args.device}"
    )

    if args.summary_json:
        summary_path = args.summary_json.resolve()
        results_path = args.results_json.resolve() if args.results_json else None
        run_error = None
    else:
        summary_path, results_path, run_error = run_compare_script(args)

    summary = load_json(summary_path) if summary_path and summary_path.exists() else {}
    status, reasons, next_step = classify_stop_go(summary, run_error)
    json_path, md_path = write_internal_note(
        status=status,
        reasons=reasons,
        next_step=next_step,
        command_text=command_text,
        summary_path=summary_path,
        results_path=results_path,
        run_error=run_error,
    )

    print(status)
    print(f"sanity_note_markdown={md_path}")
    print(f"sanity_note_json={json_path}")
    if summary_path:
        print(f"summary_json={summary_path}")
    if results_path:
        print(f"results_json={results_path}")


if __name__ == "__main__":
    main()
