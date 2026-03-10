from __future__ import annotations

import argparse
from pathlib import Path

from .trace import run_trace
from .viz import run_viz


def main() -> None:
    ap = argparse.ArgumentParser(description="Transformer Oscilloscope CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # trace
    ap_trace = sub.add_parser("trace", help="run read-only trace collection")
    ap_trace.add_argument("--prompt-jsonl", required=True)
    ap_trace.add_argument("--model", default="gpt2")
    ap_trace.add_argument("--layers", nargs="+", type=int, default=[1, 6, 9, 11])
    ap_trace.add_argument("--device", default="cpu")
    ap_trace.add_argument("--out-dir", required=True)
    ap_trace.add_argument("--run-name", default="")
    ap_trace.add_argument("--topk", type=int, default=5)
    ap_trace.add_argument("--store-projections", action="store_true")
    ap_trace.add_argument("--sae-state", type=str, default=None)
    ap_trace.add_argument("--sae-topk", type=int, default=8)
    ap_trace.add_argument("--units", nargs="+", type=int, default=[472, 468, 57, 156, 346])
    ap_trace.add_argument("--basis-mode", choices=["mean", "pc1", "pc2"], default="pc2")

    # viz
    ap_viz = sub.add_parser("viz", help="generate plots from trace.jsonl")
    ap_viz.add_argument("--trace", required=True)
    ap_viz.add_argument("--out-dir", required=True)

    # report
    ap_rep = sub.add_parser("report", help="generate HTML report from trace.jsonl")
    ap_rep.add_argument("--trace", required=True)
    ap_rep.add_argument("--out-dir", required=True)
    ap_rep.add_argument("--report-name", default="report.html")

    args = ap.parse_args()

    if args.cmd == "trace":
        run_name = args.run_name or f"transformer_oscilloscope_{Path(args.prompt_jsonl).stem}"
        trace_path = run_trace(
            prompt_jsonl=Path(args.prompt_jsonl),
            model_name=args.model,
            layers=[int(l) for l in args.layers],
            device=args.device,
            out_dir=Path(args.out_dir) / run_name,
            run_name=run_name,
            topk=args.topk,
            store_projections=args.store_projections,
            sae_state=Path(args.sae_state) if args.sae_state else None,
            sae_topk=args.sae_topk,
            units=args.units,
            basis_mode=args.basis_mode,
        )
        print(f"Saved trace: {trace_path}")
    elif args.cmd == "viz":
        run_viz(Path(args.trace), Path(args.out_dir))
        print(f"Saved plots to {args.out_dir}")
    elif args.cmd == "report":
        from .report import build_report

        html = build_report(Path(args.trace), Path(args.out_dir))
        out_path = Path(args.out_dir) / args.report_name
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
