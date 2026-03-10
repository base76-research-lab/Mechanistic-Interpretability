from __future__ import annotations

import argparse
from pathlib import Path

from .viz import run_viz


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Transformer Oscilloscope Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; }}
    img {{ max-width: 100%; height: auto; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
  </style>
</head>
<body>
  <h1>Transformer Oscilloscope Report</h1>
  <p>Trace: {trace}</p>
  <h2>Overall</h2>
  <div class="grid">
    <div><h3>Entropy</h3><img src="all_entropy.png" /></div>
    <div><h3>Gap</h3><img src="all_gap.png" /></div>
    {all_pca}
  </div>
  {per_prompt}
</body>
</html>
"""


def build_report(trace: Path, plots_dir: Path) -> str:
    # run viz to ensure plots exist
    run_viz(trace, plots_dir)
    per_prompt_blocks = []
    for img in sorted(plots_dir.glob("*_entropy.png")):
        slug = img.stem.replace("_entropy", "")
        gap = plots_dir / f"{slug}_gap.png"
        pca = plots_dir / f"{slug}_pca.png"
        block = f"""
        <h2>Prompt {slug}</h2>
        <div class="grid">
          <div><h3>Entropy</h3><img src="{img.name}" /></div>
          <div><h3>Gap</h3><img src="{gap.name}" /></div>
          {f'<div><h3>PCA</h3><img src=\"{pca.name}\" /></div>' if pca.exists() else ''}
        </div>
        """
        per_prompt_blocks.append(block)

    all_pca = ""
    if (plots_dir / "all_pca.png").exists():
        all_pca = '<div><h3>PCA</h3><img src="all_pca.png" /></div>'

    return HTML_TEMPLATE.format(
        trace=trace,
        all_pca=all_pca,
        per_prompt="\n".join(per_prompt_blocks),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report-name", default="report.html")
    args = ap.parse_args()

    trace = Path(args.trace)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = build_report(trace, out_dir)
    (out_dir / args.report_name).write_text(html, encoding="utf-8")
    print(f"Saved report to {out_dir / args.report_name}")


if __name__ == "__main__":
    main()
