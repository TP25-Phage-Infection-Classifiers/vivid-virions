#make_markdown_report.py
"""
Create a visual Markdown report that compares MLP, RandomForest, and XGBoost across splits.

Inputs (produced by: python -m src.models.comparison)
----------------------------------------------------
pipeline_results/report/
  metrics_all_splits.csv
  metrics_lines_*.png
  metrics_boxplot_*.png
  metrics_radar.png
  confusion_matrices_split_*.png

Output
------
pipeline_results/report/report.md

Usage
-----
python -m src.models.make_markdown_report
python -m src.models.make_markdown_report --regenerate
python -m src.models.make_markdown_report --report-dir pipeline_results/report --title "Custom Title"
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import datetime as dt
import importlib
import pandas as pd

# ----------------------------
# Core helpers (no globals)
# ----------------------------
def ensure_generated_artifacts(report_dir: Path, regenerate: bool) -> None:
    """
    Ensure metrics CSV and figures exist. If not or if --regenerate, call src.models.comparison.main().
    """
    metrics_csv = report_dir / "metrics_all_splits.csv"
    need = regenerate or (not metrics_csv.exists())
    if need:
        try:
            comp = importlib.import_module("src.models.comparison")
            print("Regenerating metrics & plots with src.models.comparison ...")
            comp.main()  # writes CSV + PNGs under report_dir
        except Exception as e:
            print("⚠ Could not run src.models.comparison automatically.")
            print("   Error:", e)
            print("   Please run:  python -m src.models.comparison")
            sys.exit(1)
    if not metrics_csv.exists():
        print(f"⚠ {metrics_csv} not found. Run `python -m src.models.comparison` first, or use --regenerate.")
        sys.exit(1)


def fmt_mean_std(series: pd.Series) -> str:
    return f"{series.mean():.3f} ± {series.std():.3f}"


def build_tables(df: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (summary_md, per_split_md)
    - summary_md: mean ± std per model
    - per_split_md: per metric pivot table (Split x Model)
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1"]

    # Summary: mean ± std per model
    rows = []
    for model, g in df.groupby("Model"):
        row = {"Model": model}
        for m in metrics:
            row[m] = fmt_mean_std(g[m])
        rows.append(row)
    summary_df = pd.DataFrame(rows, columns=["Model"] + metrics).sort_values("Model")
    summary_md = summary_df.to_markdown(index=False)

    # Per-split tables (one table per metric to keep width manageable)
    parts = []
    for m in metrics:
        pivot = df.pivot(index="Split", columns="Model", values=m).sort_index()
        parts.append(f"\n**{m} per Split**\n\n{pivot.to_markdown(index=True)}\n")
    per_split_md = "\n".join(parts)

    return summary_md, per_split_md


def list_existing_figs(report_dir: Path) -> dict[str, list[Path]]:
    """Collect figure paths (only those that exist)."""
    out = {
        "lines": [
            report_dir / "metrics_lines_accuracy.png",
            report_dir / "metrics_lines_precision.png",
            report_dir / "metrics_lines_recall.png",
            report_dir / "metrics_lines_f1.png",
        ],
        "boxes": [
            report_dir / "metrics_boxplot_accuracy.png",
            report_dir / "metrics_boxplot_precision.png",
            report_dir / "metrics_boxplot_recall.png",
            report_dir / "metrics_boxplot_f1.png",
        ],
        "radar": [report_dir / "metrics_radar.png"],
        "grids": sorted(report_dir.glob("confusion_matrices_split_*.png")),
    }
    # Filter to existing
    for k, lst in out.items():
        out[k] = [p for p in lst if p.exists()]
    return out


def write_markdown(report_dir: Path, title: str, summary_md: str, per_split_md: str) -> Path:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    md_path = report_dir / "report.md"
    figs = list_existing_figs(report_dir)

    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"_Generated: {ts}_\n")
    lines.append("\n---\n")

    # Intro
    lines.append("## Overview\n")
    lines.append(
        "This report compares **XGBoost**, **MLP**, and **RandomForest** over LOGO splits (0–5) "
        "and a stratified split (6). Metrics include **Accuracy**, **Precision**, **Recall**, and **F1**.\n"
    )

    # Summary
    lines.append("## Summary (Mean ± Std across Splits)\n")
    lines.append(summary_md)
    lines.append("\n")

    # Line charts
    if figs["lines"]:
        lines.append("## Metrics Across Splits (Line Charts)\n")
        for fig in figs["lines"]:
            lines.append(f"![{fig.stem}]({fig.name})\n")

    # Box plots
    if figs["boxes"]:
        lines.append("\n## Metrics Distribution (Box Plots)\n")
        for fig in figs["boxes"]:
            lines.append(f"![{fig.stem}]({fig.name})\n")

    # Radar
    if figs["radar"]:
        lines.append("\n## Radar Plot (Mean Across Splits)\n")
        lines.append(f"![{figs['radar'][0].stem}]({figs['radar'][0].name})\n")

    # Per-split tables
    lines.append("\n## Per-Split Metrics (Tables)\n")
    lines.append(per_split_md)
    lines.append("\n")

    # Confusion grids
    if figs["grids"]:
        lines.append("## Confusion Matrices (Normalized, per Split)\n")
        for g in figs["grids"]:
            lines.append(f"### {g.stem}\n")
            lines.append(f"![{g.stem}]({g.name})\n")

    # Files inventory
    lines.append("\n---\n")
    lines.append("## Files\n")
    lines.append("- `metrics_all_splits.csv` — raw metrics per model/split\n")
    lines.append("- `metrics_lines_*.png`, `metrics_boxplot_*.png`, `metrics_radar.png` — summary figures\n")
    lines.append("- `confusion_matrices_split_*.png` — per-split confusion matrix grids\n")

    report_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default="pipeline_results/report",
                    help="Directory containing metrics/figures and where report.md will be written.")
    ap.add_argument("--title", default="Phage Classification – Model Comparison Report")
    ap.add_argument("--regenerate", action="store_true",
                    help="Recompute metrics & plots via src.models.comparison before writing the report.")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Ensure metrics & plots exist (or rebuild)
    ensure_generated_artifacts(report_dir, regenerate=args.regenerate)

    # Load metrics
    metrics_csv = report_dir / "metrics_all_splits.csv"
    df = pd.read_csv(metrics_csv)
    if df.empty:
        print(f"⚠ No rows in {metrics_csv}. Nothing to report.")
        sys.exit(1)

    # Build tables and write markdown
    summary_md, per_split_md = build_tables(df)
    md_path = write_markdown(report_dir, args.title, summary_md, per_split_md)
    print(f"✅ Wrote Markdown report -> {md_path.resolve()}")


if __name__ == "__main__":
    main()
