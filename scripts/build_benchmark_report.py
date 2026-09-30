"""Assemble a markdown benchmark report from accuracy + runtime tables and a figure.

Reads:
  <eval-dir>/accuracy_table.tsv     — from evaluate_accuracy.py
  <eval-dir>/accuracy_figure.pdf    — from evaluate_accuracy.py
  <eval-dir>/runtimes_long.tsv      — from collect_runtimes.py (optional)
  <eval-dir>/runtimes_pivot.tsv     — from collect_runtimes.py (optional)

Writes:
  <eval-dir>/REPORT.md              — assembled markdown

Usage:
  python scripts/build_benchmark_report.py \\
      --eval-dir results/vertebrates_test/eval_accuracy_run010_varus \\
      --title "Vertebrates test — VARUS.bam campaign (run010)" \\
      [--figure-png accuracy_figure.png]

Note: PDFs are not directly renderable inside GitHub-flavored markdown. To
embed an image, first render the PDF to PNG (e.g. `pdftoppm -png -r 150
accuracy_figure.pdf accuracy_figure`) then pass --figure-png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LEVEL_COLS = [
    "gene_S", "gene_P", "gene_F1",
    "transcript_S", "transcript_P", "transcript_F1",
    "exon_S", "exon_P", "exon_F1",
]


def _to_md(df: pd.DataFrame, index: bool = True) -> str:
    """Minimal GitHub-flavored markdown table (no tabulate dependency)."""
    df = df.copy()
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _accuracy_section(eval_dir: Path) -> str:
    tsv = eval_dir / "accuracy_table.tsv"
    if not tsv.exists():
        return "_(accuracy_table.tsv not found)_\n"
    df = pd.read_csv(tsv, sep="\t")

    lines = ["## Accuracy per species\n"]
    for sp, sub in df.groupby("species", sort=False):
        lines.append(f"### {sp.replace('_', ' ')}\n")
        sub_show = sub.drop(columns=["species"]).set_index("gene_set")
        lines.append(sub_show[LEVEL_COLS].pipe(_to_md))
        lines.append("")

    # per-clade average across species
    avg = df.groupby("gene_set", sort=False)[LEVEL_COLS].mean().round(2)
    lines.append("## Average across species\n")
    lines.append(avg.pipe(_to_md))
    lines.append("")
    return "\n".join(lines)


def _runtime_section(eval_dir: Path) -> str:
    pivot = eval_dir / "runtimes_pivot.tsv"
    long = eval_dir / "runtimes_long.tsv"
    parts = ["## Runtimes\n"]
    if pivot.exists():
        pv = pd.read_csv(pivot, sep="\t", index_col=0)
        # convert seconds to HH:MM:SS
        def _hms(s):
            if pd.isna(s):
                return ""
            s = int(s)
            return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
        pv_h = pv.map(_hms)
        parts.append("### Per (species × step) — HH:MM:SS\n")
        parts.append(pv_h.pipe(_to_md))
        parts.append("")

        # totals per step
        if long.exists():
            df = pd.read_csv(long, sep="\t")
            step_total = df.groupby(["tool", "phase"])["seconds"].sum().reset_index()
            step_total["hms"] = step_total["seconds"].map(_hms)
            parts.append("### Totals per step (all species)\n")
            parts.append(_to_md(step_total, index=False))
            parts.append("")
    else:
        parts.append("_(runtimes_pivot.tsv not found — did you run collect_runtimes.py?)_\n")
    return "\n".join(parts)


def _figure_section(eval_dir: Path, figure_png: str | None) -> str:
    if figure_png:
        return f"## Accuracy figure\n\n![accuracy]({figure_png})\n"
    if (eval_dir / "accuracy_figure.pdf").exists():
        return ("## Accuracy figure\n\n"
                "See `accuracy_figure.pdf` in this directory. "
                "To embed inline, render to PNG and re-run with `--figure-png`.\n")
    return "## Accuracy figure\n\n_(no figure found)_\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--title", type=str, required=True)
    ap.add_argument("--figure-png", type=str, default=None,
                    help="Relative filename of the accuracy PNG (rendered from PDF).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Override default <eval-dir>/REPORT.md")
    args = ap.parse_args()

    if not args.eval_dir.exists():
        raise SystemExit(f"missing eval-dir: {args.eval_dir}")

    out = args.out or (args.eval_dir / "REPORT.md")

    parts = [f"# {args.title}\n"]
    parts.append(_figure_section(args.eval_dir, args.figure_png))
    parts.append(_accuracy_section(args.eval_dir))
    parts.append(_runtime_section(args.eval_dir))

    out.write_text("\n".join(parts))
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
