"""Compare ORF-prediction gene sets against a reference annotation using gffcompare.

Four gene sets are evaluated per species:
  orf_prediction  — <pred-dir>/<species>/prediction.gtf
  tiberius        — tiberius_benchmarking/paper/Insecta/<species>/.../tiberius_seqlen.gtf
  merged          — tiberius + orf_prediction via merge_annotations.py --mode full
  braker3         — tiberius_benchmarking/paper/Insecta/<species>/.../braker3.gtf

gffcompare is called as:
  gffcompare --strict-match -e 3 -T -r <ref> -o <prefix> <query>

Outputs:
  <out-dir>/accuracy_table.tsv       — one row per (species, gene_set)
  <out-dir>/accuracy_figure.pdf      — 3 subplots: gene / transcript / exon F1
  <out-dir>/gffcompare_runs/         — raw gffcompare output kept for inspection

Usage:
  python scripts/evaluate_accuracy.py \\
      --pred-dir results/predictions/run_002_epoch41 \\
      --out-dir  results/accuracy/run_002_epoch41
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Fixed cluster paths
# ---------------------------------------------------------------------------
_BENCH = Path("/home/gabriell/tiberius_benchmarking")

REF_TMPL        = str(_BENCH / "Insecta/{sp}/annot_cds.gff")
TIB_TMPL        = str(_BENCH / "paper/Insecta/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf")
TIB_MERGE_TMPL  = str(_BENCH / "paper/Insecta/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf")
BRK_TMPL        = str(_BENCH / "paper/Insecta/{sp}/results/predictions/braker3/braker3.gtf")
MERGE_SCRIPT    = Path("/home/gabriell/tib_hidten/Tiberius/tiberius/scripts/merge_annotations.py")

GENE_SETS = ["orf_prediction", "tiberius", "merged", "braker3"]
GS_LABELS = {
    "orf_prediction": "ORF prediction",
    "tiberius":       "Tiberius",
    "merged":         "Merged",
    "braker3":        "BRAKER3",
}
GS_MARKERS = {"orf_prediction": "o", "tiberius": "s", "merged": "^", "braker3": "D"}

LEVELS = ["gene", "transcript", "exon"]
LEVEL_LABELS = {"gene": "Gene", "transcript": "Transcript", "exon": "Exon"}
LEVEL_COLORS = {"gene": "#e41a1c", "transcript": "#377eb8", "exon": "#4daf4a"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate ORF predictions across species/gene-sets.")
    ap.add_argument("--pred-dir", type=Path, required=True,
                    help="Root dir containing <species>/prediction.gtf subdirs.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for table, figure, and gffcompare runs.")
    ap.add_argument("--species", nargs="*", default=None,
                    help="Explicit list of species (default: all with prediction.gtf).")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Species discovery
# ---------------------------------------------------------------------------
def _discover_species(pred_dir: Path, subset: list[str] | None) -> list[str]:
    found = sorted(
        d.name for d in pred_dir.iterdir()
        if d.is_dir() and (d / "prediction.gtf").exists()
    )
    if not subset:
        return found
    missing = set(subset) - set(found)
    if missing:
        print(f"[warn] not found in pred-dir: {sorted(missing)}", flush=True)
    return [s for s in subset if s in set(found)]


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------
def _run_merge(tib_gtf: Path, pred_gtf: Path, out_gtf: Path) -> bool:
    cmd = [sys.executable, str(MERGE_SCRIPT), "--mode", "full",
           str(tib_gtf), str(pred_gtf)]
    try:
        r = subprocess.run(cmd, check=True, capture_output=True)
        out_gtf.write_bytes(r.stdout)
        return out_gtf.stat().st_size > 0
    except subprocess.CalledProcessError as e:
        print(f"    [warn] merge failed: {e.stderr.decode()[:300]}", flush=True)
        return False


# ---------------------------------------------------------------------------
# gffcompare
# ---------------------------------------------------------------------------
def _run_gffcompare(ref: Path, query: Path, prefix: Path) -> Path:
    cmd = [
        "gffcompare", "--strict-match", "-e", "3", "-T",
        "-r", str(ref), "-o", str(prefix), str(query),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return Path(str(prefix) + ".stats")


def _parse_stats(stats_path: Path) -> dict[str, tuple[float, float]]:
    """Return {level: (sensitivity, precision)} for gene/transcript/exon."""
    text = stats_path.read_text()
    patterns = {
        "exon":       r"Exon level:\s+([\d.]+)\s*\|\s*([\d.]+)",
        "transcript": r"Transcript level:\s+([\d.]+)\s*\|\s*([\d.]+)",
        "gene":       r"Locus level:\s+([\d.]+)\s*\|\s*([\d.]+)",
    }
    result: dict[str, tuple[float, float]] = {}
    for level, pat in patterns.items():
        m = re.search(pat, text)
        result[level] = (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))
    return result


def _f1(s: float, p: float) -> float:
    if s != s or p != p or s + p == 0:
        return float("nan")
    return 2 * s * p / (s + p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.out_dir / "gffcompare_runs"
    work_root.mkdir(exist_ok=True)

    species_list = _discover_species(args.pred_dir, args.species)
    if not species_list:
        sys.exit("No species with prediction.gtf found — check --pred-dir.")
    print(f"Species ({len(species_list)}): {species_list}", flush=True)

    table_rows: list[dict] = []

    for sp in species_list:
        print(f"\n=== {sp} ===", flush=True)
        ref      = Path(REF_TMPL.format(sp=sp))
        pred_gtf = args.pred_dir / sp / "prediction.gtf"
        tib_gtf  = Path(TIB_TMPL.format(sp=sp))
        tib_merge_gtf = Path(TIB_MERGE_TMPL.format(sp=sp))
        brk_gtf  = Path(BRK_TMPL.format(sp=sp))

        if not ref.exists():
            print(f"  [skip] missing reference: {ref}", flush=True)
            continue

        sp_work = work_root / sp
        sp_work.mkdir(exist_ok=True)

        # resolve gene set paths
        gene_set_gtfs: dict[str, Path] = {}

        if pred_gtf.exists():
            gene_set_gtfs["orf_prediction"] = pred_gtf
        else:
            print(f"  [warn] missing prediction.gtf: {pred_gtf}", flush=True)

        if tib_gtf.exists():
            gene_set_gtfs["tiberius"] = tib_gtf
        else:
            print(f"  [warn] missing tiberius GTF: {tib_gtf}", flush=True)

        if brk_gtf.exists():
            gene_set_gtfs["braker3"] = brk_gtf
        else:
            print(f"  [warn] missing braker3 GTF: {brk_gtf}", flush=True)

        # build merged
        merged_gtf = sp_work / "merged.gtf"
        if tib_merge_gtf.exists() and pred_gtf.exists():
            print("  merging tiberius + orf_prediction", flush=True)
            if _run_merge(tib_merge_gtf, pred_gtf, merged_gtf):
                gene_set_gtfs["merged"] = merged_gtf
            else:
                print("  [warn] merge produced empty output — skipping merged set", flush=True)
        else:
            missing_m = [str(p) for p in (tib_merge_gtf, pred_gtf) if not p.exists()]
            print(f"  [warn] skipping merge — missing: {missing_m}", flush=True)

        # run gffcompare for each available gene set
        for gs in GENE_SETS:
            gtf = gene_set_gtfs.get(gs)
            if gtf is None:
                continue
            print(f"  gffcompare: {gs}", flush=True)
            try:
                stats_path = _run_gffcompare(ref, gtf, sp_work / gs)
                metrics = _parse_stats(stats_path)
            except subprocess.CalledProcessError as e:
                print(f"  [warn] gffcompare failed ({gs}): {e}", flush=True)
                continue

            row: dict = {"species": sp, "gene_set": gs}
            for level in ("gene", "transcript", "exon"):
                s, p = metrics[level]
                row[f"{level}_S"]  = round(s, 2)
                row[f"{level}_P"]  = round(p, 2)
                row[f"{level}_F1"] = round(_f1(s, p), 2)
            table_rows.append(row)

    if not table_rows:
        sys.exit("No results produced — check paths and gffcompare installation.")

    # -----------------------------------------------------------------------
    # TSV table
    # -----------------------------------------------------------------------
    cols = [
        "species", "gene_set",
        "gene_S",  "gene_P",  "gene_F1",
        "transcript_S", "transcript_P", "transcript_F1",
        "exon_S",  "exon_P",  "exon_F1",
    ]
    table_path = args.out_dir / "accuracy_table.tsv"
    with open(table_path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in table_rows:
            fh.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"\nTable: {table_path}", flush=True)

    # -----------------------------------------------------------------------
    # Figure: one subplot per species + one for the average.
    # x = Sensitivity, y = Precision (both 0–100 %).
    # Color = accuracy level (gene / transcript / exon), marker = gene set.
    # -----------------------------------------------------------------------
    n_sp    = len(species_list)
    n_plots = n_sp + 1
    ncols   = min(4, n_plots)
    nrows   = math.ceil(n_plots / ncols)

    fig, axes_arr = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4, nrows * 4),
                                 squeeze=False)
    axes_flat = [axes_arr[r][c] for r in range(nrows) for c in range(ncols)]

    def _plot_ax(ax: plt.Axes, rows: list[dict], title: str) -> None:
        for row in rows:
            gs = row["gene_set"]
            for level in LEVELS:
                s = row.get(f"{level}_S", float("nan"))
                p = row.get(f"{level}_P", float("nan"))
                if s != s or p != p:
                    continue
                ax.scatter(p, s,
                           color=LEVEL_COLORS[level], marker=GS_MARKERS[gs],
                           s=80, zorder=3, linewidths=0.5, edgecolors="k")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Precision (%)", fontsize=8)
        ax.set_ylabel("Sensitivity (%)", fontsize=8)
        ax.grid(alpha=0.35, linewidth=0.8)

    for i, sp in enumerate(species_list):
        sp_rows = [r for r in table_rows if r["species"] == sp]
        _plot_ax(axes_flat[i], sp_rows, sp.replace("_", " "))

    # average subplot: mean S and P per (gene_set, level)
    gs_level_vals: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in table_rows:
        gs = row["gene_set"]
        for level in LEVELS:
            s = row.get(f"{level}_S", float("nan"))
            p = row.get(f"{level}_P", float("nan"))
            if s == s and p == p:
                gs_level_vals[gs][level].append((s, p))

    avg_rows: list[dict] = []
    for gs, level_data in gs_level_vals.items():
        avg_row: dict = {"gene_set": gs}
        for level, pts in level_data.items():
            avg_row[f"{level}_S"] = sum(x[0] for x in pts) / len(pts)
            avg_row[f"{level}_P"] = sum(x[1] for x in pts) / len(pts)
        avg_rows.append(avg_row)

    _plot_ax(axes_flat[n_sp], avg_rows, "Average")

    for j in range(n_plots, nrows * ncols):
        axes_flat[j].set_visible(False)

    legend_handles = [
        Patch(facecolor=LEVEL_COLORS[lv], edgecolor="k", linewidth=0.5,
              label=LEVEL_LABELS[lv])
        for lv in LEVELS
    ] + [
        Line2D([0], [0], marker=GS_MARKERS[gs], color="k",
               linestyle="None", markersize=8, label=GS_LABELS[gs])
        for gs in GENE_SETS
    ]
    fig.legend(handles=legend_handles, bbox_to_anchor=(1.01, 0.5),
               loc="center left", frameon=True, fontsize=9,
               title="Level / Gene set", title_fontsize=9)

    fig.suptitle("ORF prediction accuracy (Sensitivity vs Precision)", fontsize=13)
    fig.tight_layout()

    fig_path = args.out_dir / "accuracy_figure.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure: {fig_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
