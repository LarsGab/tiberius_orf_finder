"""Evaluate 12 LGB-filtered gene sets for ORF/Tiberius/BRAKER3 predictions.

Supports Vertebrata, Fungi, and Embryophyta test species via --kingdom.

Gene sets per species (14 total):
  orfs_full               <results-dir>/<sp>/<annot-tag>/orfs.filtered.gtf
  orfs_lgb3               .../orfs_lgb3_filtered.gtf  (P(not-wrong)>=0.5)
  orfs_lgb3_correct       orfs_lgb3 filtered to lgb_class "correct"
  orfs_lgb3_partial       orfs_lgb3 filtered to lgb_class "partial"
  tib_full                tiberius benchmarking tiberius_seqlen.gtf
  tib_lgb3                <results-dir>/<sp>/tiberius_lgb_filtered/tiberius_lgb_filtered.gtf
  tib_lgb3_correct        tib_lgb3 filtered to lgb_class "correct"
  tib_lgb3_partial        tib_lgb3 filtered to lgb_class "partial"
  merge_full              orfs_full + tib_full (concatenated)
  merge_correct           tib_lgb3_correct + orfs_lgb3_correct
  merge_tib_c_orfs        tib_lgb3_correct + orfs_full
  tib_correct_hint_partial        tib_lgb3 correct + partial where chain introns ⊆ tx introns
  merge_correct_hint_partial_orfs tib_correct_hint_partial + orfs_full
  braker3                 braker3 benchmarking braker3.gtf

Outputs:
  <out-dir>/lgb_comparison_table.tsv
  <out-dir>/lgb_comparison.pdf    -- grouped bar chart (S / P / F1) per species
  <out-dir>/gffcompare_runs/      -- raw gffcompare output kept for inspection
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
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Kingdom configurations
# ---------------------------------------------------------------------------

BENCH   = Path("/home/gabriell/tiberius_benchmarking")
PROJDIR = Path("/projects/AI-GUSTUS/tiberius_orf_finder")

KINGDOM_CONFIGS: dict[str, dict] = {
    "vertebrata": {
        "results_dir": PROJDIR / "results/vertebrates_test",
        "annot_tag":   "annotate_epoch_74_filt_tpm1cov3len300_lorf",
        "bench_group": "Vertebrata",
        "species": [
            "Gallus_gallus",
            "Pristiophorus_japonicus",
            "Takifugu_rubripes",
            "Zootoca_vivipara",
            "Archocentrus_centrarchus",
            "Betta_splendens",
            "Bos_taurus",
            "Delphinapterus_leucas",
            "Homo_sapiens",
        ],
    },
    "fungi": {
        "results_dir": PROJDIR / "results/fungi_test",
        "annot_tag":   "annotate_run006_e300",
        "bench_group": "Fungi",
        "species": [
            "Agaricus_bisporus",
            "Aspergillus_fumigatus",
            "Cryphonectria_parasitica",
            "Parastagonospora_nodorum",
            "Puccinia_striiformis",
            "Punctularia_strigosozonata",
            "Tilletiopsis_washingtonensis",
        ],
    },
    "embryophyta": {
        "results_dir": PROJDIR / "results/training_embryophyta_test_v2",
        "annot_tag":   "annotate_run001_e300",
        "bench_group": "Embryophyta",
        "species": [
            "Arabidopsis_thaliana",
            "Eschscholzia_californica",
            "Freycinetia_multiflora",
            "Medicago_truncatula",
            "Mimulus_guttatus",
            "Urochloa_brizantha",
        ],
    },
}

# ---------------------------------------------------------------------------
# Gene set definitions
# ---------------------------------------------------------------------------

GS_ORDER = [
    "orfs_full",
    "orfs_lgb3",
    "orfs_lgb3_correct",
    "orfs_lgb3_partial",
    "tib_full",
    "tib_lgb3",
    "tib_lgb3_correct",
    "tib_lgb3_partial",
    "hint_rescue",
    "tib_c_plus_rescue",
    "tib_c_plus_rescue_orfs",
    "merge_full",
    "merge_correct",
    "merge_tib_c_orfs",
    "tib_correct_hint_partial",
    "merge_correct_hint_partial_orfs",
    "braker3",
]

GS_LABELS = {
    "orfs_full":              "ORFs\n(full)",
    "orfs_lgb3":              "ORFs\n(lgb3)",
    "orfs_lgb3_correct":      "ORFs\n(lgb3 correct)",
    "orfs_lgb3_partial":      "ORFs\n(lgb3 partial)",
    "tib_full":               "Tib\n(full)",
    "tib_lgb3":               "Tib\n(lgb3)",
    "tib_lgb3_correct":       "Tib\n(lgb3 correct)",
    "tib_lgb3_partial":       "Tib\n(lgb3 partial)",
    "hint_rescue":            "Hint\nrescue",
    "tib_c_plus_rescue":      "Tib-c\n+rescue",
    "tib_c_plus_rescue_orfs": "Tib-c\n+rescue+ORFs",
    "merge_full":             "ORFs+Tib\n(full)",
    "merge_correct":          "Tib-c\n+ORFs-c",
    "merge_tib_c_orfs":       "Tib-c\n+ORFs",
    "tib_correct_hint_partial":        "Tib\n(c+hint-p)",
    "merge_correct_hint_partial_orfs": "Tib(c+hint-p)\n+ORFs",
    "braker3":                "BRAKER3",
}

COLORS = {"S": "#1f77b4", "P": "#ff7f0e", "F1": "#2ca02c"}


# ---------------------------------------------------------------------------
# GTF preparation helpers
# ---------------------------------------------------------------------------

def _filter_by_class(src: Path, cls: str, out: Path, force: bool = False) -> Path | None:
    """Write lines from src containing lgb_class "<cls>" to out."""
    if not force and out.exists() and out.stat().st_size > 0:
        return out
    with src.open() as fin, out.open("w") as fout:
        for line in fin:
            if f'lgb_class "{cls}"' in line:
                fout.write(line)
    return out if out.stat().st_size > 0 else None


def _merge_gtfs(srcs: list[Path], out: Path, force: bool = False) -> Path | None:
    """Concatenate GTF files."""
    if not force and out.exists() and out.stat().st_size > 0:
        return out
    with out.open("wb") as fout:
        for src in srcs:
            with src.open("rb") as fin:
                fout.write(fin.read())
    return out if out.stat().st_size > 0 else None


# ---------------------------------------------------------------------------
# gffcompare helpers
# ---------------------------------------------------------------------------

def _run_gffcompare(ref: Path, query_gtf: Path, prefix: Path,
                    force: bool = False) -> Path | None:
    """Extract CDS from query_gtf, run gffcompare, return .stats Path or None."""
    cds_file = Path(str(prefix) + "_cds.gff")
    if force or not cds_file.exists() or cds_file.stat().st_size == 0:
        subprocess.run(
            f"grep -w CDS {query_gtf} > {cds_file}",
            shell=True, check=False
        )
    if not cds_file.exists() or cds_file.stat().st_size == 0:
        return None
    cmd = [
        "gffcompare", "--strict-match", "-e", "3", "-T",
        "-r", str(ref), "-o", str(prefix), str(cds_file),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"    [warn] gffcompare error: {e.stderr.decode()[:300]}", flush=True)
        return None
    stats = Path(str(prefix) + ".stats")
    return stats if stats.exists() else None


def _parse_stats(stats_path: Path) -> dict[str, tuple[float, float]]:
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
# Per-species evaluation
# ---------------------------------------------------------------------------

def evaluate_species(sp: str, cfg: dict, work_root: Path,
                     force: bool = False) -> list[dict]:
    results_dir: Path = cfg["results_dir"]
    annot_tag: str    = cfg["annot_tag"]
    bench_group: str  = cfg["bench_group"]

    sp_dir  = results_dir / sp
    ref     = sp_dir / "assembly/annot_cds.gff"
    sp_work = work_root / sp
    sp_work.mkdir(exist_ok=True)

    print(f"\n=== {sp} ===", flush=True)

    if not ref.exists():
        print(f"  [skip] missing reference: {ref}", flush=True)
        return []

    # Source paths
    orfs_full_src = sp_dir / annot_tag / "orfs.filtered.gtf"
    orfs_lgb3_src = sp_dir / annot_tag / "orfs_lgb3_filtered.gtf"
    tib_full_src  = BENCH / f"paper/{bench_group}/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf"
    tib_lgb3_src  = sp_dir / "tiberius_lgb_filtered/tiberius_lgb_filtered.gtf"
    braker3_src   = BENCH / f"paper/{bench_group}/{sp}/results/predictions/braker3/braker3.gtf"

    # Build gene-set GTF dict in dependency order
    gtfs: dict[str, Path] = {}

    def _add_src(gs: str, src: Path) -> None:
        if src.exists() and src.stat().st_size > 0:
            gtfs[gs] = src
        else:
            print(f"  [skip] {gs}: {src}", flush=True)

    tib_chp_src        = sp_dir / "tiberius_lgb_filtered/tib_correct_hint_partial.gtf"
    hint_rescue_src    = sp_dir / "hint_rescue/tiberius_hint_rescue.gtf"

    _add_src("orfs_full",                orfs_full_src)
    _add_src("orfs_lgb3",                orfs_lgb3_src)
    _add_src("tib_full",                 tib_full_src)
    _add_src("tib_lgb3",                 tib_lgb3_src)
    _add_src("hint_rescue",              hint_rescue_src)
    _add_src("tib_correct_hint_partial", tib_chp_src)
    _add_src("braker3",                  braker3_src)

    for cls in ("correct", "partial"):
        if "orfs_lgb3" in gtfs:
            out = sp_work / f"orfs_lgb3_{cls}.gtf"
            result = _filter_by_class(gtfs["orfs_lgb3"], cls, out, force=force)
            if result:
                gtfs[f"orfs_lgb3_{cls}"] = result
            else:
                print(f"  [skip] orfs_lgb3_{cls}: empty after filter", flush=True)

        if "tib_lgb3" in gtfs:
            out = sp_work / f"tib_lgb3_{cls}.gtf"
            result = _filter_by_class(gtfs["tib_lgb3"], cls, out, force=force)
            if result:
                gtfs[f"tib_lgb3_{cls}"] = result
            else:
                print(f"  [skip] tib_lgb3_{cls}: empty after filter", flush=True)

    # Merged sets
    if "orfs_full" in gtfs and "tib_full" in gtfs:
        out = sp_work / "merge_full.gtf"
        result = _merge_gtfs([gtfs["orfs_full"], gtfs["tib_full"]], out, force=force)
        if result:
            gtfs["merge_full"] = result

    if "tib_lgb3_correct" in gtfs and "orfs_lgb3_correct" in gtfs:
        out = sp_work / "merge_correct.gtf"
        result = _merge_gtfs([gtfs["tib_lgb3_correct"], gtfs["orfs_lgb3_correct"]], out, force=force)
        if result:
            gtfs["merge_correct"] = result

    if "tib_lgb3_correct" in gtfs and "orfs_full" in gtfs:
        out = sp_work / "merge_tib_c_orfs.gtf"
        result = _merge_gtfs([gtfs["tib_lgb3_correct"], gtfs["orfs_full"]], out, force=force)
        if result:
            gtfs["merge_tib_c_orfs"] = result

    if "tib_correct_hint_partial" in gtfs and "orfs_full" in gtfs:
        out = sp_work / "merge_correct_hint_partial_orfs.gtf"
        result = _merge_gtfs([gtfs["tib_correct_hint_partial"], gtfs["orfs_full"]], out, force=force)
        if result:
            gtfs["merge_correct_hint_partial_orfs"] = result

    # hint-rescue combos: tib_lgb3_correct + rescue, and same + all orfs
    if "tib_lgb3_correct" in gtfs and "hint_rescue" in gtfs:
        out = sp_work / "tib_c_plus_rescue.gtf"
        result = _merge_gtfs([gtfs["tib_lgb3_correct"], gtfs["hint_rescue"]], out, force=force)
        if result:
            gtfs["tib_c_plus_rescue"] = result

    if "tib_lgb3_correct" in gtfs and "hint_rescue" in gtfs and "orfs_full" in gtfs:
        out = sp_work / "tib_c_plus_rescue_orfs.gtf"
        result = _merge_gtfs(
            [gtfs["tib_lgb3_correct"], gtfs["hint_rescue"], gtfs["orfs_full"]],
            out, force=force,
        )
        if result:
            gtfs["tib_c_plus_rescue_orfs"] = result

    # Run gffcompare for each gene set in canonical order
    rows = []
    for gs in GS_ORDER:
        gtf = gtfs.get(gs)
        if gtf is None:
            continue
        print(f"  gffcompare: {gs}", flush=True)
        stats_path = _run_gffcompare(ref, gtf, sp_work / gs, force=force)
        if stats_path is None:
            print(f"    [warn] no stats", flush=True)
            continue
        metrics = _parse_stats(stats_path)
        row: dict = {"species": sp, "gene_set": gs}
        for level in ("gene", "transcript", "exon"):
            s, p = metrics[level]
            row[f"{level}_S"]  = round(s, 2)
            row[f"{level}_P"]  = round(p, 2)
            row[f"{level}_F1"] = round(_f1(s, p), 2)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_species(ax: plt.Axes, rows: list[dict], title: str) -> None:
    gs_data: dict[str, dict] = {gs: {} for gs in GS_ORDER}
    for row in rows:
        gs_data[row["gene_set"]] = row

    x      = np.arange(len(GS_ORDER))
    width  = 0.25

    s_vals  = [gs_data[gs].get("transcript_S",  float("nan")) for gs in GS_ORDER]
    p_vals  = [gs_data[gs].get("transcript_P",  float("nan")) for gs in GS_ORDER]
    f1_vals = [gs_data[gs].get("transcript_F1", float("nan")) for gs in GS_ORDER]

    ax.bar(x - width, s_vals,  width, color=COLORS["S"],  alpha=0.85, label="Sensitivity")
    ax.bar(x,         p_vals,  width, color=COLORS["P"],  alpha=0.85, label="Precision")
    ax.bar(x + width, f1_vals, width, color=COLORS["F1"], alpha=0.85, label="F1")

    ax.set_title(title.replace("_", " "), fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [GS_LABELS[gs] for gs in GS_ORDER],
        fontsize=6, rotation=45, ha="right"
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Value (%)", fontsize=8)
    ax.grid(axis="y", alpha=0.35, linewidth=0.8)


def plot(table_rows: list[dict], species_list: list[str], out_dir: Path,
         title: str = "LGB-filter gene-set comparison — transcript level") -> None:
    n_sp    = len(species_list)
    n_plots = n_sp + 1
    ncols   = 4
    nrows   = math.ceil(n_plots / ncols)

    fig, axes_arr = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.5, nrows * 4.5),
        squeeze=False
    )
    axes_flat = [axes_arr[r][c] for r in range(nrows) for c in range(ncols)]

    for i, sp in enumerate(species_list):
        sp_rows = [r for r in table_rows if r["species"] == sp]
        _plot_species(axes_flat[i], sp_rows, sp)

    # Average panel
    gs_metric_vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in table_rows:
        gs = row["gene_set"]
        for metric in ("transcript_S", "transcript_P", "transcript_F1"):
            v = row.get(metric, float("nan"))
            if v == v:
                gs_metric_vals[gs][metric].append(v)

    avg_rows: list[dict] = []
    for gs in GS_ORDER:
        row: dict = {"gene_set": gs}
        for metric in ("transcript_S", "transcript_P", "transcript_F1"):
            vals = gs_metric_vals[gs][metric]
            row[metric] = sum(vals) / len(vals) if vals else float("nan")
        avg_rows.append(row)

    _plot_species(axes_flat[n_sp], avg_rows, "Average")

    for j in range(n_plots, nrows * ncols):
        axes_flat[j].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["S"],  label="Sensitivity"),
        mpatches.Patch(facecolor=COLORS["P"],  label="Precision"),
        mpatches.Patch(facecolor=COLORS["F1"], label="F1"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=True)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    fig_path = out_dir / "lgb_comparison.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure: {fig_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir",  type=Path, required=True,
                    help="Output directory for table, figure, and gffcompare runs.")
    ap.add_argument("--kingdom",  type=str, default="vertebrata",
                    choices=list(KINGDOM_CONFIGS),
                    help="Kingdom/clade to evaluate (sets default paths & species).")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Override results base dir (default from --kingdom).")
    ap.add_argument("--annot-tag",   type=str, default=None,
                    help="Override ORF annotation subdirectory tag (default from --kingdom).")
    ap.add_argument("--bench-group", type=str, default=None,
                    help="Override benchmarking group name Vertebrata/Fungi/Embryophyta.")
    ap.add_argument("--species",  nargs="*", default=None,
                    help="Species to evaluate (default: all for the kingdom).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute all intermediate GTFs and gffcompare stats, "
                         "ignoring any cached files in gffcompare_runs/.")
    args = ap.parse_args(argv)

    cfg = dict(KINGDOM_CONFIGS[args.kingdom])
    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir
    if args.annot_tag is not None:
        cfg["annot_tag"] = args.annot_tag
    if args.bench_group is not None:
        cfg["bench_group"] = args.bench_group
    if args.species is not None:
        cfg["species"] = args.species

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.out_dir / "gffcompare_runs"
    work_root.mkdir(exist_ok=True)

    print(f"Kingdom:     {args.kingdom}", flush=True)
    print(f"Results dir: {cfg['results_dir']}", flush=True)
    print(f"Annot tag:   {cfg['annot_tag']}", flush=True)
    print(f"Bench group: {cfg['bench_group']}", flush=True)
    print(f"Species:     {cfg['species']}", flush=True)

    all_rows: list[dict] = []
    for sp in cfg["species"]:
        all_rows.extend(evaluate_species(sp, cfg, work_root, force=args.force))

    if not all_rows:
        sys.exit("No results produced — check paths and gffcompare installation.")

    # Write TSV
    cols = [
        "species", "gene_set",
        "gene_S",        "gene_P",        "gene_F1",
        "transcript_S",  "transcript_P",  "transcript_F1",
        "exon_S",        "exon_P",        "exon_F1",
    ]
    table_path = args.out_dir / "lgb_comparison_table.tsv"
    with open(table_path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in all_rows:
            fh.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"\nTable: {table_path}", flush=True)

    # Print summary
    print("\n--- transcript S / P / F1 ---", flush=True)
    print(f"{'species':<30} {'gene_set':<22} {'S':>6} {'P':>6} {'F1':>6}", flush=True)
    for row in all_rows:
        print(
            f"{row['species']:<30} {row['gene_set']:<22}"
            f" {row.get('transcript_S', float('nan')):>6.1f}"
            f" {row.get('transcript_P', float('nan')):>6.1f}"
            f" {row.get('transcript_F1', float('nan')):>6.1f}",
            flush=True
        )

    plot_title = (f"LGB-filter gene-set comparison — transcript level"
                  f" ({args.kingdom.capitalize()})")
    plot(all_rows, cfg["species"], args.out_dir, title=plot_title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
