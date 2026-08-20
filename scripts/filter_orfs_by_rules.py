"""Apply biological filter rules to orf_features.tsv across all vertebrate test species.

Rules applied (in priority order):
  BYPASS  support_level == fullSupport  →  always keep, skip all rules
  R1      sORF class + n_exons > 1     →  remove
  R2      0 < best_norm_bitscore < 3   →  remove (weak but non-zero protein evidence)
  R3      best_target_coverage < 0.75  →  remove (unless protein_extends_5prime_codons > 0)
  R4      n_exons == 1 AND no protein AND no start/stop hint  →  remove
  R5      sORF_NOUPSTOP AND best_norm_bitscore < 2  →  remove
  R6      n_exons >= 3 AND frac_introns_supported < 0.5  →  remove

Outputs per species (under --annot-tag dir):
  orf_features_kept.tsv     rows passing all rules
  orf_features_filtered.tsv rows removed by at least one rule
  filter_stats.tsv          per-rule counts

Also writes pooled TSVs to --out-dir and generates two PDFs
(kept and filtered sets) using plot_orf_features_by_class.py.

Usage
-----
python scripts/filter_orfs_by_rules.py \\
  --base-dir /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --annot-tag annotate_epoch_74_filt_tpm1cov3len300_lorf \\
  --out-dir   /projects/AI-GUSTUS/tiberius_orf_finder/results/filter_analysis \\
  --plot-script /home/gabriell/tiberius_orf_finder/scripts/plot_orf_features_by_class.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─── rule definitions ──────────────────────────────────────────────────────

RULES: list[tuple[str, str]] = [
    ("BYPASS", "Keep unconditionally: support_level == fullSupport"),
    ("R1",     "sORF class AND n_exons > 1"),
    ("R2",     "0 < best_norm_bitscore < 3  (weak protein evidence)"),
    ("R3",     "best_target_coverage < 0.75 AND protein_extends_5prime_codons == 0"),
    ("R4",     "n_exons == 1 AND no protein AND no start/stop hint"),
    ("R5",     "sORF_NOUPSTOP AND best_norm_bitscore < 2"),
    ("R6",     "n_exons >= 3 AND frac_introns_supported < 0.5"),
]


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["n_exons", "best_norm_bitscore", "best_target_coverage",
                "protein_extends_5prime_codons", "has_protein_support",
                "has_start_hint", "has_stop_hint", "frac_introns_supported"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with columns: bypass, r1..r6, filtered (bool), which_rules (str)."""
    df = _coerce(df.copy())

    bypass = df["support_level"] == "fullSupport"

    sorf = df["lorf_class"].isin(["sORF_UPSTOP", "sORF_NOUPSTOP"])
    r1 = sorf & (df["n_exons"] > 1)

    bitscore = df["best_norm_bitscore"].fillna(0)
    r2 = (bitscore > 0) & (bitscore < 3)

    cov = df["best_target_coverage"].fillna(0)
    ext5 = df["protein_extends_5prime_codons"].fillna(0)
    r3 = (cov < 0.75) & (ext5 == 0)

    r4 = (
        (df["n_exons"] == 1) &
        (df["has_protein_support"].fillna(0) == 0) &
        (df["has_start_hint"].fillna(0) == 0) &
        (df["has_stop_hint"].fillna(0) == 0)
    )

    r5 = (df["lorf_class"] == "sORF_NOUPSTOP") & (bitscore < 2)

    fis = df["frac_introns_supported"].fillna(1.0)
    r6 = (df["n_exons"] >= 3) & (fis < 0.5)

    rule_masks = {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5, "R6": r6}

    any_rule = r1 | r2 | r3 | r4 | r5 | r6
    filtered = any_rule & ~bypass

    which = []
    for i in range(len(df)):
        if bypass.iloc[i]:
            which.append("BYPASS")
        elif filtered.iloc[i]:
            rules_hit = [name for name, mask in rule_masks.items() if mask.iloc[i]]
            which.append("+".join(rules_hit))
        else:
            which.append("")

    df["bypass"] = bypass
    for name, mask in rule_masks.items():
        df[f"rule_{name.lower()}"] = mask
    df["filtered"] = filtered
    df["which_rules"] = which
    return df


# ─── stats helpers ──────────────────────────────────────────────────────────

def rule_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-rule breakdown: total removed, and count per gffcompare class."""
    classes = sorted(df["gffcompare_class"].dropna().unique())
    rows = []
    n_total = len(df)
    n_bypass = df["bypass"].sum()

    for rule_code, desc in RULES:
        if rule_code == "BYPASS":
            mask = df["bypass"]
        else:
            col = f"rule_{rule_code.lower()}"
            mask = df[col] & ~df["bypass"]  # only non-bypass
        n = mask.sum()
        row = {"rule": rule_code, "description": desc, "n_affected": int(n),
               "pct_total": round(100 * n / n_total, 2)}
        for cls in classes:
            row[f"class_{cls}"] = int((mask & (df["gffcompare_class"] == cls)).sum())
        rows.append(row)

    # summary row
    n_filtered = df["filtered"].sum()
    n_kept = n_total - n_filtered
    rows.append({
        "rule": "TOTAL_REMOVED", "description": "Any rule (excl bypass)",
        "n_affected": int(n_filtered),
        "pct_total": round(100 * n_filtered / n_total, 2),
        **{f"class_{cls}": int((df["filtered"] & (df["gffcompare_class"] == cls)).sum())
           for cls in classes}
    })
    rows.append({
        "rule": "TOTAL_KEPT", "description": "Passing (incl bypass)",
        "n_affected": int(n_kept),
        "pct_total": round(100 * n_kept / n_total, 2),
        **{f"class_{cls}": int((~df["filtered"] & (df["gffcompare_class"] == cls)).sum())
           for cls in classes}
    })
    return pd.DataFrame(rows)


def print_stats(stats: pd.DataFrame, species: str):
    print(f"\n{'='*60}")
    print(f"  {species}")
    print(f"{'='*60}")
    for _, row in stats.iterrows():
        print(f"  {row['rule']:<16} {row['n_affected']:>6}  ({row['pct_total']:>5.1f}%)  "
              f"  [{row['description']}]")


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir",    required=True, type=Path)
    p.add_argument("--annot-tag",   default="annotate_epoch_74_filt_tpm1cov3len300_lorf")
    p.add_argument("--out-dir",     required=True, type=Path,
                   help="Directory for pooled TSVs, stats, and PDFs")
    p.add_argument("--plot-script", type=Path, default=None,
                   help="Path to plot_orf_features_by_class.py for automatic PDF generation")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_kept: list[pd.DataFrame] = []
    all_filt: list[pd.DataFrame] = []
    all_stats: list[pd.DataFrame] = []

    for sp_dir in sorted(args.base_dir.iterdir()):
        if not sp_dir.is_dir():
            continue
        tsv = sp_dir / args.annot_tag / "orf_features.tsv"
        if not tsv.exists():
            continue

        df = pd.read_csv(tsv, sep="\t", low_memory=False)
        df["species"] = sp_dir.name
        df = apply_rules(df)

        kept = df[~df["filtered"]].copy()
        filt = df[df["filtered"]].copy()

        # per-species output
        sp_out = sp_dir / args.annot_tag
        kept_cols = [c for c in df.columns if not c.startswith("rule_") and c not in ("bypass", "filtered", "which_rules")]
        kept.drop(columns=[c for c in ["bypass","filtered","which_rules"] + [c for c in df.columns if c.startswith("rule_")]], errors="ignore").to_csv(
            sp_out / "orf_features_kept.tsv", sep="\t", index=False)
        filt.drop(columns=[c for c in ["bypass","filtered"] + [c for c in df.columns if c.startswith("rule_")]], errors="ignore").to_csv(
            sp_out / "orf_features_filtered.tsv", sep="\t", index=False)

        stats = rule_stats(df)
        stats["species"] = sp_dir.name
        stats.to_csv(sp_out / "filter_stats.tsv", sep="\t", index=False)
        print_stats(stats, sp_dir.name)

        all_kept.append(kept)
        all_filt.append(filt)
        all_stats.append(stats)

    # pooled outputs
    drop_cols = ["bypass", "filtered"] + [c for c in all_kept[0].columns if c.startswith("rule_")]
    pooled_kept = pd.concat(all_kept, ignore_index=True)
    pooled_filt = pd.concat(all_filt, ignore_index=True)

    def drop_extra(d): return d.drop(columns=[c for c in drop_cols if c in d.columns], errors="ignore")

    pooled_kept_path = args.out_dir / "orf_features_kept_pooled.tsv"
    pooled_filt_path = args.out_dir / "orf_features_filtered_pooled.tsv"
    drop_extra(pooled_kept).to_csv(pooled_kept_path, sep="\t", index=False)
    drop_extra(pooled_filt).to_csv(pooled_filt_path, sep="\t", index=False)

    pooled_stats = pd.concat(all_stats, ignore_index=True)
    pooled_stats.to_csv(args.out_dir / "filter_stats_pooled.tsv", sep="\t", index=False)

    n_total = len(pooled_kept) + len(pooled_filt)
    print(f"\nPooled: kept {len(pooled_kept):,} / {n_total:,} "
          f"({100*len(pooled_kept)/n_total:.1f}%), "
          f"filtered {len(pooled_filt):,} ({100*len(pooled_filt)/n_total:.1f}%)")

    # class breakdown of filtered set
    print("\nFiltered-set gffcompare class distribution:")
    cls_counts = pooled_filt["gffcompare_class"].value_counts()
    for cls, cnt in cls_counts.items():
        print(f"  {cls}  {cnt:>6}")

    # optionally run plots
    if args.plot_script and args.plot_script.exists():
        for label, tsv_path, pdf_name in [
            ("kept",     pooled_kept_path, "orf_features_kept_by_class.pdf"),
            ("filtered", pooled_filt_path, "orf_features_filtered_by_class.pdf"),
        ]:
            # plot script expects base-dir with species subdirs; for pooled we
            # write a temp single-species dir structure
            tmp_base = args.out_dir / f"_tmp_{label}"
            tmp_sp = tmp_base / "pooled" / args.annot_tag
            tmp_sp.mkdir(parents=True, exist_ok=True)
            (tmp_sp / "orf_features.tsv").write_text(
                open(tsv_path).read()
            )
            pdf_out = args.out_dir / pdf_name
            cmd = [
                sys.executable, str(args.plot_script),
                "--base-dir", str(tmp_base),
                "--annot-tag", args.annot_tag,
                "--out-pdf", str(pdf_out),
            ]
            print(f"\nGenerating {pdf_name} …", flush=True)
            subprocess.run(cmd, check=True)
            print(f"  → {pdf_out}")


if __name__ == "__main__":
    main()
