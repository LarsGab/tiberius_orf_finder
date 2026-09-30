"""Aggregate per-step runtime TSVs into one pivot table for a benchmark report.

Reads any number of runtimes.tsv files (produced by scripts/lib/log_runtime.sh)
and writes:

  <out-dir>/runtimes_long.tsv   — concatenated raw rows
  <out-dir>/runtimes_pivot.tsv  — (species x tool_phase) matrix, seconds
  <out-dir>/runtimes_pivot.md   — same as GitHub-flavored markdown table

Usage:
  python scripts/collect_runtimes.py \\
      --inputs run_a/runtimes.tsv run_b/runtimes.tsv \\
      --out-dir results/vertebrates_test/eval_accuracy_run010_varus/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists() or p.stat().st_size == 0:
            print(f"[warn] skipping missing/empty: {p}")
            continue
        frames.append(pd.read_csv(p, sep="\t"))
    if not frames:
        raise SystemExit("No input runtime TSVs found.")
    df = pd.concat(frames, ignore_index=True)
    # keep only the last row per (species,tool,phase) — later rows override earlier retries
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["species", "tool", "phase"], keep="last"
    )
    return df


def _fmt_hms(sec: float) -> str:
    if pd.isna(sec):
        return ""
    s = int(sec)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["step"] = df["tool"] + ":" + df["phase"]
    pv = df.pivot_table(
        index="species", columns="step", values="seconds",
        aggfunc="sum", fill_value=None,
    )
    pv["total_s"] = pv.sum(axis=1, min_count=1)
    return pv


def _to_markdown(pv: pd.DataFrame) -> str:
    pv_h = pv.copy()
    for c in pv_h.columns:
        pv_h[c] = pv_h[c].map(_fmt_hms)
    pv_h.index.name = "species"
    return pv_h.to_markdown()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True,
                    help="runtimes.tsv files to aggregate")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = _load(args.inputs)
    long_out = args.out_dir / "runtimes_long.tsv"
    df.to_csv(long_out, sep="\t", index=False)
    print(f"[out] {long_out}  ({len(df)} rows)")

    pv = _pivot(df)
    pivot_tsv = args.out_dir / "runtimes_pivot.tsv"
    pv.to_csv(pivot_tsv, sep="\t")
    print(f"[out] {pivot_tsv}  ({pv.shape[0]} species x {pv.shape[1]} steps)")

    pivot_md = args.out_dir / "runtimes_pivot.md"
    pivot_md.write_text(_to_markdown(pv))
    print(f"[out] {pivot_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
