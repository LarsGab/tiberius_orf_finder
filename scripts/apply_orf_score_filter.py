"""Apply a pre-computed ORFfinder score filter to a gene prediction GTF.

Reads a scores TSV produced by score_tiberius.py or filter_by_orf_score.py,
determines passing transcripts by the requested thresholds, then streams the
original GTF writing only lines that belong to passing transcripts.

No model loading or GPU required — just file I/O.

Filter logic
------------
A transcript is KEPT when ALL specified conditions are satisfied:

  mean_coding_prob >= --min-coding   (default 0.0, i.e. no filter)
  start_prob       >= --min-start    (default 0.0)
  min(mean_coding_prob, start_prob) >= --min-combined  (default 0.0)

Set only the flags you want; others default to 0 (no effect).

Typical invocations
-------------------
# Filter A: mean_coding_prob >= 0.85 only
python scripts/apply_orf_score_filter.py \\
  --gtf      tiberius_seqlen.gtf \\
  --scores   score_tiberius_epoch_74_up500/scores.tsv \\
  --out-gtf  tiberius_filtered_coding085.gtf \\
  --min-coding 0.85

# Filter B: min(coding, start) >= 0.766
python scripts/apply_orf_score_filter.py \\
  --gtf      tiberius_seqlen.gtf \\
  --scores   score_tiberius_epoch_74_up500/scores.tsv \\
  --out-gtf  tiberius_filtered_combined0766.gtf \\
  --min-combined 0.766
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# GTF parse
# ---------------------------------------------------------------------------

def _attr(attr_col: str, key: str) -> str | None:
    for chunk in attr_col.strip().strip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(None, 1)
        if len(parts) == 2 and parts[0] == key:
            return parts[1].strip().strip('"')
    return None


def _parse_gtf_structure(gtf_path: Path) -> tuple[dict, dict]:
    """Group GTF lines by transcript_id; collect gene lines separately.

    Returns
    -------
    tid_lines  : {tid: [raw_lines]}
    gene_info  : {gene_id: {line: str|None, tids: [str]}}
    """
    tid_lines: dict = defaultdict(list)
    gene_info: dict = {}

    for raw in Path(gtf_path).read_text().splitlines():
        if not raw or raw.startswith("#"):
            continue
        f = raw.split("\t")
        if len(f) < 9:
            continue
        feature = f[2]
        tid = _attr(f[8], "transcript_id")
        gid = _attr(f[8], "gene_id")

        if feature == "gene" or tid is None:
            if gid:
                if gid not in gene_info:
                    gene_info[gid] = {"line": None, "tids": []}
                gene_info[gid]["line"] = raw
            continue

        tid_lines[tid].append(raw)
        if gid:
            if gid not in gene_info:
                gene_info[gid] = {"line": None, "tids": []}
            if tid not in gene_info[gid]["tids"]:
                gene_info[gid]["tids"].append(tid)

    return dict(tid_lines), gene_info


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def _build_passing_set(
    scores_tsv: Path,
    min_coding:   float,
    min_start:    float,
    min_combined: float,
) -> set[str]:
    df = pd.read_csv(scores_tsv, sep="\t", low_memory=False)
    mask = pd.Series([True] * len(df), index=df.index)
    if min_coding > 0.0:
        mask &= df["mean_coding_prob"].fillna(0.0) >= min_coding
    if min_start > 0.0:
        mask &= df["start_prob"].fillna(0.0) >= min_start
    if min_combined > 0.0:
        combined = df[["mean_coding_prob", "start_prob"]].fillna(0.0).min(axis=1)
        mask &= combined >= min_combined
    return set(df.loc[mask, "transcript_id"])


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _write_filtered(
    out_gtf:      Path,
    passing_tids: set[str],
    tid_lines:    dict,
    gene_info:    dict,
) -> tuple[int, int]:
    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    n_genes = n_tx = 0
    with open(out_gtf, "w") as fh:
        for gid, info in gene_info.items():
            passing_children = [t for t in info["tids"] if t in passing_tids]
            if not passing_children:
                continue
            n_genes += 1
            if info["line"] is not None:
                fh.write(info["line"] + "\n")
            for tid in passing_children:
                for line in tid_lines.get(tid, []):
                    fh.write(line + "\n")
                n_tx += 1
        # transcripts not linked to any gene line
        linked = {t for info in gene_info.values() for t in info["tids"]}
        for tid in sorted(passing_tids - linked):
            for line in tid_lines.get(tid, []):
                fh.write(line + "\n")
            n_tx += 1
    return n_genes, n_tx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Apply pre-computed ORFfinder scores to filter a prediction GTF."
    )
    ap.add_argument("--gtf",          type=Path, required=True,
                    help="Input prediction GTF.")
    ap.add_argument("--scores",       type=Path, required=True,
                    help="Scores TSV from score_tiberius.py.")
    ap.add_argument("--out-gtf",      type=Path, required=True,
                    help="Output filtered GTF.")
    ap.add_argument("--min-coding",   type=float, default=0.0,
                    help="Keep if mean_coding_prob >= this (default 0, disabled).")
    ap.add_argument("--min-start",    type=float, default=0.0,
                    help="Keep if start_prob >= this (default 0, disabled).")
    ap.add_argument("--min-combined", type=float, default=0.0,
                    help="Keep if min(coding, start) >= this (default 0, disabled).")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    # Determine passing transcripts from scores
    passing = _build_passing_set(
        args.scores, args.min_coding, args.min_start, args.min_combined,
    )

    # Load GTF structure
    tid_lines, gene_info = _parse_gtf_structure(args.gtf)
    n_total = sum(1 for info in gene_info.values() for _ in info["tids"])

    n_dropped = len(tid_lines) - len(passing & set(tid_lines))
    print(
        f"[apply_filter] {args.gtf.name}: "
        f"{len(passing):,} pass / {n_dropped:,} dropped / {len(tid_lines):,} total",
        flush=True,
    )

    n_genes, n_tx = _write_filtered(args.out_gtf, passing, tid_lines, gene_info)
    print(f"[apply_filter] -> {args.out_gtf}  ({n_tx:,} tx in {n_genes:,} genes)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
