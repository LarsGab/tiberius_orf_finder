#!/usr/bin/env python3
"""Filter an ORF prediction GTF using any combination of feature columns.

Loads orf_features.tsv (from compute_orf_features.py) and optionally joins
score_tiberius.py output, then applies a pandas query expression or a named
preset to determine which transcripts to keep.

Usage examples
--------------
# Named preset (lorf-filtered = GeneMark-ETP style)
python scripts/filter_orfs_by_features.py \\
    --features orf_features.tsv \\
    --gtf      orfs.filtered.gtf \\
    --out      orfs.lorf_filtered.gtf \\
    --preset   lorf_filtered

# Arbitrary pandas query (any column name is valid)
python scripts/filter_orfs_by_features.py \\
    --features orf_features.tsv \\
    --gtf      orfs.filtered.gtf \\
    --out      orfs.custom.gtf \\
    --query    "lorf_class != 'LORF_NOUPSTOP' or (has_protein_support == 1 and best_identity >= 0.8)"

# Combine features + HMM scores (joined on transcript_id)
python scripts/filter_orfs_by_features.py \\
    --features  orf_features.tsv \\
    --hmm       scores.tsv \\
    --gtf       orfs.filtered.gtf \\
    --out       orfs.hmm_filtered.gtf \\
    --query     "start_prob >= 0.5 and support_level != 'noSupport'"

# Show column names and preset definitions without filtering
python scripts/filter_orfs_by_features.py --list-presets

Named presets
-------------
lorf_filtered       LORF_UPSTOP + sORF_UPSTOP + upLORF + LORF_NOUPSTOP with protein
                    support  (GeneMark-ETP style; equivalent to what split_by_protein_
                    support.py produces but using the feature table directly)
full_support        All three boundaries (start, introns, stop) confirmed by miniprothint
any_support         At least one miniprothint boundary hint matches
protein_only        Any miniprot alignment overlaps the ORF (≥30% CDS)
no_conflict         No competing protein chain with a different 5' boundary
no_split_partner    Drop ORFs that look like downstream fragments of split genes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Presets ────────────────────────────────────────────────────────────────────

PRESETS: dict[str, str] = {
    "lorf_filtered": (
        "lorf_class in ['LORF_UPSTOP', 'sORF_UPSTOP', 'upLORF'] "
        "or (lorf_class == 'LORF_NOUPSTOP' and has_protein_support == 1)"
    ),
    "full_support": "support_level == 'fullSupport'",
    "any_support":  "support_level in ['fullSupport', 'anySupport']",
    "protein_only": "has_protein_support == 1",
    "no_conflict":  "has_conflict == 0",
    "no_split_partner": "has_upstream_partner == 0",
}


# ── GTF helpers ────────────────────────────────────────────────────────────────

def _parse_gtf(path: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Return (tid → [lines], comment_lines)."""
    from collections import defaultdict
    import re

    tid_lines: dict[str, list[str]] = defaultdict(list)
    comments: list[str] = []
    for raw in path.read_text().splitlines():
        if not raw or raw.startswith("#"):
            comments.append(raw)
            continue
        f = raw.split("\t")
        if len(f) < 9:
            continue
        m = re.search(r'transcript_id\s+"([^"]+)"', f[8])
        if m:
            tid_lines[m.group(1)].append(raw)
    return dict(tid_lines), comments


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--features", type=Path,
                    help="orf_features.tsv from compute_orf_features.py")
    ap.add_argument("--hmm",      type=Path, default=None,
                    help="scores.tsv from score_tiberius.py (optional, joined on "
                         "transcript_id)")
    ap.add_argument("--gtf",      type=Path,
                    help="Input ORF GTF (orfs.filtered.gtf)")
    ap.add_argument("--out",      type=Path,
                    help="Output filtered GTF")
    ap.add_argument("--query",    default=None,
                    help="Pandas query string (uses column names from feature table)")
    ap.add_argument("--preset",   default=None, choices=list(PRESETS),
                    help="Named filter preset (see --list-presets)")
    ap.add_argument("--list-presets", action="store_true",
                    help="Print preset names and their query strings and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.list_presets:
        width = max(len(k) for k in PRESETS)
        for name, expr in PRESETS.items():
            print(f"  {name:<{width}}  {expr}")
        return 0

    if not args.features or not args.gtf or not args.out:
        print("ERROR: --features, --gtf, and --out are required unless --list-presets",
              file=sys.stderr)
        return 1
    if args.query is None and args.preset is None:
        print("ERROR: one of --query or --preset is required", file=sys.stderr)
        return 1

    # Build query string
    query = args.query if args.query else PRESETS[args.preset]
    print(f"Filter: {query}", file=sys.stderr)

    # Load feature table
    df = pd.read_csv(args.features, sep="\t", low_memory=False)
    print(f"Loaded {len(df):,} transcripts from {args.features.name}", file=sys.stderr)

    # Optionally join HMM scores
    if args.hmm:
        hmm = pd.read_csv(args.hmm, sep="\t", low_memory=False)
        df = df.merge(hmm, on="transcript_id", how="left", suffixes=("", "_hmm"))
        print(f"Joined HMM scores: {len(hmm):,} rows", file=sys.stderr)

    # Apply filter
    try:
        passing = df.query(query)
    except Exception as e:
        print(f"ERROR evaluating query: {e}", file=sys.stderr)
        return 1

    passing_tids = set(passing["transcript_id"])
    n_pass = len(passing_tids)
    n_total = len(df)
    print(f"Passing: {n_pass:,} / {n_total:,}  ({100*n_pass/n_total:.1f}%)",
          file=sys.stderr)

    # Load and filter GTF
    tid_lines, comments = _parse_gtf(args.gtf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.out, "w") as fh:
        for c in comments:
            fh.write(c + "\n")
        for tid in sorted(passing_tids):
            for line in tid_lines.get(tid, []):
                fh.write(line + "\n")
            if tid in tid_lines:
                n_written += 1
    print(f"Wrote {n_written:,} transcripts -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
