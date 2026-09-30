#!/usr/bin/env python3
"""Join gffcompare class codes from a .tracking file into an orf_features.tsv.

gffcompare v0.12.10 with -T -r produces a .tracking file (NOT .tmap).
Format (tab-separated, one row per reference transcript):
  col 0  TCONS_id
  col 1  XLOC_id
  col 2  ref_transcript_id (gene|tx or -)
  col 3  class_code  (=, j, c, u, ...)
  col 4  per-sample query entries: q1:STRG.x.y|gene_id|n_exons|...

The transcript_id used in orf_features.tsv is extracted from col 4 as the
string after 'q1:' up to the first '|'.

Adds or replaces the gffcompare_class column in orf_features.tsv in-place.

Usage
-----
python scripts/join_tmap_labels.py \\
    --tracking gffcompare/orfs.tracking \\
    --features orf_features.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking", required=True, type=Path,
                    help="gffcompare .tracking file")
    ap.add_argument("--features", required=True, type=Path,
                    help="orf_features.tsv to update in-place")
    args = ap.parse_args(argv)

    # Parse tracking: {transcript_id: class_code}
    # col 3 = class_code; col 4 = "q1:<tid>|..." (may have multiple q entries)
    labels: dict[str, str] = {}
    with open(args.tracking) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 5:
                continue
            class_code = cols[3]
            # col 4 onward: q1:<gene_id>|<transcript_id>|<n_exons>|...
            # Register both IDs as keys: for StringTie ORFs they are identical,
            # but Tiberius emits distinct gene_id (g3) vs transcript_id (g3_t1).
            for field in cols[4:]:
                if field.startswith("q") and ":" in field:
                    parts = field.split(":", 1)[1].split("|")
                    for k in parts[:2]:
                        if k and k != "-":
                            labels[k] = class_code
    print(f"  {len(labels)} tracking entries loaded", flush=True)

    cc = Counter(labels.values())
    for code, n in sorted(cc.items(), key=lambda x: -x[1]):
        print(f"    {code}: {n}", flush=True)

    # Rewrite feature TSV with gffcompare_class column
    LABEL = "gffcompare_class"
    tmp = args.features.with_suffix(".tmp")
    with open(args.features) as fin, open(tmp, "w", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")
        header = next(reader)
        if LABEL in header:
            idx = header.index(LABEL)
            writer.writerow(header)
            for row in reader:
                if len(row) > idx:
                    row[idx] = labels.get(row[0], "NA")
                writer.writerow(row)
        else:
            writer.writerow(header + [LABEL])
            for row in reader:
                writer.writerow(row + [labels.get(row[0], "NA")])
    tmp.replace(args.features)
    print(f"  updated -> {args.features}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
