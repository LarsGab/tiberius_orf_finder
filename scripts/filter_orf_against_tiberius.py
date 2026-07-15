"""Drop ORF-finder predictions whose CDS is a sub-sequence of some
Tiberius isoform's CDS on the same contig and strand.

Rationale: after merging with Tiberius, we want to keep the ORF prediction
only when it adds coding sequence that Tiberius does not already have.
When Tiberius has an isoform whose CDS strictly contains the ORF-finder
prediction, the StringTie transcript backing the ORF prediction was
likely incomplete; the Tiberius prediction is more complete and should
survive the merge alone.

Reuses ``is_subsequence`` and ``parse_prediction_gtf`` from
``filter_subsequence_predictions.py``. Tolerance flags mirror that
script; defaults are strict.

Usage::

    python scripts/filter_orf_against_tiberius.py \\
        --orf-gtf      orfs.filtered.gtf \\
        --tiberius-gtf tiberius.protein_filt.gtf \\
        --out-gtf      orfs.filtered_vs_tib.gtf \\
        --report-tsv   dropped_vs_tib.tsv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_SUBSEQ_PATH = _HERE / "filter_subsequence_predictions.py"


def _load_subseq():
    spec = importlib.util.spec_from_file_location(
        "_filter_subseq", _SUBSEQ_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_filter_subseq"] = mod
    spec.loader.exec_module(mod)
    return mod


_fs = _load_subseq()
parse_prediction_gtf = _fs.parse_prediction_gtf
is_subsequence = _fs.is_subsequence


def build_tib_index(
    tib_by_tid: dict[str, dict],
) -> dict[tuple[str, str], list[tuple[int, int, str, list[tuple[int, int]]]]]:
    """Bucket Tiberius records by (contig, strand). Each bucket is a list
    of (span_start, span_end, tid, exons) sorted by span_start."""
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for tid, rec in tib_by_tid.items():
        exons = rec["exons"]
        if not exons:
            continue
        buckets[(rec["contig"], rec["strand"])].append(
            (exons[0][0], exons[-1][1], tid, exons)
        )
    for lst in buckets.values():
        lst.sort()
    return buckets


def find_dropable_vs_tib(
    orf_by_tid: dict[str, dict],
    tib_by_tid: dict[str, dict],
    terminal_overhang_nt: int = 0,
    splice_shift_nt: int = 0,
    allow_exon_skip: bool = False,
) -> dict[str, str]:
    """Return {dropped_orf_tid: keeper_tib_tid} for ORF preds that are
    a (near-)subsequence of some Tiberius isoform on the same
    (contig, strand)."""
    tib_buckets = build_tib_index(tib_by_tid)
    drop: dict[str, str] = {}
    for orf_tid, orf_rec in orf_by_tid.items():
        exons = orf_rec["exons"]
        if not exons:
            continue
        key = (orf_rec["contig"], orf_rec["strand"])
        candidates = tib_buckets.get(key)
        if not candidates:
            continue
        a_span_s, a_span_e = exons[0][0], exons[-1][1]
        for b_span_s, b_span_e, b_tid, b_exons in candidates:
            if b_span_s > a_span_e:
                break  # bucket sorted by span_start; no more overlaps
            if b_span_e < a_span_s:
                continue
            if is_subsequence(
                exons, b_exons,
                terminal_overhang_nt=terminal_overhang_nt,
                splice_shift_nt=splice_shift_nt,
                allow_exon_skip=allow_exon_skip,
            ):
                drop[orf_tid] = b_tid
                break
    return drop


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--orf-gtf", type=Path, required=True)
    ap.add_argument("--tiberius-gtf", type=Path, required=True)
    ap.add_argument("--out-gtf", type=Path, required=True)
    ap.add_argument("--report-tsv", type=Path, required=True)
    ap.add_argument("--terminal-overhang-nt", type=int, default=0)
    ap.add_argument("--splice-shift-nt", type=int, default=0)
    ap.add_argument("--allow-exon-skip", action="store_true")
    args = ap.parse_args(argv)

    orf = parse_prediction_gtf(args.orf_gtf)
    tib = parse_prediction_gtf(args.tiberius_gtf)

    drop = find_dropable_vs_tib(
        orf, tib,
        terminal_overhang_nt=args.terminal_overhang_nt,
        splice_shift_nt=args.splice_shift_nt,
        allow_exon_skip=args.allow_exon_skip,
    )

    args.out_gtf.parent.mkdir(parents=True, exist_ok=True)
    args.report_tsv.parent.mkdir(parents=True, exist_ok=True)

    n_kept_tx = 0
    n_kept_cds = 0
    n_drop_cds = 0
    with args.out_gtf.open("w", encoding="utf-8") as fh:
        for tid, rec in orf.items():
            if tid in drop:
                n_drop_cds += len(rec["lines"])
                continue
            n_kept_tx += 1
            for ln in rec["lines"]:
                fh.write(ln + "\n")
                n_kept_cds += 1

    with args.report_tsv.open("w", encoding="utf-8") as fh:
        fh.write("dropped_orf_tid\tkeeper_tib_tid\treason\n")
        for orf_tid, tib_tid in sorted(drop.items()):
            fh.write(f"{orf_tid}\t{tib_tid}\tsub_sequence_of_tiberius\n")

    n_in = len(orf)
    n_drop = len(drop)
    pct = 100.0 * n_drop / max(1, n_in)
    print(f"[filter_orf_vs_tiberius] tolerances     : "
          f"overhang={args.terminal_overhang_nt}nt "
          f"splice_shift={args.splice_shift_nt}nt "
          f"allow_exon_skip={args.allow_exon_skip}")
    print(f"[filter_orf_vs_tiberius] orf input tx   : {n_in}")
    print(f"[filter_orf_vs_tiberius] tib input tx   : {len(tib)}")
    print(f"[filter_orf_vs_tiberius] dropped tx     : {n_drop} ({pct:.1f}%)")
    print(f"[filter_orf_vs_tiberius] kept tx        : {n_kept_tx}")
    print(f"[filter_orf_vs_tiberius] CDS kept/drop  : {n_kept_cds} / {n_drop_cds}")
    print(f"[filter_orf_vs_tiberius] out gtf        : {args.out_gtf}")
    print(f"[filter_orf_vs_tiberius] report tsv     : {args.report_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
