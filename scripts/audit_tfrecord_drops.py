"""Audit how many StringTie transcripts the TFRecord pipeline dropped.

Given the same test manifest predict.py uses (``species<TAB>tfrecord_path``),
walks each species and reconciles three transcript-id sets per species:

  A. StringTie GTF       — every transcript_id assembled (denominator)
  B. labels.npz keys     — what project_labels kept (curation output)
  C. TFRecord tx_id      — what actually became data (chunked output)

Reports A vs B vs C counts and the drop, plus the per-category breakdown
from ``stats.tsv`` (written by ``label_transcripts.write_outputs``) when
present next to ``labels.npz``.

Path conventions match predict.py's ``_resolve_species_dirs``::

    <tfrecord>     = <species_dir>/tfrecord/<*.tfrecords>
    labels.npz     = <species_dir>/labels/labels.npz
    stats.tsv      = <species_dir>/labels/stats.tsv          (optional)
    stringtie.gtf  = <species_dir>/stringtie/stringtie.gtf

Usage::

    python scripts/audit_tfrecord_drops.py \\
        --test-manifest results/test/tfrecord_manifest_available.tsv \\
        --out-tsv       results/test/tfrecord_drops_audit.tsv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-manifest", type=Path, required=True,
                    help="TSV manifest (species<TAB>...<TAB>tfrecord_path).")
    ap.add_argument("--out-tsv", type=Path, default=None,
                    help="Optional output TSV; per-species + total summary.")
    ap.add_argument("--no-tfrecord", action="store_true",
                    help="Skip reading TFRecords (faster). Only compares the "
                         "StringTie GTF against labels.npz.")
    return ap.parse_args(argv)


def _resolve_paths(tfrec_path: Path) -> dict[str, Path]:
    species_dir = tfrec_path.parent.parent
    return {
        "tfrecord":  tfrec_path,
        "labels":    species_dir / "labels" / "labels.npz",
        "stats":     species_dir / "labels" / "stats.tsv",
        "stringtie": species_dir / "stringtie" / "stringtie.gtf",
    }


def _stringtie_tx_ids(gtf_path: Path) -> set[str]:
    """Collect every transcript_id mentioned in a StringTie GTF."""
    tids: set[str] = set()
    for raw in gtf_path.read_text().splitlines():
        if not raw or raw.startswith("#"):
            continue
        f = raw.split("\t")
        if len(f) < 9:
            continue
        for chunk in f[8].strip().strip(";").split(";"):
            chunk = chunk.strip()
            parts = chunk.split(None, 1)
            if len(parts) == 2 and parts[0] == "transcript_id":
                tids.add(parts[1].strip().strip('"'))
                break
    return tids


def _labels_tx_ids(npz_path: Path) -> set[str]:
    with np.load(npz_path, allow_pickle=False) as npz:
        return set(npz.files)


def _tfrecord_tx_ids(tfrec_path: Path) -> set[str]:
    import tensorflow as tf

    spec = {
        "tx_id":     tf.io.FixedLenFeature([], tf.string),
        "chunk_idx": tf.io.FixedLenFeature([], tf.int64),
    }
    tids: set[str] = set()
    for serialized in tf.data.TFRecordDataset([str(tfrec_path)]):
        parsed = tf.io.parse_single_example(serialized, spec)
        tids.add(parsed["tx_id"].numpy().decode("utf-8"))
    return tids


def _read_stats(stats_path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in stats_path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        cat, n = line.split("\t")
        out[cat] = int(n)
    return out


def _load_manifest(path: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) < 2:
            continue
        rows.append((parts[0], Path(parts[-1])))
    return rows


# Order matches label_transcripts.project_labels categories.
# Current rescue scheme keeps every transcript:
#   * antisense_ir   was dropped_antisense_only
#   * kept_partial   was dropped_not_contained
#   * ref_partial_ir was dropped_ref_partial
# The legacy dropped_* names are kept here so audits over old stats.tsv
# files still render their counts (zero on new runs).
_KEEP_CATEGORIES = (
    "ir_only",
    "kept_single",
    "kept_multi",
    "antisense_ir",
    "kept_partial",
    "ref_partial_ir",
)
_LEGACY_DROP_CATEGORIES = (
    "dropped_antisense_only",
    "dropped_ref_partial",
    "dropped_not_contained",
)
_CATEGORIES = [*_KEEP_CATEGORIES, *_LEGACY_DROP_CATEGORIES]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    rows = _load_manifest(args.test_manifest)
    if not rows:
        sys.exit(f"empty / unreadable manifest: {args.test_manifest}")

    print(f"{len(rows)} species in manifest", flush=True)

    out_cols = [
        "species",
        "stringtie_tx",
        "labels_tx",
        "tfrecord_tx",
        "dropped_at_curation",
        "drop_pct",
        *_CATEGORIES,
    ]

    per_species: list[dict] = []
    cat_totals: dict[str, int] = defaultdict(int)
    grand = defaultdict(int)

    for species, tfrec in rows:
        p = _resolve_paths(tfrec)
        if not p["stringtie"].exists():
            print(f"  [skip] {species}: missing {p['stringtie']}", flush=True)
            continue
        if not p["labels"].exists():
            print(f"  [skip] {species}: missing {p['labels']}", flush=True)
            continue

        st = _stringtie_tx_ids(p["stringtie"])
        lb = _labels_tx_ids(p["labels"])
        tr = (_tfrecord_tx_ids(p["tfrecord"])
              if (not args.no_tfrecord and p["tfrecord"].exists())
              else set())

        dropped = len(st) - len(lb)
        drop_pct = (100.0 * dropped / len(st)) if st else 0.0

        row = {
            "species":            species,
            "stringtie_tx":       len(st),
            "labels_tx":          len(lb),
            "tfrecord_tx":        len(tr) if tr else "-",
            "dropped_at_curation": dropped,
            "drop_pct":           f"{drop_pct:.1f}",
        }

        stats = _read_stats(p["stats"]) if p["stats"].exists() else {}
        for cat in _CATEGORIES:
            row[cat] = stats.get(cat, 0)
            cat_totals[cat] += stats.get(cat, 0)

        # Sanity check: labels keys should equal the sum of every "keep"
        # category if stats.tsv was produced by the same run as labels.npz.
        if stats:
            kept_expected = sum(stats.get(c, 0) for c in _KEEP_CATEGORIES)
            if kept_expected != len(lb):
                row["species"] += " (!stats vs labels mismatch)"

        per_species.append(row)
        grand["stringtie_tx"] += len(st)
        grand["labels_tx"]    += len(lb)
        grand["dropped"]      += dropped
        if tr:
            grand["tfrecord_tx"] += len(tr)

        print(
            f"  {species:35s} "
            f"GTF={len(st):>7d}  labels={len(lb):>7d}  "
            f"tfrec={(len(tr) if tr else 0):>7d}  "
            f"dropped={dropped:>6d} ({drop_pct:4.1f}%)",
            flush=True,
        )

    # ------------------------------------------------------------ totals
    if not per_species:
        sys.exit("no species processed")

    total_st = grand["stringtie_tx"]
    total_drop = grand["dropped"]
    total_pct = 100.0 * total_drop / total_st if total_st else 0.0
    print()
    print(
        f"TOTAL  StringTie tx: {total_st}  "
        f"labels tx: {grand['labels_tx']}  "
        f"TFRecord tx: {grand.get('tfrecord_tx', '-')}  "
        f"dropped: {total_drop} ({total_pct:.1f}%)",
        flush=True,
    )
    if any(cat_totals.values()):
        print("\nCategory totals (across all species):")
        for cat in _CATEGORIES:
            print(f"  {cat:25s} {cat_totals[cat]:>8d}")

    if args.out_tsv is not None:
        args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_tsv.open("w") as fh:
            fh.write("\t".join(out_cols) + "\n")
            for row in per_species:
                fh.write("\t".join(str(row.get(c, "")) for c in out_cols) + "\n")
            total_row = {
                "species":             "TOTAL",
                "stringtie_tx":        total_st,
                "labels_tx":           grand["labels_tx"],
                "tfrecord_tx":         grand.get("tfrecord_tx", "-"),
                "dropped_at_curation": total_drop,
                "drop_pct":            f"{total_pct:.1f}",
                **{cat: cat_totals[cat] for cat in _CATEGORIES},
            }
            fh.write("\t".join(str(total_row.get(c, "")) for c in out_cols) + "\n")
        print(f"\nWrote {args.out_tsv}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
