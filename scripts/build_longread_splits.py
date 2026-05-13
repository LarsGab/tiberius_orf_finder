"""Build train/val/test CSVs for long-read training from a candidate pool.

Inputs:
  - --candidates-csv  output of scripts/discover_longread_insect_species.py
                      (columns: species,annotation_source,accession,in_braker,n_runs,
                      family,order)
  - --short-test-csv  existing short-read test CSV (preserved verbatim as the
                      long-read test set, so val/test metrics are comparable
                      across short-read and long-read models).

Behaviour:
  - Keep only candidates with annotation_source in {RefSeq, BRAKER}.
  - Exclude any species that appears in the short-read test CSV.
  - Group the remaining annotated candidates by family.
  - For each family with >=2 species, pick exactly one species into the val
    set. Continue across families (shuffled with a fixed --seed) until
    --n-val species are chosen OR no more eligible families remain. The
    species left in each family stay in train, so every val species' family
    is also represented in train (same generalisation-distance structure as
    the short-read split).
  - Everything else among the annotated candidates -> train.
  - Test CSV is the input --short-test-csv copied verbatim. Species without
    long-read RNA-seq will be dropped by the Nextflow pipeline at runtime
    (errorStrategy='ignore'); we deliberately keep them so the test set is
    identical to the short-read experiment for like-for-like comparison.

Output: three CSVs with the columns the Nextflow pipeline expects::

    species,accession,annotation

Usage::

    python scripts/build_longread_splits.py \\
        --candidates-csv nextflow/conf/species_longread_candidates.csv \\
        --short-test-csv nextflow/conf/species_test.csv \\
        --out-training   nextflow/conf/species_training_longread.csv \\
        --out-val        nextflow/conf/species_val_longread.csv \\
        --out-test       nextflow/conf/species_test_longread.csv \\
        --n-val 6 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


OUT_FIELDS = ["species", "accession", "annotation"]


def _read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return rows, reader.fieldnames or []


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in OUT_FIELDS})


def _to_split_row(c: dict) -> dict:
    """Project a candidate row onto the (species, accession, annotation) schema."""
    ann = c.get("annotation_source") or ""
    acc = c.get("accession") or ""
    if ann == "BRAKER":
        acc = ""  # BRAKER species are staged by name, not by accession
    return {"species": c["species"], "accession": acc, "annotation": ann}


def build_splits(
    candidates: list[dict],
    test_species: set[str],
    n_val: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Return (train_rows, val_rows) projected to the output schema."""
    annotated = [
        c for c in candidates
        if c.get("annotation_source") in ("RefSeq", "BRAKER")
    ]
    pool = [c for c in annotated if c["species"] not in test_species]

    by_family: dict[str, list[dict]] = defaultdict(list)
    for c in pool:
        fam = c.get("family") or "_unknown"
        by_family[fam].append(c)

    rng = random.Random(seed)
    eligible_families = [
        f for f, sps in by_family.items() if len(sps) >= 2 and f != "_unknown"
    ]
    rng.shuffle(eligible_families)
    # Stable secondary order within each family
    for f in eligible_families:
        rng.shuffle(by_family[f])

    val_species: set[str] = set()
    val_rows: list[dict] = []
    for fam in eligible_families:
        if len(val_rows) >= n_val:
            break
        chosen = by_family[fam][0]
        val_species.add(chosen["species"])
        val_rows.append(_to_split_row(chosen))

    train_rows = [
        _to_split_row(c) for c in pool if c["species"] not in val_species
    ]

    # Stable alphabetical ordering for diffability
    train_rows.sort(key=lambda r: r["species"])
    val_rows.sort(key=lambda r: r["species"])
    return train_rows, val_rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidates-csv", required=True, type=Path)
    ap.add_argument("--short-test-csv", required=True, type=Path,
                    help="Existing short-read test CSV; copied verbatim to --out-test.")
    ap.add_argument("--out-training", required=True, type=Path)
    ap.add_argument("--out-val", required=True, type=Path)
    ap.add_argument("--out-test", required=True, type=Path)
    ap.add_argument("--n-val", type=int, default=6,
                    help="Target number of val species (default: 6).")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducible family shuffling (default: 42).")
    args = ap.parse_args(argv)

    cands, _ = _read_csv(args.candidates_csv)
    test_rows, _ = _read_csv(args.short_test_csv)
    test_species = {r["species"] for r in test_rows}

    train_rows, val_rows = build_splits(
        cands, test_species, args.n_val, args.seed,
    )

    if len(val_rows) < args.n_val:
        print(
            f"warning: only {len(val_rows)} val species filled "
            f"(asked for {args.n_val}); not enough multi-species families in pool",
            file=sys.stderr,
        )

    _write_csv(args.out_training, train_rows)
    _write_csv(args.out_val, val_rows)
    # Test: copy verbatim, keeping only species/accession/annotation columns.
    _write_csv(
        args.out_test,
        [{"species": r["species"],
          "accession": r.get("accession", ""),
          "annotation": r.get("annotation", "")} for r in test_rows],
    )

    # Stdout report
    annotated = [c for c in cands if c.get("annotation_source") in ("RefSeq", "BRAKER")]
    print(f"Annotated candidates:        {len(annotated)}")
    print(f"  excluded (in short-test):  {len(annotated) - len(train_rows) - len(val_rows)}")
    print(f"Train species written:       {len(train_rows)}  -> {args.out_training}")
    print(f"Val species written:         {len(val_rows)}    -> {args.out_val}")
    print(f"Test species (verbatim):     {len(test_rows)}   -> {args.out_test}")
    print()
    train_families = {
        c["family"] for c in cands
        if c["species"] in {r["species"] for r in train_rows} and c.get("family")
    }
    val_family_of = {
        c["species"]: c.get("family") or ""
        for c in cands
        if c["species"] in {r["species"] for r in val_rows}
    }
    print("Val species and their family (family also in train: yes/no):")
    for r in val_rows:
        fam = val_family_of.get(r["species"], "")
        in_train = "yes" if fam in train_families else "no"
        print(f"  {r['species']:<32s} {fam:<20s} family-in-train={in_train}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
