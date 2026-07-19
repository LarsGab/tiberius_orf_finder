"""Discover long-read RNA-seq candidates on ENA for a list of species.

Queries the ENA portal REST API (no key required) for each species and
returns per-species candidate runs. Prefers PacBio Iso-Seq; falls back to
Oxford Nanopore if no PacBio candidate exists.

Outputs
-------
* ``<out-tsv>`` : one row per candidate run, ranked (best first) per species
* ``stdout``    : concise per-species summary

Ranking within a species:
* PacBio Iso-Seq runs are always preferred over any ONT run.
* Within a platform, rank by ``base_count`` descending; ties broken by
  ``read_count`` descending.

Usage::

    python scripts/discover_longread_sra.py \\
        --species Gallus_gallus Bos_taurus ... \\
        --out-tsv data/longread/discovery.tsv \\
        [--top-n 5]
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.parse
import urllib.request
from pathlib import Path


_ENA_ENDPOINT = "https://www.ebi.ac.uk/ena/portal/api/search"

_FIELDS = [
    "run_accession", "sample_accession", "study_accession",
    "scientific_name", "tax_id",
    "instrument_platform", "instrument_model",
    "library_strategy", "library_source", "library_selection", "library_layout",
    "base_count", "read_count",
    "study_title", "sample_title", "experiment_title",
]

# Query all long-read reads per platform; classify as RNA-seq downstream
# using library_source, library_strategy, and title text. Some Iso-Seq
# runs are filed with library_source="OTHER" or unset, so a strict
# source-only filter misses them.
_PACBIO_QUERY = 'instrument_platform="PACBIO_SMRT"'
_ONT_QUERY    = 'instrument_platform="OXFORD_NANOPORE"'

_RNA_STRATEGIES = {"RNA-Seq", "ncRNA-Seq", "FL-cDNA", "ssRNA-seq",
                   "miRNA-Seq"}
_RNA_SOURCES    = {"TRANSCRIPTOMIC", "TRANSCRIPTOMIC_SINGLE_CELL"}


def _looks_like_rnaseq(r: dict) -> bool:
    if r.get("library_source", "") in _RNA_SOURCES:
        return True
    if r.get("library_strategy", "") in _RNA_STRATEGIES:
        return True
    txt = " ".join(str(r.get(k, "")) for k in
                   ("experiment_title", "study_title", "sample_title",
                    "library_selection")).lower()
    if any(k in txt for k in ("iso-seq", "isoseq", "rna-seq", "rnaseq",
                              "transcriptome", "cdna", "mrna", "full-length")):
        return True
    return False


def _ena_query(scientific_name: str, extra_query: str) -> list[dict]:
    """Return list of run dicts for scientific_name matching extra_query."""
    q = f'scientific_name="{scientific_name}" AND ({extra_query})'
    params = {
        "result":   "read_run",
        "query":    q,
        "fields":   ",".join(_FIELDS),
        "format":   "tsv",
        "limit":    "500",
    }
    url = _ENA_ENDPOINT + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except Exception as e:
        print(f"[warn] {scientific_name}: ENA query failed: {e}", flush=True)
        return []
    lines = [l for l in body.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    rows: list[dict] = []
    for line in lines[1:]:
        vals = line.split("\t")
        if len(vals) != len(header):
            continue
        row = dict(zip(header, vals))
        rows.append(row)
    return rows


def _rank(rows: list[dict]) -> list[dict]:
    def _key(r):
        try:
            bc = int(r.get("base_count") or 0)
        except ValueError:
            bc = 0
        try:
            rc = int(r.get("read_count") or 0)
        except ValueError:
            rc = 0
        return (-bc, -rc)
    return sorted(rows, key=_key)


def _looks_like_isoseq(r: dict) -> bool:
    """PacBio Iso-Seq indicators."""
    txt = " ".join(str(r.get(k, "")) for k in
                   ("library_selection", "library_strategy",
                    "experiment_title", "study_title", "sample_title",
                    "instrument_model")).lower()
    if "iso-seq" in txt or "isoseq" in txt:
        return True
    if "ccs" in txt or "hifi" in txt:
        return True
    # Fallback: if it's PACBIO_SMRT + TRANSCRIPTOMIC library_source, count it
    return True


def _looks_like_directRNA(r: dict) -> bool:
    txt = str(r.get("library_selection", "")).lower()
    return "cdna" not in txt and "randompcr" not in txt


def _species_scan(sp: str, top_n: int) -> tuple[list[dict], str]:
    """Return (candidates, chosen_platform)."""
    sci = sp.replace("_", " ")
    pacbio = [r for r in _ena_query(sci, _PACBIO_QUERY) if _looks_like_rnaseq(r)]
    if pacbio:
        return _rank(pacbio)[:top_n], "PACBIO_SMRT"
    ont = [r for r in _ena_query(sci, _ONT_QUERY) if _looks_like_rnaseq(r)]
    if ont:
        return _rank(ont)[:top_n], "OXFORD_NANOPORE"
    return [], "none"


def _human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--species", nargs="+", required=True,
                    help="Underscore-separated scientific names.")
    ap.add_argument("--out-tsv", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=5,
                    help="How many top candidate runs to keep per species.")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    all_rows: list[dict] = []
    summary: list[dict] = []

    for sp in args.species:
        rows, platform = _species_scan(sp, args.top_n)
        for r in rows:
            r["species_input"] = sp
            r["chosen_platform"] = platform
        all_rows.extend(rows)

        total_bc = 0
        for r in rows:
            try:
                total_bc += int(r.get("base_count") or 0)
            except ValueError:
                pass
        summary.append({
            "species": sp, "platform": platform,
            "n_runs": len(rows), "top_bases": total_bc,
        })

    if not all_rows:
        print("No runs found.", flush=True)
        return 1

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    header = ["species_input", "chosen_platform", *_FIELDS]
    with args.out_tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"[discovery] wrote {args.out_tsv}", flush=True)

    # Stdout summary.
    print()
    print(f"{'species':<26} {'platform':<20} {'#runs':>6} {'total top bases':>18}")
    for s in summary:
        print(f"{s['species']:<26} {s['platform']:<20} "
              f"{s['n_runs']:>6d} {_human(s['top_bases']):>18}")

    # Per-species top-3 detail.
    print("\n--- top-3 candidates per species ---")
    by_sp: dict[str, list[dict]] = {}
    for r in all_rows:
        by_sp.setdefault(r["species_input"], []).append(r)
    for sp, rows in by_sp.items():
        print(f"\n{sp}  ({rows[0]['chosen_platform']}):")
        for r in rows[:3]:
            print(f"  {r['run_accession']:<12} "
                  f"model={r.get('instrument_model',''):<24} "
                  f"strat={r.get('library_strategy',''):<10} "
                  f"sel={r.get('library_selection',''):<10} "
                  f"bases={_human(int(r.get('base_count') or 0)):>8}  "
                  f"{r.get('experiment_title','')[:60]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
