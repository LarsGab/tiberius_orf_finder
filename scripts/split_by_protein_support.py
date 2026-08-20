"""Split a LORF-annotated ORF GTF by LORF class x miniprot CDS overlap.

An ORF is "protein-supported" if at least one of its CDS segments overlaps
(same contig, same strand) a CDS feature in the miniprot GFF.

Outputs written to --out-dir:
  LORF_UPSTOP.gtf            all LORF_UPSTOP ORFs
  LORF_NOUPSTOP_prot.gtf     LORF_NOUPSTOP with miniprot support
  LORF_NOUPSTOP_noprot.gtf   LORF_NOUPSTOP without miniprot support
  lorf_filtered.gtf          LORF_UPSTOP + LORF_NOUPSTOP_prot  (the combined set)

Coordinates are 1-based inclusive (GTF/GFF standard) throughout.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orfs-gtf",     required=True,  type=Path)
    ap.add_argument("--miniprot-gff", required=True,  type=Path)
    ap.add_argument("--out-dir",      required=True,  type=Path)
    return ap.parse_args()


def _load_miniprot_cds(gff: Path) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Return {(contig, strand): sorted [(start, end), ...]} for all CDS rows."""
    ivs: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    with open(gff) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            ivs[(f[0], f[6])].append((int(f[3]), int(f[4])))
    for key in ivs:
        ivs[key].sort()
    return dict(ivs)


def _has_overlap(
    segments: list[tuple[int, int]],
    ivs: list[tuple[int, int]],
) -> bool:
    """True if any segment overlaps any interval in ivs (both 1-based inclusive)."""
    starts = [s for s, _ in ivs]
    for seg_s, seg_e in segments:
        # Find rightmost interval starting <= seg_e
        idx = bisect.bisect_right(starts, seg_e) - 1
        while idx >= 0:
            iv_s, iv_e = ivs[idx]
            if iv_e < seg_s:
                break
            if iv_s <= seg_e and iv_e >= seg_s:
                return True
            idx -= 1
    return False


def _gtf_attr(col: str, key: str) -> str | None:
    for chunk in col.strip().rstrip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(None, 1)
        if len(parts) == 2 and parts[0] == key:
            return parts[1].strip().strip('"')
    return None


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading miniprot CDS from {args.miniprot_gff} ...", flush=True)
    mp_ivs = _load_miniprot_cds(args.miniprot_gff)
    total_mp = sum(len(v) for v in mp_ivs.values())
    print(f"  {total_mp} CDS intervals across {len(mp_ivs)} (contig, strand) keys",
          flush=True)

    # First pass: collect all CDS lines per transcript + metadata
    tx_lines:    dict[str, list[str]]         = defaultdict(list)
    tx_lorf:     dict[str, str]               = {}
    tx_segments: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)

    print(f"Parsing {args.orfs_gtf} ...", flush=True)
    with open(args.orfs_gtf) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            tid = _gtf_attr(f[8], "transcript_id")
            if tid is None:
                continue
            lc = _gtf_attr(f[8], "lorf_class")
            if lc is None:
                continue
            tx_lines[tid].append(line)
            tx_lorf[tid] = lc
            tx_segments[tid].append((f[0], f[6], int(f[3]), int(f[4])))

    print(f"  {len(tx_lorf)} transcripts", flush=True)

    # Second pass: check protein support per transcript
    tx_prot: dict[str, bool] = {}
    for tid, segs in tx_segments.items():
        contig, strand = segs[0][0], segs[0][1]
        key = (contig, strand)
        if key not in mp_ivs:
            tx_prot[tid] = False
            continue
        intervals_for_key = mp_ivs[key]
        coords = [(s, e) for _, _, s, e in segs]
        tx_prot[tid] = _has_overlap(coords, intervals_for_key)

    # Counts
    upstop_all     = [t for t, lc in tx_lorf.items() if lc == "LORF_UPSTOP"]
    noupstop_prot  = [t for t, lc in tx_lorf.items()
                      if lc == "LORF_NOUPSTOP" and tx_prot[t]]
    noupstop_noprot= [t for t, lc in tx_lorf.items()
                      if lc == "LORF_NOUPSTOP" and not tx_prot[t]]
    sorf_all       = [t for t, lc in tx_lorf.items() if lc == "sORF_UPSTOP"]
    uplorf_all     = [t for t, lc in tx_lorf.items() if lc == "upLORF"]

    print(f"  LORF_UPSTOP:            {len(upstop_all)}", flush=True)
    print(f"  LORF_NOUPSTOP + prot:   {len(noupstop_prot)}", flush=True)
    print(f"  LORF_NOUPSTOP + noprot: {len(noupstop_noprot)}", flush=True)
    print(f"  sORF_UPSTOP:            {len(sorf_all)}", flush=True)
    print(f"  upLORF:                 {len(uplorf_all)}", flush=True)

    def _write(path: Path, tids: list[str]) -> None:
        with open(path, "w") as fh:
            for tid in sorted(tids):
                for ln in tx_lines[tid]:
                    fh.write(ln + "\n")
        print(f"  wrote {len(tids)} tx -> {path}", flush=True)

    _write(args.out_dir / "LORF_UPSTOP.gtf",          upstop_all)
    _write(args.out_dir / "LORF_NOUPSTOP_prot.gtf",   noupstop_prot)
    _write(args.out_dir / "LORF_NOUPSTOP_noprot.gtf", noupstop_noprot)
    _write(args.out_dir / "lorf_filtered.gtf",
           upstop_all + noupstop_prot + sorf_all + uplorf_all)


if __name__ == "__main__":
    main()
