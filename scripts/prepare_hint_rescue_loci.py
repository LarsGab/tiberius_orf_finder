#!/usr/bin/env python3
"""
Prepare a multi-FASTA + combined hints GFF for Tiberius hint-guided rescue.

Input:
  - LGB-classified partial transcripts (tiberius_lgb_partial.gtf)
  - Chain-tagged hints produced by chainedHints.py  (chain_id= attribute)
  - Genome FASTA + FAI index

Logic:
  For each partial transcript locus that has at least one intron hint:
    1. Add flanking, merge overlapping loci.
    2. Find all distinct chains (chain_id values) with intron hints at the locus.
    3. Optionally cap to the --max_chains highest-supported chains.
    4. For each (locus, chain) pair emit one FASTA entry and the chain's local hints.

FASTA entry names are opaque indices (locus_NNNNNN) to avoid special-
character issues.  A manifest TSV maps each index back to (chr, bed_start,
bed_end, chain_id) for coordinate back-conversion after Tiberius.

FASTA extraction uses samtools faidx (must be on PATH).

Output (all in --outdir):
  combined_loci.fa     — multi-FASTA for Tiberius --genome
  combined_hints.gff   — per-entry hints for Tiberius --hints
  loci_manifest.tsv    — locus_id TAB chr TAB bed_start TAB bed_end TAB chain_id
"""
import argparse
import bisect
import subprocess
import sys
from collections import defaultdict


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--partial_gtf',  required=True, help='tiberius_lgb_partial.gtf')
    p.add_argument('--chained_hints',required=True,
                   help='chain_id-tagged hints (output of chainedHints.py)')
    p.add_argument('--genome',       required=True, help='genome.fa (must have .fai)')
    p.add_argument('--outdir',       required=True, help='output directory')
    p.add_argument('--flank',  type=int, default=25000,
                   help='bp to add on each side of each locus (default 25000)')
    p.add_argument('--max_chains', type=int, default=None,
                   help='max chains per locus, ranked by intron count (default: no cap)')
    return p.parse_args()


# ── GFF attribute helpers ─────────────────────────────────────────────────────

def get_attr(attr_col, key):
    """Return value of 'key=' in a GFF9 attribute column, or None."""
    for field in attr_col.split(';'):
        field = field.strip()
        if field.startswith(key + '='):
            return field[len(key)+1:]
    return None


# ── FAI index ─────────────────────────────────────────────────────────────────

def load_chrom_sizes(fai_path):
    sizes = {}
    with open(fai_path) as fh:
        for line in fh:
            parts = line.split('\t')
            if len(parts) >= 2:
                sizes[parts[0]] = int(parts[1])
    return sizes


# ── Partial transcript loader ─────────────────────────────────────────────────

def load_transcripts(gtf_path):
    txs = []
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 9 or parts[2] != 'transcript':
                continue
            try:
                txs.append((parts[0], int(parts[3]) - 1, int(parts[4])))
            except ValueError:
                continue
    return txs


# ── Interval merge ────────────────────────────────────────────────────────────

def merge_intervals(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    merged = [list(ivs[0])]
    for chrom, start, end in ivs[1:]:
        last = merged[-1]
        if chrom == last[0] and start <= last[2]:
            last[2] = max(last[2], end)
        else:
            merged.append([chrom, start, end])
    return merged


# ── Hint index (for locus→chains and chain→hints lookups) ────────────────────

def load_chained_hints(gff_path):
    """
    Parse chain_id-tagged hints.

    Returns
    -------
    intron_index : dict  chrom -> (sorted_starts, max_end_prefix, list_of_hint_dicts)
        For fast "which chains overlap locus?" queries.
    hints_by_chain : dict  (chrom, chain_id) -> list_of_hint_lines
        Raw GFF lines per chain, for writing per-locus hint subsets.
    """
    # Collect intron hints per chromosome (for overlap index)
    introns_by_chrom = defaultdict(list)   # chrom -> [(start0, end0, chain_id)]
    # Collect all hints per (chrom, chain_id)
    hints_by_chain = defaultdict(list)     # (chrom, chain_id) -> [raw_line, ...]

    with open(gff_path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9:
                continue
            chrom, feature = parts[0], parts[2].lower()
            chain_id = get_attr(parts[8], 'chain_id')
            if not chain_id:
                continue
            try:
                start0 = int(parts[3]) - 1   # 0-based
                end0   = int(parts[4])        # 0-based exclusive
            except ValueError:
                continue
            if end0 <= start0:
                continue

            hints_by_chain[(chrom, chain_id)].append(line.rstrip('\n'))
            if feature == 'intron':
                introns_by_chrom[chrom].append((start0, end0, chain_id))

    # Build sorted intron index with max-end prefix for O(log n) overlap queries
    intron_index = {}
    for chrom, ivs in introns_by_chrom.items():
        ivs.sort()
        starts   = [iv[0] for iv in ivs]
        max_end  = []
        cur = 0
        for iv in ivs:
            cur = max(cur, iv[1])
            max_end.append(cur)
        intron_index[chrom] = (starts, max_end, ivs)

    return intron_index, hints_by_chain


def chains_at_locus(intron_index, chrom, start0, end0):
    """Return list of (chain_id, intron_count) for chains overlapping [start0, end0)."""
    if chrom not in intron_index:
        return []
    starts, max_end, ivs = intron_index[chrom]
    # Find rightmost intron whose start < end0
    right = bisect.bisect_left(starts, end0)
    if right == 0 or max_end[right - 1] <= start0:
        return []
    # Collect all overlapping introns
    counts = defaultdict(int)
    for i in range(right):
        if ivs[i][1] > start0:   # end > locus start → overlaps
            counts[ivs[i][2]] += 1
    return sorted(counts.items(), key=lambda x: -x[1])  # descending by count


# ── FASTA extraction ──────────────────────────────────────────────────────────

def fetch_sequence(genome, chrom, start0, end0):
    """
    Extract [start0, end0) (0-based) via samtools faidx.
    Returns the bare sequence string (no header, no newlines).
    """
    fa_start = start0 + 1          # samtools uses 1-based inclusive
    region   = f'{chrom}:{fa_start}-{end0}'
    result   = subprocess.run(
        ['samtools', 'faidx', genome, region],
        capture_output=True, text=True, check=True
    )
    lines = result.stdout.split('\n')
    return ''.join(lines[1:]).replace('\n', '')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    import os
    os.makedirs(args.outdir, exist_ok=True)

    genome_fai = args.genome + '.fai'
    chrom_sizes = load_chrom_sizes(genome_fai)
    print(f'[index]  {len(chrom_sizes)} chromosomes', file=sys.stderr)

    print(f'[load]   loading chain-tagged hints from {args.chained_hints}',
          file=sys.stderr)
    intron_index, hints_by_chain = load_chained_hints(args.chained_hints)
    n_chains_total = len(hints_by_chain)
    print(f'[load]   {n_chains_total} (chrom, chain) pairs', file=sys.stderr)

    transcripts = load_transcripts(args.partial_gtf)
    print(f'[load]   {len(transcripts)} partial transcripts', file=sys.stderr)

    # Filter to transcripts with hint coverage + add flanking
    with_hints = []
    for chrom, start, end in transcripts:
        if chains_at_locus(intron_index, chrom, start, end):
            s = max(0, start - args.flank)
            e = min(chrom_sizes.get(chrom, end + args.flank), end + args.flank)
            with_hints.append((chrom, s, e))
    print(f'[filter] {len(with_hints)} transcripts have intron hint coverage',
          file=sys.stderr)

    merged = merge_intervals(with_hints)
    print(f'[merge]  {len(merged)} loci after merging (flank={args.flank} bp)',
          file=sys.stderr)

    # ── Build per-(locus, chain) records ─────────────────────────────────────
    out_fa    = os.path.join(args.outdir, 'combined_loci.fa')
    out_gff   = os.path.join(args.outdir, 'combined_hints.gff')
    out_mfst  = os.path.join(args.outdir, 'loci_manifest.tsv')

    total_entries = 0
    with open(out_fa,   'w') as fa_fh, \
         open(out_gff,  'w') as gff_fh, \
         open(out_mfst, 'w') as mfst_fh:

        mfst_fh.write('locus_id\tchr\tbed_start\tbed_end\tchain_id\n')

        for locus_idx, (chrom, bed_start, bed_end) in enumerate(merged):
            chains = chains_at_locus(intron_index, chrom, bed_start, bed_end)
            if not chains:
                continue
            if args.max_chains:
                chains = chains[:args.max_chains]

            # Fetch the locus sequence once (reused for all chains at this locus)
            seq = fetch_sequence(args.genome, chrom, bed_start, bed_end)
            if not seq:
                print(f'[warn]   empty sequence for {chrom}:{bed_start}-{bed_end}',
                      file=sys.stderr)
                continue

            for chain_id, n_introns in chains:
                entry_idx = total_entries
                locus_id  = f'locus_{entry_idx:07d}'
                total_entries += 1

                # FASTA: same sequence, unique header per (locus, chain)
                fa_fh.write(f'>{locus_id}\n')
                # Write in 60-char lines
                for i in range(0, len(seq), 60):
                    fa_fh.write(seq[i:i+60] + '\n')

                # Hints: re-coordinate to locus-local 1-based positions
                # genome GFF pos (1-based) → local = pos - bed_start (still 1-based)
                raw_hints = hints_by_chain.get((chrom, chain_id), [])
                for raw_line in raw_hints:
                    parts = raw_line.split('\t')
                    try:
                        g_start = int(parts[3])
                        g_end   = int(parts[4])
                    except (ValueError, IndexError):
                        continue
                    # Keep only hints overlapping the locus
                    if g_start > bed_end or g_end <= bed_start:
                        continue
                    local_start = g_start - bed_start
                    local_end   = g_end   - bed_start
                    # Clamp to [1, locus_len]
                    locus_len = bed_end - bed_start
                    local_start = max(1, local_start)
                    local_end   = min(locus_len, local_end)
                    if local_end <= 0 or local_start > locus_len:
                        continue
                    parts[0] = locus_id
                    parts[3] = str(local_start)
                    parts[4] = str(local_end)
                    gff_fh.write('\t'.join(parts) + '\n')

                # Manifest
                mfst_fh.write(
                    f'{locus_id}\t{chrom}\t{bed_start}\t{bed_end}\t{chain_id}\n'
                )

            if (locus_idx + 1) % 500 == 0:
                print(f'[progress] {locus_idx+1}/{len(merged)} loci processed, '
                      f'{total_entries} entries so far', file=sys.stderr)

    print(f'[out]    {total_entries} (locus, chain) entries', file=sys.stderr)
    print(f'[out]    FASTA  : {out_fa}',  file=sys.stderr)
    print(f'[out]    hints  : {out_gff}', file=sys.stderr)
    print(f'[out]    manifest: {out_mfst}', file=sys.stderr)


if __name__ == '__main__':
    main()
