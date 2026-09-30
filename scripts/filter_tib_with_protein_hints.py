#!/usr/bin/env python3
"""
Filter Tiberius LGB predictions using a two-tier selection:

  Tier 1  transcripts with lgb_class "correct"
  Tier 2  transcripts with lgb_class "partial" whose CDS-derived introns
          are a superset of at least one protein chain's intron hints
          (i.e. all chain introns are present in the transcript)

Input GTF must carry lgb_class and lgb_prob_correct attributes written by
apply_lgb_model_gtf.py.  chained_hints.gff must be in genome coordinates
with chain_id= attributes (output of chainedHints.py, hint_integration branch).

Usage:
    python filter_tib_with_protein_hints.py \\
        tiberius_lgb_filtered.gtf chained_hints.gff out.gtf
"""
import argparse
import re
import sys
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('tib_gtf',       help='tiberius_lgb_filtered.gtf (lgb_class attr)')
    p.add_argument('chained_hints', help='chained_hints.gff (genome coords, chain_id-tagged)')
    p.add_argument('out_gtf',       help='output GTF')
    return p.parse_args()


def _gtf_attr(attr_col, key):
    m = re.search(r'%s "([^"]+)"' % re.escape(key), attr_col)
    return m.group(1) if m else None


def _gff_attr(attr_col, key):
    m = re.search(r'(?:^|;)' + re.escape(key) + r'=([^;]+)', attr_col)
    return m.group(1) if m else None


def load_chain_introns(chained_hints_gff):
    """
    Returns {(chr, strand): {chain_id: frozenset((start, end), ...)}}
    Only intron features are collected; chains with no introns are omitted.
    """
    raw = defaultdict(lambda: defaultdict(set))
    with open(chained_hints_gff) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 9 or parts[2].lower() != 'intron':
                continue
            chrom  = parts[0]
            strand = parts[6]
            try:
                start = int(parts[3])
                end   = int(parts[4])
            except ValueError:
                continue
            chain_id = (_gff_attr(parts[8], 'chain_id') or _gff_attr(parts[8], 'grp') or '').strip()
            if not chain_id:
                continue
            raw[(chrom, strand)][chain_id].add((start, end))

    return {
        key: {cid: frozenset(intr) for cid, intr in chains.items()}
        for key, chains in raw.items()
    }


def load_tib_transcripts(tib_gtf):
    """
    Returns:
      tx_lines : tid -> [raw lines]
      tx_meta  : tid -> {chr, strand, lgb_class, introns: frozenset}
    """
    tx_lines = defaultdict(list)
    tx_meta  = {}

    with open(tib_gtf) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 9:
                continue
            chrom   = parts[0]
            feature = parts[2]
            strand  = parts[6]
            attrs   = parts[8]
            try:
                start = int(parts[3])
                end   = int(parts[4])
            except ValueError:
                continue

            tid = _gtf_attr(attrs, 'transcript_id')
            if not tid:
                continue

            tx_lines[tid].append(line)

            if tid not in tx_meta:
                tx_meta[tid] = {
                    'chr':       chrom,
                    'strand':    strand,
                    'lgb_class': _gtf_attr(attrs, 'lgb_class') or '',
                    'cds_list':  [],
                }

            if feature == 'CDS':
                tx_meta[tid]['cds_list'].append((start, end))

    # derive introns from sorted consecutive CDS pairs
    for meta in tx_meta.values():
        cds = sorted(meta.pop('cds_list'))
        meta['introns'] = frozenset(
            (cds[i][1] + 1, cds[i + 1][0] - 1)
            for i in range(len(cds) - 1)
        )

    return tx_lines, tx_meta


def main():
    args = parse_args()

    chain_introns     = load_chain_introns(args.chained_hints)
    tx_lines, tx_meta = load_tib_transcripts(args.tib_gtf)

    print(f'[filter] {len(tx_lines)} transcripts loaded', file=sys.stderr)
    print(f'[filter] {sum(len(v) for v in chain_introns.values())} chains indexed '
          f'across {len(chain_introns)} (chr,strand) combinations', file=sys.stderr)

    n_tier1   = 0
    n_tier2   = 0
    n_dropped = 0
    kept_tids = set()

    for tid, meta in tx_meta.items():
        # Tier 1: LGB-classified as correct
        if meta['lgb_class'] == 'correct':
            kept_tids.add(tid)
            n_tier1 += 1
            continue

        # Tier 2: LGB-classified as partial + all introns of some chain present
        if meta['lgb_class'] != 'partial' or not meta['introns']:
            n_dropped += 1
            continue

        key    = (meta['chr'], meta['strand'])
        chains = chain_introns.get(key, {})

        supported = any(
            intr and intr.issubset(meta['introns'])
            for intr in chains.values()
        )

        if supported:
            kept_tids.add(tid)
            n_tier2 += 1
        else:
            n_dropped += 1

    print(f'[filter] tier-1 (lgb_class=correct): {n_tier1}', file=sys.stderr)
    print(f'[filter] tier-2 (lgb_class=partial, hint-supported): {n_tier2}', file=sys.stderr)
    print(f'[filter] dropped: {n_dropped}', file=sys.stderr)

    with open(args.out_gtf, 'w') as fout:
        for tid in kept_tids:
            for line in tx_lines[tid]:
                fout.write(line)

    print(f'[out] {len(kept_tids)} transcripts → {args.out_gtf}', file=sys.stderr)


if __name__ == '__main__':
    main()
