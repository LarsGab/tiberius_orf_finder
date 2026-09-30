#!/usr/bin/env python3
"""
Convert combined_hints.gff to a JBrowse2-viewable GTF.

Each (locus, chain) entry → one transcript.
Each hint (intron / start_codon / stop_codon) → one CDS feature.
Coordinates are converted to genome space using loci_manifest.tsv.

Usage:
    python hints_to_viz_gtf.py loci_manifest.tsv combined_hints.gff out.gtf
"""
import sys
from collections import defaultdict


def sanitize(s):
    """Make a chain_id safe for use in a GTF attribute string."""
    return s.replace(':', '_').replace(';', '_').replace('"', '_')


def main():
    if len(sys.argv) != 4:
        sys.exit(f'Usage: {sys.argv[0]} manifest.tsv hints.gff out.gtf')

    manifest_path, hints_path, out_path = sys.argv[1:]

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest = {}    # locus_id -> (chr, bed_start, chain_id)
    with open(manifest_path) as fh:
        next(fh)     # header
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5:
                continue
            locus_id, chrom, bed_start, _bed_end, chain_id = parts[:5]
            manifest[locus_id] = (chrom, int(bed_start), chain_id)

    # ── Read and group hints by locus_id ──────────────────────────────────────
    hints_by_locus = defaultdict(list)
    with open(hints_path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 8:
                continue
            hints_by_locus[parts[0]].append(parts)

    # ── Write GTF ─────────────────────────────────────────────────────────────
    n_tx = 0
    with open(out_path, 'w') as fh:
        for locus_id in sorted(hints_by_locus):
            if locus_id not in manifest:
                continue
            chrom, bed_start, chain_id = manifest[locus_id]
            chain_safe = sanitize(chain_id)

            # Convert local 1-based GFF coords → genome 1-based coords
            cds_rows = []
            for row in hints_by_locus[locus_id]:
                try:
                    g_start = int(row[3]) + bed_start
                    g_end   = int(row[4]) + bed_start
                except ValueError:
                    continue
                strand    = row[6] if len(row) > 6 else '.'
                score     = row[5] if len(row) > 5 else '.'
                hint_type = row[2]
                cds_rows.append((g_start, g_end, strand, score, hint_type))

            if not cds_rows:
                continue

            strand   = cds_rows[0][2]
            tx_start = min(r[0] for r in cds_rows)
            tx_end   = max(r[1] for r in cds_rows)

            tx_attr = f'gene_id "{chain_safe}"; transcript_id "{locus_id}";'
            fh.write('\t'.join([
                chrom, 'miniprothint', 'transcript',
                str(tx_start), str(tx_end), '.', strand, '.', tx_attr,
            ]) + '\n')

            for g_start, g_end, strand, score, hint_type in cds_rows:
                cds_attr = (f'gene_id "{chain_safe}"; transcript_id "{locus_id}"; '
                            f'hint_type "{hint_type}";')
                fh.write('\t'.join([
                    chrom, 'miniprothint', 'CDS',
                    str(g_start), str(g_end), score, strand, '0', cds_attr,
                ]) + '\n')

            n_tx += 1

    print(f'[out] {n_tx} transcripts → {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
