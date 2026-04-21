// Project reference CDS onto StringTie transcripts and emit per-position
// labels + a subset FASTA + per-category stats.

process LABEL_TRANSCRIPTS {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/labels" }, mode: 'copy', overwrite: true
    cpus 1

    input:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff),
              path(stringtie_gtf), path(transcripts_fa)

    output:
        tuple val(species),
              path("transcripts_labelled.fa"),
              path("labels.npz"), emit: labelled
        path "stats.tsv",          emit: stats

    script:
    """
    set -euo pipefail
    python3 -m tiberius_orf.data.label_transcripts \\
        --stringtie-gtf ${stringtie_gtf} \\
        --reference-gff ${ref_gff} \\
        --transcripts-fa ${transcripts_fa} \\
        --out-dir .
    test -s labels.npz
    test -s transcripts_labelled.fa
    """

    stub:
    """
    touch transcripts_labelled.fa labels.npz stats.tsv
    """
}
