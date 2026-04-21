// Assemble transcripts from a VARUS BAM with StringTie, then extract the
// transcript FASTA with gffread.

process RUN_STRINGTIE {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/stringtie" }, mode: 'copy', overwrite: true
    cpus params.threads

    input:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff), path(bam)

    output:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff),
              path("stringtie.gtf"),
              path("transcripts.fa"), emit: assembly

    script:
    """
    set -euo pipefail
    samtools sort -@ ${task.cpus} -o sorted.bam ${bam}
    stringtie sorted.bam -o stringtie.gtf -p ${task.cpus}
    gffread -w transcripts.fa -g ${genome} stringtie.gtf
    """

    stub:
    """
    touch stringtie.gtf transcripts.fa
    """
}
