// Assemble transcripts from a VARUS BAM with StringTie, then extract the
// transcript FASTA with gffread.

process RUN_STRINGTIE {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/stringtie" },
        mode: 'copy', overwrite: true,
        pattern: "{stringtie.gtf,transcripts.fa.gz}"
    cpus params.threads

    input:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff), path(bam)

    output:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff),
              path("stringtie.gtf"),
              path("transcripts.fa"), emit: assembly
        path "transcripts.fa.gz", emit: transcripts_gz

    script:
    """
    set -euo pipefail
    samtools sort -@ ${task.cpus} -o sorted.bam ${bam}
    stringtie sorted.bam -o stringtie.gtf -p ${task.cpus}
    gffread -w transcripts.fa -g ${genome} stringtie.gtf
    gzip -kf transcripts.fa
    """

    stub:
    """
    touch stringtie.gtf transcripts.fa transcripts.fa.gz
    """
}
