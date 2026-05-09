// Long-read variant of RUN_STRINGTIE: assembles transcripts from a VARUS BAM
// produced by minimap2 splice alignment, using StringTie's `-L` long-read
// mode. Otherwise identical to modules/stringtie.nf — same inputs, same
// (stringtie.gtf, transcripts.fa) outputs — so it slots into the same
// downstream LABEL_TRANSCRIPTS/WRITE_TFRECORD modules.

process RUN_STRINGTIE_LR {
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
    stringtie sorted.bam -L -o stringtie.gtf -p ${task.cpus}
    gffread -w transcripts.fa -g ${genome} stringtie.gtf
    """

    stub:
    """
    touch stringtie.gtf transcripts.fa
    """
}
