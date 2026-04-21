// Chunk labelled transcripts into 9999-nt windows and write one TFRecord
// shard per species.

process WRITE_TFRECORD {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/tfrecord" }, mode: 'copy', overwrite: true
    cpus 1

    input:
        tuple val(species), path(transcripts_fa), path(labels_npz)

    output:
        tuple val(species), path("data.tfrecords"), emit: shard

    script:
    """
    set -euo pipefail
    python3 -m tiberius_orf.data.chunk_tfrecord \\
        --fasta ${transcripts_fa} \\
        --labels ${labels_npz} \\
        --out data.tfrecords \\
        --chunk-len ${params.chunk_len}
    test -s data.tfrecords
    """

    stub:
    """
    touch data.tfrecords
    """
}
