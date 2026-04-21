// Chunk labelled transcripts into fixed-length windows (--chunk_len, default
// 9999 nt) and write one TFRecord shard per species.  Each chunk is stored as
// an `input` tensor [L,6] (one-hot nt + pad channel) and an `output` tensor
// [L,6] (one-hot label classes: IR, START, E1, E2, E0, STOP).

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
