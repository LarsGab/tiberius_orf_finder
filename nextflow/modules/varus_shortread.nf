// VARUS v2 short-read processes for the Tiberius ORF finder pipeline.
//
// Wraps the new Python `varus` CLI (Gaius-Augustus/VARUS v2) for Illumina
// short-read RNA-seq: SRA runs are downloaded, aligned with HISAT2, and
// distilled into a single per-species VARUS.bam.
//
// Tools on $PATH inside the env (`orffinder`):
//   hisat2, samtools, fastq-dump, varus (python -m varus.cli)
//
// Three-stage pipeline (mirror of nextflow/modules/varus_longread.nf,
// minus --longreads):
//   VARUS_RUNLIST_SR -> Runlist.tsv  (NCBI Entrez query)
//   VARUS_INDEX_SR   -> hisatidx.*   (HISAT2 index)
//   VARUS_RUN_SR     -> VARUS.bam    (online sampling + HISAT2 alignment)
//
// Inputs/outputs carry the (species, accession, annotation, genome, ref_gff)
// tuple unchanged so they slot straight into the existing STRINGTIE / LABEL /
// TFRECORD downstream modules.

process VARUS_RUNLIST_SR {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/varus" },
        mode: 'copy', overwrite: true,
        pattern: "Runlist.tsv"
    cpus 1

    input:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff)

    output:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff),
              path("Runlist.tsv")

    script:
    def maxRuns  = params.varus_max_runs ?: 0
    def email    = params.ncbi_email ?: ''
    def apiKey   = params.ncbi_api_key ?: ''
    def emailArg = email  ? "--email ${email}"     : ''
    def keyArg   = apiKey ? "--api-key ${apiKey}"  : ''
    def pairedArg = params.varus_paired_only ? '--paired-only' : ''
    def varusCmd = params.varus_cmd ?: 'python -m varus.cli'
    """
    set -euo pipefail
    ${varusCmd} runlist '${species}' \\
        --outdir . \\
        --max-runs ${maxRuns} \\
        ${pairedArg} \\
        ${emailArg} ${keyArg}
    test -s Runlist.tsv || { echo "Runlist.tsv is empty for '${species}' (no short-read SRA runs)" >&2; exit 2; }
    """

    stub:
    """
    printf '@Run_acc\\ttotal_spots\\ttotal_bases\\tavg_len\\tbool:paired\\tcolor_space\\tplatform\\nSRR000001\\t1000\\t1000000\\t100.0\\t1\\t0\\tILLUMINA\\n' > Runlist.tsv
    """
}


process VARUS_INDEX_SR {
    tag { species }
    cpus { params.varus_index_cpus ?: 8 }

    input:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff), path(runlist)

    output:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff), path(runlist),
              path("genome_index")

    script:
    def varusCmd = params.varus_cmd ?: 'python -m varus.cli'
    """
    set -euo pipefail
    mkdir -p genome_index
    ${varusCmd} index ${genome} \\
        --outdir genome_index \\
        --threads ${task.cpus} \\
        --prefix hisatidx
    # HISAT2 index is a set of .ht2 (or .ht2l for genomes >4 Gb) files,
    # typically 8 of them. Sanity-check that at least one .1.* exists.
    test -s genome_index/hisatidx.1.ht2 || test -s genome_index/hisatidx.1.ht2l
    """

    stub:
    """
    mkdir -p genome_index
    touch genome_index/hisatidx.1.ht2
    """
}


process VARUS_RUN_SR {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/varus" },
        mode: 'copy', overwrite: true,
        pattern: "{VARUS.bam,Coverage.csv,RunStatistics.csv,runtime.varus.txt}"
    cpus params.threads

    input:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff), path(runlist),
              path(index_dir)

    output:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff), path("VARUS.bam"), emit: bam
        path "introns.gff",       optional: true,                       emit: introns
        path "Coverage.csv",      optional: true,                       emit: coverage
        path "RunStatistics.csv", optional: true,                       emit: stats
        path "runtime.varus.txt",                                       emit: runtime

    script:
    def maxBatches = params.varus_max_batches ?: 1000
    def batchSize  = params.varus_batch_size  ?: 50000
    def tileSize   = params.varus_tile_size   ?: 5000
    def minUniqPct = params.varus_min_uniq_pct ?: 5.0
    def seed       = params.varus_seed         ?: 1
    def bootstrap  = params.varus_bootstrap_all ? '--bootstrap-all' : ''
    def profitCond = params.varus_profit_condition ? '--profit-condition' : ''
    def pipelineDl = params.varus_pipeline_downloads ? '--pipeline-downloads' : ''
    def varusCmd   = params.varus_cmd ?: 'python -m varus.cli'
    """
    set -euo pipefail
    /usr/bin/time -p -o runtime.varus.txt \\
      ${varusCmd} run '${species}' ${genome} \\
        --runlist ${runlist} \\
        --index ${index_dir}/hisatidx \\
        --outdir . \\
        --batch-size ${batchSize} \\
        --max-batches ${maxBatches} \\
        --tile-size ${tileSize} \\
        --min-uniq-pct ${minUniqPct} \\
        --threads ${task.cpus} \\
        --seed ${seed} \\
        ${bootstrap} ${profitCond} ${pipelineDl}

    test -s VARUS.bam || { echo "VARUS run produced no BAM" >&2; exit 2; }
    """

    stub:
    """
    touch VARUS.bam runtime.varus.txt introns.gff Coverage.csv RunStatistics.csv
    """
}
