// VARUS v2 long-read processes for the Tiberius ORF finder pipeline.
//
// These wrap the new Python `varus` CLI (Gaius-Augustus/VARUS v2) in
// --longreads mode: PacBio Iso-Seq / ONT cDNA-Direct-RNA RNA-seq are
// downloaded from NCBI SRA, aligned with minimap2 (splice preset), and
// distilled into a single per-species VARUS.bam.
//
// Required tools on $PATH inside the conda/micromamba env:
//   minimap2, samtools, fastq-dump
//
// The new Python VARUS v2 CLI is invoked as `python -m varus.cli` (set via
// params.varus_cmd) so it resolves to the package installed in the active
// env's site-packages, regardless of whether an older `varus` binary is
// shadowing it on PATH (e.g. /home/gabriell/programs/VARUS, which is the
// legacy Perl tool, must NOT be used here).
//
// Three-stage pipeline (mirrors VARUS' own nextflow/varus.nf):
//   VARUS_RUNLIST_LR -> Runlist.tsv  (NCBI Entrez query, --longreads)
//   VARUS_INDEX_LR   -> mm2idx.mmi   (minimap2 splice index)
//   VARUS_RUN_LR     -> VARUS.bam    (online sampling loop, minimap2)
//
// Inputs/outputs carry the (species, accession, annotation, genome, ref_gff)
// tuple unchanged so they slot straight into the existing STRINGTIE / LABEL /
// TFRECORD downstream modules.

process VARUS_RUNLIST_LR {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/varus" }, mode: 'copy', overwrite: true
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
    def varusCmd = params.varus_cmd ?: 'python -m varus.cli'
    """
    set -euo pipefail
    ${varusCmd} runlist '${species}' \\
        --outdir . \\
        --max-runs ${maxRuns} \\
        --longreads \\
        ${emailArg} ${keyArg}
    test -s Runlist.tsv || { echo "Runlist.tsv is empty for '${species}' (no long-read SRA runs)" >&2; exit 2; }
    """

    stub:
    """
    printf '@Run_acc\\ttotal_spots\\ttotal_bases\\tavg_len\\tbool:paired\\tcolor_space\\tplatform\\nSRR000001\\t1000\\t1000000\\t1000.0\\t0\\t0\\tPACBIO_SMRT\\n' > Runlist.tsv
    """
}


process VARUS_INDEX_LR {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/varus" }, mode: 'copy', overwrite: true
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
        --prefix mm2idx \\
        --longreads
    test -s genome_index/mm2idx.mmi
    """

    stub:
    """
    mkdir -p genome_index
    touch genome_index/mm2idx.mmi
    """
}


process VARUS_RUN_LR {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/varus" }, mode: 'copy', overwrite: true
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
    // Long-read SRA runs typically have far fewer spots; default lower.
    def batchSize  = params.varus_batch_size  ?: 2000
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
        --index ${index_dir}/mm2idx.mmi \\
        --outdir . \\
        --batch-size ${batchSize} \\
        --max-batches ${maxBatches} \\
        --tile-size ${tileSize} \\
        --min-uniq-pct ${minUniqPct} \\
        --threads ${task.cpus} \\
        --seed ${seed} \\
        --longreads \\
        ${bootstrap} ${profitCond} ${pipelineDl}

    test -s VARUS.bam || { echo "VARUS run produced no BAM" >&2; exit 2; }
    """

    stub:
    """
    touch VARUS.bam runtime.varus.txt introns.gff Coverage.csv RunStatistics.csv
    """
}
