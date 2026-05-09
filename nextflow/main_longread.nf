#!/usr/bin/env nextflow
/*
 * Per-species long-read training-data generation for the Tiberius ORF finder.
 *
 *   Input : a CSV with columns species,accession,annotation (RefSeq|BRAKER) —
 *           same format as the short-read pipeline, but only the `species`
 *           column is consumed; the assembly is reused from a prior run.
 *
 *   Output: one .tfrecords shard per species under
 *           ${outdir}/<Genus_species>/tfrecord/data.tfrecords plus a
 *           tfrecord_manifest.tsv at the root.
 *
 * Pipeline:
 *   STAGE_ASSEMBLY    (reuse genome + reference annotation from a prior run)
 *   VARUS_RUNLIST_LR  (NCBI Entrez query, --longreads)
 *   VARUS_INDEX_LR    (minimap2 splice index)
 *   VARUS_RUN_LR      (online sampling loop, minimap2 splice alignment)
 *   RUN_STRINGTIE_LR  (samtools sort + stringtie -L + gffread)
 *   LABEL_TRANSCRIPTS (project reference CDS, write labels.npz)
 *   WRITE_TFRECORD    (chunk + tfrecord shard)
 *
 * Example (HPC):
 *   nextflow run nextflow/main_longread.nf \
 *     -c nextflow/conf/brain_longread.config \
 *     --species_csv nextflow/conf/species_training.csv \
 *     --reuse_assembly_dir /home/gabriell/tiberius_orf_finder/results/training \
 *     --outdir /home/gabriell/tiberius_orf_finder/results/training_longread \
 *     -resume
 */

nextflow.enable.dsl = 2

include { STAGE_ASSEMBLY }                                            from './modules/stage_assembly.nf'
include { VARUS_RUNLIST_LR; VARUS_INDEX_LR; VARUS_RUN_LR }            from './modules/varus_longread.nf'
include { RUN_STRINGTIE_LR }                                          from './modules/stringtie_longread.nf'
include { LABEL_TRANSCRIPTS }                                         from './modules/label.nf'
include { WRITE_TFRECORD }                                            from './modules/tfrecord.nf'


// ---------------------------- params ----------------------------

params.species_csv         = params.species_csv         ?: null
params.outdir              = params.outdir              ?: 'results_longread'

params.reuse_assembly_dir  = params.reuse_assembly_dir  ?: null

// VARUS run-time hyperparameters (long-read defaults)
params.varus_max_batches    = (params.containsKey('varus_max_batches')   && params.varus_max_batches   != null ? params.varus_max_batches   : 1000) as int
params.varus_batch_size     = (params.containsKey('varus_batch_size')    && params.varus_batch_size    != null ? params.varus_batch_size    : 2000) as int
params.varus_tile_size      = (params.containsKey('varus_tile_size')     && params.varus_tile_size     != null ? params.varus_tile_size     : 5000) as int
params.varus_min_uniq_pct   = (params.containsKey('varus_min_uniq_pct')  && params.varus_min_uniq_pct  != null ? params.varus_min_uniq_pct  : 5.0) as double
params.varus_max_runs       = (params.containsKey('varus_max_runs')      && params.varus_max_runs      != null ? params.varus_max_runs      : 0) as int
params.varus_seed           = (params.containsKey('varus_seed')          && params.varus_seed          != null ? params.varus_seed          : 1) as int
params.varus_bootstrap_all  = (params.containsKey('varus_bootstrap_all') ? params.varus_bootstrap_all : false) as boolean
params.varus_profit_condition = (params.containsKey('varus_profit_condition') ? params.varus_profit_condition : false) as boolean
params.varus_pipeline_downloads = (params.containsKey('varus_pipeline_downloads') ? params.varus_pipeline_downloads : false) as boolean
params.varus_index_cpus     = (params.containsKey('varus_index_cpus')    && params.varus_index_cpus    != null ? params.varus_index_cpus    : 8) as int

params.ncbi_email           = params.ncbi_email   ?: null
params.ncbi_api_key         = params.ncbi_api_key ?: null

// VARUS v2 entry point. Defaults to `python -m varus.cli` so the active env's
// installed package wins over any (older) `varus` binary on PATH.
params.varus_cmd            = params.varus_cmd ?: 'python -m varus.cli'

params.threads              = (params.threads ?: 8) as int
params.chunk_len            = (params.containsKey('chunk_len') && params.chunk_len != null ? params.chunk_len : 9999) as int


def die(msg) { log.error msg; System.exit(1) }

if (!params.species_csv)        die("Missing --species_csv")
if (!params.reuse_assembly_dir) die("Missing --reuse_assembly_dir (point at a finished short-read pipeline outdir)")


// ---------------------------- workflow ----------------------------

workflow {

    ch_species = Channel.fromPath(params.species_csv, checkIfExists: true)
        .splitCsv(header: true)
        .map { row -> tuple(row.species, row.accession, row.annotation) }

    assembly       = STAGE_ASSEMBLY(ch_species).assembly
    runlist_out    = VARUS_RUNLIST_LR(assembly)
    index_out      = VARUS_INDEX_LR(runlist_out)
    varus_bam      = VARUS_RUN_LR(index_out).bam
    stringtie_out  = RUN_STRINGTIE_LR(varus_bam).assembly
    labelled       = LABEL_TRANSCRIPTS(stringtie_out).labelled
    shards         = WRITE_TFRECORD(labelled).shard

    shards.map { species, path -> "${species}\t${path}" }
          .collectFile(name: "tfrecord_manifest.tsv",
                       storeDir: "${params.outdir}",
                       newLine: true)
}
