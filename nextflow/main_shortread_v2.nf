#!/usr/bin/env nextflow
/*
 * Per-species short-read training-data generation for the Tiberius ORF
 * finder, using VARUS v2 (python `varus` CLI, HISAT2 aligner).
 *
 *   Input : a CSV with columns species,accession,annotation
 *           annotation ∈ {RefSeq | Genbank | ensembl | DDBJ | EMBL | NCBI |
 *                         BRAKER | Phytozome}
 *
 *   Output: one .tfrecords shard per species under
 *           ${outdir}/<Genus_species>/tfrecord/data.tfrecords plus a
 *           tfrecord_manifest.tsv at the root.
 *
 * Pipeline:
 *   FETCH_ASSEMBLY    (RefSeq via NCBI datasets, or BRAKER/Phytozome staged)
 *   VARUS_RUNLIST_SR  (NCBI Entrez query)
 *   VARUS_INDEX_SR    (HISAT2 index)
 *   VARUS_RUN_SR      (online sampling + HISAT2 alignment)
 *   RUN_STRINGTIE     (samtools sort + stringtie + gffread)
 *   LABEL_TRANSCRIPTS (project reference CDS, write labels.npz)
 *   WRITE_TFRECORD    (chunk + tfrecord shard)
 *
 * The legacy entry point (`main.nf`) continues to use the Perl runVARUS.pl
 * via `modules/varus.nf` for reproducing existing insect runs.
 *
 * Example (HPC, brain):
 *   nextflow run nextflow/main_shortread_v2.nf \
 *     -c nextflow/conf/brain_shortread_v2.config \
 *     --species_csv nextflow/conf/fungi/species_training.csv \
 *     --outdir /home/gabriell/tiberius_orf_finder/results/training_fungi \
 *     -resume
 */

nextflow.enable.dsl = 2

include { FETCH_ASSEMBLY }                                          from './modules/fetch.nf'
include { VARUS_RUNLIST_SR; VARUS_INDEX_SR; VARUS_RUN_SR }          from './modules/varus_shortread.nf'
include { RUN_STRINGTIE }                                           from './modules/stringtie.nf'
include { LABEL_TRANSCRIPTS }                                       from './modules/label.nf'
include { WRITE_TFRECORD }                                          from './modules/tfrecord.nf'


// ---------------------------- params ----------------------------

params.species_csv         = params.species_csv         ?: null
params.outdir              = params.outdir              ?: 'results_shortread_v2'

params.braker_data_dir     = params.braker_data_dir     ?: null
params.phytozome_data_dir  = params.phytozome_data_dir  ?: null

// VARUS v2 run-time hyperparameters (short-read defaults match the CLI).
params.varus_max_batches    = (params.containsKey('varus_max_batches')   && params.varus_max_batches   != null ? params.varus_max_batches   : 1000) as int
params.varus_batch_size     = (params.containsKey('varus_batch_size')    && params.varus_batch_size    != null ? params.varus_batch_size    : 50000) as int
params.varus_tile_size      = (params.containsKey('varus_tile_size')     && params.varus_tile_size     != null ? params.varus_tile_size     : 5000) as int
params.varus_min_uniq_pct   = (params.containsKey('varus_min_uniq_pct')  && params.varus_min_uniq_pct  != null ? params.varus_min_uniq_pct  : 5.0) as double
params.varus_max_runs       = (params.containsKey('varus_max_runs')      && params.varus_max_runs      != null ? params.varus_max_runs      : 0) as int
params.varus_seed           = (params.containsKey('varus_seed')          && params.varus_seed          != null ? params.varus_seed          : 1) as int
params.varus_bootstrap_all  = (params.containsKey('varus_bootstrap_all') ? params.varus_bootstrap_all : false) as boolean
params.varus_profit_condition = (params.containsKey('varus_profit_condition') ? params.varus_profit_condition : false) as boolean
params.varus_pipeline_downloads = (params.containsKey('varus_pipeline_downloads') ? params.varus_pipeline_downloads : false) as boolean
params.varus_paired_only    = (params.containsKey('varus_paired_only')   ? params.varus_paired_only   : false) as boolean
params.varus_index_cpus     = (params.containsKey('varus_index_cpus')    && params.varus_index_cpus    != null ? params.varus_index_cpus    : 8) as int

params.ncbi_email           = params.ncbi_email   ?: null
params.ncbi_api_key         = params.ncbi_api_key ?: null

// VARUS v2 entry point.
params.varus_cmd            = params.varus_cmd ?: 'python -m varus.cli'

params.threads              = (params.threads ?: 8) as int
params.chunk_len            = (params.containsKey('chunk_len') && params.chunk_len != null ? params.chunk_len : 9999) as int


def die(msg) { log.error msg; System.exit(1) }


// ---------------------------- workflow ----------------------------

workflow {

    if (!params.species_csv) die("Missing --species_csv")

    ch_species = Channel.fromPath(params.species_csv, checkIfExists: true)
        .splitCsv(header: true, quote: '"')
        .map { row -> tuple(row.species, row.accession, row.annotation) }

    assembly       = FETCH_ASSEMBLY(ch_species).assembly
    runlist_out    = VARUS_RUNLIST_SR(assembly)
    index_out      = VARUS_INDEX_SR(runlist_out)
    varus_bam      = VARUS_RUN_SR(index_out).bam
    stringtie_out  = RUN_STRINGTIE(varus_bam).assembly
    labelled       = LABEL_TRANSCRIPTS(stringtie_out).labelled
    shards         = WRITE_TFRECORD(labelled).shard

    shards.map { species, path -> "${species}\t${path}" }
          .collectFile(name: "tfrecord_manifest.tsv",
                       storeDir: "${params.outdir}",
                       newLine: true)
}
