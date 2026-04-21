#!/usr/bin/env nextflow
/*
 * Per-species training-data generation for the Tiberius ORF finder.
 *
 *   Input : a CSV with columns species,accession,annotation (RefSeq|BRAKER)
 *   Output: one .tfrecords shard per species, plus intermediate artefacts
 *           (genome, annotation, VARUS bam, StringTie gtf, labels, stats).
 *
 * Example:
 *   nextflow run nextflow/main.nf \
 *     --species_csv nextflow/conf/species_training.csv \
 *     --varus_dir /home/gabriell/programs/VARUS \
 *     --varus_impl /home/gabriell/programs/VARUS/Implementation \
 *     --hisat_dir /home/gabriell/programs/hisat2-2.2.1 \
 *     --braker_data_dir /home/nas-hs/projs/tiberius-insects/data/insects_data_braker \
 *     --outdir results/training
 */

nextflow.enable.dsl = 2

include { FETCH_ASSEMBLY }                       from './modules/fetch.nf'
include { MAKE_VARUS_PARAMS; RUN_VARUS }         from './modules/varus.nf'
include { RUN_STRINGTIE }                        from './modules/stringtie.nf'
include { LABEL_TRANSCRIPTS }                    from './modules/label.nf'
include { WRITE_TFRECORD }                       from './modules/tfrecord.nf'


// ---------------------------- params ----------------------------

params.species_csv      = params.species_csv      ?: null
params.outdir           = params.outdir           ?: 'results'

params.varus_dir        = params.varus_dir        ?: null
params.varus_impl       = params.varus_impl       ?: null
params.varus_runpl      = params.varus_runpl      ?: null
params.hisat_dir        = params.hisat_dir        ?: null
params.varus_max_batches = (params.containsKey('varus_max_batches') && params.varus_max_batches != null ? params.varus_max_batches : 1000) as int

params.braker_data_dir  = params.braker_data_dir  ?: null

params.threads          = (params.threads ?: 8) as int
params.chunk_len        = (params.chunk_len ?: 9999) as int


def die(msg) { log.error msg; System.exit(1) }

if (!params.species_csv)   die("Missing --species_csv")
if (!params.braker_data_dir) die("Missing --braker_data_dir")
if (!(params.varus_dir || params.varus_runpl)) die("Missing --varus_dir or --varus_runpl")
if (!params.varus_impl)    die("Missing --varus_impl")
if (!params.hisat_dir)     die("Missing --hisat_dir")


// ---------------------------- workflow ----------------------------

workflow {

    ch_species = Channel.fromPath(params.species_csv, checkIfExists: true)
        .splitCsv(header: true)
        .map { row -> tuple(row.species, row.accession, row.annotation) }

    assembly       = FETCH_ASSEMBLY(ch_species).assembly
    varus_params   = MAKE_VARUS_PARAMS(assembly)
    varus_bam      = RUN_VARUS(varus_params).bam
    stringtie_out  = RUN_STRINGTIE(varus_bam).assembly
    labelled       = LABEL_TRANSCRIPTS(stringtie_out).labelled
    shards         = WRITE_TFRECORD(labelled).shard

    shards.map { species, path -> "${species}\t${path}" }
          .collectFile(name: "tfrecord_manifest.tsv",
                       storeDir: "${params.outdir}",
                       newLine: true)
}
