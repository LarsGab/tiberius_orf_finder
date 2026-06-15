#!/usr/bin/env nextflow
/*
 * Reuse-BAM variant of the short-read training-data pipeline.
 *
 * For species whose pre-crash VARUS.bam under /home_old survived the
 * brain crash with status 'ok' (per
 *   scripts/check_varus_assembly_integrity.py
 * +
 *   scripts/split_species_by_bam_status.py),
 * re-fetch the assembly fresh and pipe the salvaged BAM straight into
 * StringTie + label projection + tfrecord writing.
 *
 * Pipeline:
 *   FETCH_ASSEMBLY      (fresh, into ${outdir})
 *   STAGE_OLD_VARUS_BAM (symlink-only, ${old_outdir} is read-only)
 *   RUN_STRINGTIE
 *   LABEL_TRANSCRIPTS   (current code -> includes partial-ref categories)
 *   WRITE_TFRECORD
 *
 * Inputs:
 *   --species_csv  CSV (species,accession,annotation) restricted to
 *                  the 'reuse_bam' subset.
 *   --old_outdir   Read-only pre-crash results dir; expected layout:
 *                    <old_outdir>/<Genus_species>/varus/VARUS.bam
 *   --outdir       Fresh writable results dir for this run.
 *
 * Example (brain):
 *   nextflow run nextflow/main_reuse_bam.nf \
 *     -c nextflow/conf/brain_shortread_v2.config \
 *     --species_csv results/integrity_checks/split_vertebrates/reuse_bam.csv \
 *     --old_outdir  /home_old/gabriell/tiberius_orf_finder/results/training_vertebrates \
 *     --outdir      /home/gabriell/tiberius_orf_finder/results/training_vertebrates_v2 \
 *     -resume
 */

nextflow.enable.dsl = 2

include { FETCH_ASSEMBLY }     from './modules/fetch.nf'
include { RUN_STRINGTIE }      from './modules/stringtie.nf'
include { LABEL_TRANSCRIPTS }  from './modules/label.nf'
include { WRITE_TFRECORD }     from './modules/tfrecord.nf'


// ---------------------------- params ----------------------------

params.species_csv         = params.species_csv         ?: null
params.old_outdir          = params.old_outdir          ?: null
params.outdir              = params.outdir              ?: 'results_reuse_bam'

params.braker_data_dir     = params.braker_data_dir     ?: null
params.phytozome_data_dir  = params.phytozome_data_dir  ?: null

params.threads             = (params.threads ?: 8) as int
params.chunk_len           = (params.containsKey('chunk_len') && params.chunk_len != null ? params.chunk_len : 9999) as int


def die(msg) { log.error msg; System.exit(1) }


// ---------------------------- workflow ----------------------------

workflow {

    if (!params.species_csv) die("Missing --species_csv")
    if (!params.old_outdir)  die("Missing --old_outdir (path to pre-crash BAMs, read-only).")

    def oldRoot = file(params.old_outdir)
    if (!oldRoot.exists()) die("--old_outdir does not exist: ${params.old_outdir}")

    ch_species = Channel.fromPath(params.species_csv, checkIfExists: true)
        .splitCsv(header: true, quote: '"')
        .map { row -> tuple(row.species, row.accession, row.annotation) }

    // (species, oldBam) channel. Fail fast if any expected BAM is missing
    // so we don't silently drop species.
    ch_oldbams = Channel.fromPath(params.species_csv, checkIfExists: true)
        .splitCsv(header: true, quote: '"')
        .map { row ->
            def underscored = row.species.replaceAll(' ', '_')
            def bam = file("${params.old_outdir}/${underscored}/varus/VARUS.bam")
            if (!bam.exists() || bam.size() == 0)
                throw new RuntimeException("Missing or empty old BAM for ${row.species}: ${bam}")
            tuple(row.species, bam)
        }

    assembly = FETCH_ASSEMBLY(ch_species).assembly
    //   assembly: (species, accession, annotation, genome, ref_gff)

    // Join assembly with the old BAM on species. Result tuple matches
    // RUN_STRINGTIE input contract:
    //   (species, accession, annotation, genome, ref_gff, bam)
    stringtie_in = assembly.join(ch_oldbams)

    stringtie_out = RUN_STRINGTIE(stringtie_in).assembly
    labelled      = LABEL_TRANSCRIPTS(stringtie_out).labelled
    shards        = WRITE_TFRECORD(labelled).shard

    shards.map { species, path -> "${species}\t${path}" }
          .collectFile(name: "tfrecord_manifest.tsv",
                       storeDir: "${params.outdir}",
                       newLine: true)
}
