// Stage a previously-fetched genome + reference annotation from a sibling
// pipeline run (training or val), to avoid re-downloading / re-staging.
//
// Looks for the per-species assembly under:
//   ${params.reuse_assembly_dir}/<Genus_species>/assembly/genome.fa
//   ${params.reuse_assembly_dir}/<Genus_species>/assembly/annotation.gff
//
// If both files exist non-empty, copies them into the work dir; otherwise the
// process fails (the caller is expected to point reuse_assembly_dir at an
// already-finished short-read pipeline output).

process STAGE_ASSEMBLY {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/assembly" }, mode: 'copy', overwrite: true
    cpus 1

    input:
        tuple val(species), val(accession), val(annotation)

    output:
        tuple val(species),
              val(accession),
              val(annotation),
              path("genome.fa"),
              path("annotation.gff"), emit: assembly

    script:
    def underscored = species.replaceAll(' ', '_')
    """
    set -euo pipefail
    src="${params.reuse_assembly_dir}/${underscored}/assembly"
    test -s "\$src/genome.fa"       || { echo "missing reuse genome: \$src/genome.fa" >&2; exit 2; }
    test -s "\$src/annotation.gff"  || { echo "missing reuse annotation: \$src/annotation.gff" >&2; exit 2; }
    cp "\$src/genome.fa"      genome.fa
    cp "\$src/annotation.gff" annotation.gff
    """

    stub:
    """
    touch genome.fa annotation.gff
    """
}
