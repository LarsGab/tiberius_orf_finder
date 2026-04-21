// Fetch or stage the genome and reference annotation for one species.
//
// RefSeq species are downloaded via the NCBI `datasets` CLI.
// BRAKER species are staged from a shared filesystem under
// ${params.braker_data_dir}/<Genus_species>/ where the expected files are:
//   <Genus_species>_renamed.fna
//   <Genus_species>_cds_longest.gtf

process FETCH_ASSEMBLY {
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
    if (annotation == 'RefSeq')
    """
    set -euo pipefail
    datasets download genome accession ${accession} \\
        --include genome,gff3 \\
        --filename ncbi.zip
    unzip -o -q ncbi.zip -d ncbi
    cat ncbi/ncbi_dataset/data/${accession}/*_genomic.fna > genome.fa
    cp ncbi/ncbi_dataset/data/${accession}/genomic.gff annotation.gff
    """
    else if (annotation == 'BRAKER')
    """
    set -euo pipefail
    src="${params.braker_data_dir}/${underscored}"
    test -d "\$src" || { echo "missing BRAKER dir: \$src" >&2; exit 2; }
    cp "\$src/${underscored}_renamed.fna" genome.fa
    cp "\$src/${underscored}_cds_longest.gtf" annotation.gff
    """
    else
    """
    echo "unknown annotation source: ${annotation}" >&2
    exit 2
    """

    stub:
    """
    touch genome.fa annotation.gff
    """
}
