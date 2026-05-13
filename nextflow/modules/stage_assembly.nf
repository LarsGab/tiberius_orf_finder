// Stage a previously-fetched genome + reference annotation from a sibling
// pipeline run (training or val), to avoid re-downloading / re-staging.
//
// Look-up order per species:
//   1) ${params.reuse_assembly_dir}/<Genus_species>/assembly/{genome.fa,annotation.gff}
//      -> copy from there (fast path, no network).
//   2) Otherwise, fall back to the same fetch logic as modules/fetch.nf:
//        - RefSeq: download via `datasets download genome accession ${accession}`
//        - BRAKER: copy from ${params.braker_data_dir}/<Genus_species>/
//
// Falling back keeps the long-read pipeline working for newly-added species
// that the short-read pipeline never produced an assembly for.

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
    def brakerDir   = params.braker_data_dir ?: ''
    """
    set -euo pipefail

    # 1) Reuse fast path
    reuse="${params.reuse_assembly_dir}/${underscored}/assembly"
    if [[ -s "\$reuse/genome.fa" && -s "\$reuse/annotation.gff" ]]; then
        echo "[stage] reusing assembly from \$reuse"
        cp "\$reuse/genome.fa"      genome.fa
        cp "\$reuse/annotation.gff" annotation.gff
        exit 0
    fi

    # 2) Fallback: same as modules/fetch.nf
    case "${annotation}" in
        RefSeq)
            echo "[stage] no reuse copy; downloading RefSeq ${accession}"
            datasets download genome accession ${accession} \\
                --include genome,gff3 \\
                --filename ncbi.zip
            unzip -o -q ncbi.zip -d ncbi
            cat ncbi/ncbi_dataset/data/${accession}/*_genomic.fna > genome.fa
            cp  ncbi/ncbi_dataset/data/${accession}/genomic.gff annotation.gff
            ;;
        BRAKER)
            echo "[stage] no reuse copy; staging BRAKER ${underscored}"
            test -n "${brakerDir}" || { echo "missing --braker_data_dir" >&2; exit 2; }
            src="${brakerDir}/${underscored}"
            test -d "\$src" || { echo "missing BRAKER dir: \$src" >&2; exit 2; }
            cp "\$src/${underscored}_renamed.fna"      genome.fa
            cp "\$src/${underscored}_cds_longest.gtf"  annotation.gff
            ;;
        *)
            echo "unknown annotation source: ${annotation}" >&2
            exit 2
            ;;
    esac
    """

    stub:
    """
    touch genome.fa annotation.gff
    """
}
