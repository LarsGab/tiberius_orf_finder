// Fetch or stage the genome and reference annotation for one species.
//
// Three source families are supported:
//   * RefSeq    -> download via `datasets download genome accession <GCA|GCF>`
//                  Used for any row whose annotation label denotes a
//                  publicly-fetchable NCBI assembly. Annotation labels that
//                  map to this branch: RefSeq, Genbank, ensembl, ENSEMBL,
//                  DDBJ, EMBL, NCBI.
//   * BRAKER    -> stage from ${params.braker_data_dir}/<Genus_species>/
//                  Expected files:
//                    <Genus_species>_renamed.fna
//                    <Genus_species>_cds_longest.gtf
//   * Phytozome -> stage from ${params.phytozome_data_dir}/<Genus_species>/
//                  Expected files:
//                    genome.fa
//                    annotation.gff
//                  Used for the Embryophyta (Mesangiospermae) Figshare bundle
//                  (doi:10.6084/m9.figshare.32086728), pre-staged into a
//                  per-species symlink layout by
//                  scripts/stage_mesangiospermae_figshare.py.

process FETCH_ASSEMBLY {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/assembly" },
        mode: 'copy', overwrite: true,
        pattern: "{genome.fa.gz,annotation.gff}"

    cpus 1

    input:
        tuple val(species), val(accession), val(annotation)

    output:
        tuple val(species),
              val(accession),
              val(annotation),
              path("genome.fa"),
              path("annotation.gff"), emit: assembly
        path "genome.fa.gz", emit: assembly_gz

    script:
    def underscored = species.replaceAll(' ', '_')
    def ncbiLabels  = ['RefSeq', 'Genbank', 'GenBank', 'ensembl', 'ENSEMBL', 'DDBJ', 'EMBL', 'NCBI']
    def source
    if (ncbiLabels.contains(annotation))       source = 'ncbi'
    else if (annotation == 'BRAKER')           source = 'braker'
    else if (annotation == 'Phytozome')        source = 'phytozome'
    else                                       source = 'unknown'
    """
    set -euo pipefail

    # --- 1) fetch/stage raw genome.fa + annotation.gff ---
    case "${source}" in
      ncbi)
        datasets download genome accession ${accession} \\
            --include genome,gff3 \\
            --filename ncbi.zip
        unzip -o -q ncbi.zip -d ncbi
        cat ncbi/ncbi_dataset/data/${accession}/*_genomic.fna > genome.fa
        cp ncbi/ncbi_dataset/data/${accession}/genomic.gff annotation.gff
        ;;
      braker)
        src="${params.braker_data_dir}/${underscored}"
        test -d "\$src" || { echo "missing BRAKER dir: \$src" >&2; exit 2; }
        # Two BRAKER layouts are supported:
        #   1. Pre-staged (diatoms / mesangiospermae-style):
        #         <src>/genome.fa  +  <src>/annotation.gff
        #   2. Legacy insects layout:
        #         <src>/<Genus_species>_renamed.fna
        #         <src>/<Genus_species>_cds_longest.gtf
        if [[ -s "\$src/genome.fa" && -s "\$src/annotation.gff" ]]; then
            cp -L "\$src/genome.fa"      genome.fa
            cp -L "\$src/annotation.gff" annotation.gff
        elif [[ -s "\$src/${underscored}_renamed.fna" && -s "\$src/${underscored}_cds_longest.gtf" ]]; then
            cp "\$src/${underscored}_renamed.fna"      genome.fa
            cp "\$src/${underscored}_cds_longest.gtf"  annotation.gff
        else
            echo "BRAKER source dir \$src has neither {genome.fa,annotation.gff} nor legacy ${underscored}_renamed.fna/_cds_longest.gtf" >&2
            exit 2
        fi
        ;;
      phytozome)
        src="${params.phytozome_data_dir}/${underscored}"
        test -d "\$src" || { echo "missing Phytozome dir: \$src" >&2; exit 2; }
        # symlinks (created by stage_mesangiospermae_figshare.py) are followed
        cp -L "\$src/genome.fa"      genome.fa
        cp -L "\$src/annotation.gff" annotation.gff
        ;;
      *)
        echo "unknown annotation source: ${annotation}" >&2
        exit 2
        ;;
    esac

    # --- 2) reflow FASTA to uniform 80-char line widths per record ---
    # gffread and samtools faidx reject records whose internal lines have
    # varying width. Some Phytozome bundles (Zea mays, Sorghum bicolor,
    # Thinopyrum intermedium, Agave tequilana, Manihot esculenta) ship
    # with mixed line widths inside a chromosome, which crashes RUN_STRINGTIE
    # at the gffread step. Constant-memory reflow with awk fixes it in one
    # streaming pass regardless of chromosome size.
    awk '
      /^>/ { if (buf != "") print buf; print; buf=""; next }
      {
        seq = buf \$0
        while (length(seq) >= 80) { print substr(seq,1,80); seq = substr(seq,81) }
        buf = seq
      }
      END { if (buf != "") print buf }
    ' genome.fa > genome.reflowed.fa
    mv genome.reflowed.fa genome.fa

    # --- 3) publish gzipped copy ---
    gzip -kf genome.fa
    """

    stub:
    """
    touch genome.fa annotation.gff genome.fa.gz
    """
}
