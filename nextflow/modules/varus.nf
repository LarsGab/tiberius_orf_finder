// Write a VARUS parameter file, then run VARUS to produce an aligned BAM.
// Adapted from /c/Users/lgabr/git/scripts/main.nf for the ORF-finder pipeline.

process MAKE_VARUS_PARAMS {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/varus" }, mode: 'copy', overwrite: true
    cpus 1

    input:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff)

    output:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff),
              path("VARUSparameters.txt")

    script:
    """
    cat > VARUSparameters.txt << 'EOF'
--batchSize 50000
--blockSize 5000
--components 1
--cost 0.001
--deleteLater 0
--estimator 2
--exportObservationsToFile 1
--exportParametersToFile 1
--fastqDumpCall fastq-dump
--genomeDir ./genome/
--lambda 10.0
--lessInfo 1
--loadAllOnce 0
--maxBatches ${params.varus_max_batches}
--mergeThreshold 10
--outFileNamePrefix ./
--pathToParameters ./VARUSparameters.txt
--pathToRuns ./
--pathToVARUS ${params.varus_impl}
--profitCondition 0
--pseudoCount 1
--qualityThreshold 5
--randomSeed 1
--readParametersFromFile 1
--verbosityDebug 1
EOF
    """

    stub:
    """
    touch VARUSparameters.txt
    """
}


process RUN_VARUS {
    tag { species }
    publishDir { "${params.outdir}/${species.replaceAll(' ', '_')}/varus" }, mode: 'copy', overwrite: true
    cpus params.threads

    input:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff), path(varus_params)

    output:
        tuple val(species), val(accession), val(annotation),
              path(genome), path(ref_gff), path("VARUS.bam"), emit: bam
        path "runtime.varus.txt",                             emit: runtime

    script:
    def runpl = params.varus_runpl ?: "${params.varus_dir}/runVARUS.pl"
    def parts = species.trim().split(/\s+/) as List
    def genus = parts[0]
    def sp = parts[1]
    """
    set -euo pipefail
    /usr/bin/time -p -o runtime.varus.txt \\
      perl ${runpl} \\
        --readFromTable=0 \\
        --createindex=1 \\
        --latinGenus=${genus} \\
        --latinSpecies=${sp} \\
        --speciesGenome=${genome} \\
        --aligner=HISAT \\
        --varusParameters ${varus_params} \\
        --pathToHISAT ${params.hisat_dir} \\
        --runThreadN ${task.cpus}

    bam=\$(find . -type f \\( -name "*.bam" -o -name "*VARUS*.bam" -o -name "*hisat*.bam" \\) | head -n 1 || true)
    test -n "\$bam" || { echo "no VARUS BAM produced" >&2; exit 2; }
    cp "\$bam" VARUS.bam
    """

    stub:
    """
    touch VARUS.bam runtime.varus.txt
    """
}
