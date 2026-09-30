#!/bin/bash
# Per-species prep for insects_test_v2: unzip genome, derive annot_cds.gff,
# filter StringTie GTF (tpm1cov3len300). One SLURM array task per species.
#
#SBATCH --job-name=prep_ins
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_ins_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_ins_%A_%a.err

set -euo pipefail
: "${SPECIES_FILE:?}" "${RESULTS_ROOT:?}" "${RUNTIME_TSV:?}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SCRIPTS_DIR=${PROJDIR}/scripts
source "${SCRIPTS_DIR}/lib/log_runtime.sh"
export RUNTIME_TSV
CLADE=insects_test_v2

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SPECIES=$(sed -n "${TASK_ID}p" "${SPECIES_FILE}")
[[ -n "${SPECIES}" ]] || { echo "no species for array index ${TASK_ID}" >&2; exit 2; }

SP_DIR=${RESULTS_ROOT}/${SPECIES}
ASM=${SP_DIR}/assembly
mkdir -p "${ASM}"

echo "[$(date -Iseconds)] prep sp=${SPECIES}"

# Unzip genome.fa.gz -> genome.fa (skip if already present)
if [[ ! -s "${ASM}/genome.fa" ]]; then
    if [[ -s "${ASM}/genome.fa.gz" ]]; then
        run_timed "${CLADE}" "${SPECIES}" prep unzip -- \
            bash -c "gunzip -kc '${ASM}/genome.fa.gz' > '${ASM}/genome.fa'"
    else
        echo "ERROR: no genome.fa.gz for ${SPECIES}" >&2; exit 3
    fi
fi
# fai index
if [[ ! -s "${ASM}/genome.fa.fai" ]]; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate orffinder
    run_timed "${CLADE}" "${SPECIES}" prep faidx -- \
        python -c "from pyfaidx import Fasta; Fasta('${ASM}/genome.fa')"
fi

# Derive annot_cds.gff (CDS-only) from annotation.gff
if [[ ! -s "${ASM}/annot_cds.gff" && -s "${ASM}/annotation.gff" ]]; then
    run_timed "${CLADE}" "${SPECIES}" prep annot_cds -- \
        bash -c "awk 'BEGIN{OFS=\"\t\"} /^#/ {print; next} \$3==\"CDS\" {print}' '${ASM}/annotation.gff' > '${ASM}/annot_cds.gff'"
fi

# StringTie tpm1cov3len300 filter
ST_IN=${SP_DIR}/stringtie/stringtie.gtf
ST_OUT=${SP_DIR}/stringtie/stringtie.filt_tpm1cov3len300.gtf
ST_TSV=${SP_DIR}/stringtie/stringtie.filt_tpm1cov3len300.decisions.tsv
if [[ ! -s "${ST_OUT}" && -s "${ST_IN}" ]]; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate orffinder
    run_timed "${CLADE}" "${SPECIES}" prep stringtie_filter -- \
        python "${SCRIPTS_DIR}/filter_stringtie_gtf.py" \
            --in-gtf "${ST_IN}" --out-gtf "${ST_OUT}" --out-tsv "${ST_TSV}" \
            --min-length 300 --min-cov 3.0 --min-tpm 1.0 \
            --long-length 3000 --min-tpm-long 0.5
fi

echo "[$(date -Iseconds)] done sp=${SPECIES}"
ls -la "${ASM}/genome.fa" "${ASM}/annot_cds.gff" "${ST_OUT}" 2>&1
