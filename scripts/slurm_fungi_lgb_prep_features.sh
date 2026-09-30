#!/bin/bash
# Fungi LGB training-data prep: per-species chain of
#   miniprot on Fungi.fa (no per-order exclusion — simpler first pass)
#   → miniprothint → annotate.py (DRUSILLA with fungi weights)
#   → gffcompare vs. reference annotation
#   → compute_orf_features.py with --ref-tmap
#
# One array task per species. When all tasks complete, submit slurm_train_fungi_lgb.sh.
#
#SBATCH --job-name=fungi_lgb_prep
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/fungi_lgb_prep_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/fungi_lgb_prep_%A_%a.err

set -euo pipefail
: "${SPECIES_FILE:?}" "${RESULTS_ROOT:?}" "${RUNTIME_TSV:?}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SCRIPTS_DIR=${PROJDIR}/scripts
ODB_FA=${ODB_FA:-${PROJDIR}/odb/raw/Fungi.fa.gz}
SIF=/opt/singularity-3.11.3/bin/singularity
TIBERIUS_SIF=${TIBERIUS_SIF:-docker://larsgabriel23/tiberius:2.0.2}
BLOSUM=${BLOSUM:-/home/gabriell/Tiberius/conf/blosum62.csv}
CHAINED_HINTS_PY=${CHAINED_HINTS_PY:-/projects/AI-GUSTUS/Tiberius/tiberius/scripts/chainedHints.py}
FUNGI_WEIGHTS=${FUNGI_WEIGHTS:-${PROJDIR}/results/models/cnn_lstm_run006_fungi/epoch_300.weights.h5}
FUNGI_CONFIG=${FUNGI_CONFIG:-${PROJDIR}/configs/cnn_lstm_run006.yaml}

source "${SCRIPTS_DIR}/lib/log_runtime.sh"
export RUNTIME_TSV
CLADE=fungi_train_lgb

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SPECIES=$(sed -n "${TASK_ID}p" "${SPECIES_FILE}")
SP_DIR=${RESULTS_ROOT}/${SPECIES}
GENOME=${SP_DIR}/assembly/genome.fa
[[ -s "${GENOME}" || -s "${GENOME}.gz" ]] || { echo "no genome for ${SPECIES}" >&2; exit 3; }
[[ -s "${GENOME}" ]] || run_timed "${CLADE}" "${SPECIES}" prep unzip -- \
    bash -c "gunzip -kc '${GENOME}.gz' > '${GENOME}'"

# Derive annot_cds.gff
if [[ ! -s "${SP_DIR}/assembly/annot_cds.gff" && -s "${SP_DIR}/assembly/annotation.gff" ]]; then
    run_timed "${CLADE}" "${SPECIES}" prep annot_cds -- \
        bash -c "awk 'BEGIN{OFS=\"\t\"} /^#/ {print; next} \$3==\"CDS\" {print}' '${SP_DIR}/assembly/annotation.gff' > '${SP_DIR}/assembly/annot_cds.gff'"
fi
REF=${SP_DIR}/assembly/annot_cds.gff
[[ -s "${REF}" ]] || { echo "no ref for ${SPECIES}" >&2; exit 3; }

# StringTie filter
STF=${SP_DIR}/stringtie/stringtie.filt_tpm1cov3len300.gtf
if [[ ! -s "${STF}" ]]; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate orffinder
    run_timed "${CLADE}" "${SPECIES}" prep stringtie_filter -- \
        python "${SCRIPTS_DIR}/filter_stringtie_gtf.py" \
            --in-gtf "${SP_DIR}/stringtie/stringtie.gtf" --out-gtf "${STF}" \
            --out-tsv "${SP_DIR}/stringtie/stringtie.filt_tpm1cov3len300.decisions.tsv" \
            --min-length 300 --min-cov 3.0 --min-tpm 1.0 --long-length 3000 --min-tpm-long 0.5
fi

# 1) Miniprot (Fungi.fa, no order exclusion for first pass)
MP_DIR=${SP_DIR}/proteins
MP_OUT=${MP_DIR}/miniprot_parsed.gff
mkdir -p "${MP_DIR}"
if [[ ! -s "${MP_OUT}" ]]; then
    run_timed "${CLADE}" "${SPECIES}" miniprot align -- \
        bash -c "${SIF} exec --bind /projects/AI-GUSTUS,/home/gabriell ${TIBERIUS_SIF} miniprot -t ${SLURM_CPUS_PER_TASK:-16} --aln '${GENOME}' '${ODB_FA}' > '${MP_DIR}/miniprot.aln'"
    run_timed "${CLADE}" "${SPECIES}" miniprot score -- \
        bash -c "${SIF} exec --bind /projects/AI-GUSTUS,/home/gabriell ${TIBERIUS_SIF} miniprot_boundary_scorer -s '${BLOSUM}' -o '${MP_OUT}' < '${MP_DIR}/miniprot.aln'"
fi

# 2) miniprothint
MPH_DIR=${MP_DIR}/miniprothint
mkdir -p "${MPH_DIR}"
if [[ ! -s "${MPH_DIR}/hc.gff" ]]; then
    run_timed "${CLADE}" "${SPECIES}" miniprothint -- \
        bash -c "cd '${MPH_DIR}' && ${SIF} exec --bind /projects/AI-GUSTUS,/home/gabriell ${TIBERIUS_SIF} miniprothint.py '${MP_OUT}' --workdir . --ignoreCoverage --topNperSeed 10 --minScoreFraction 0.5"
fi

# 3) Annotate (DRUSILLA) — requires GPU. Skip if weights not available or run out-of-band.
ANN_TAG=annotate_run006_lgb50_prep
ANN_DIR=${SP_DIR}/${ANN_TAG}
if [[ ! -s "${ANN_DIR}/orfs.filtered.gtf" ]]; then
    echo "[note] DRUSILLA output missing; needs GPU submit — see slurm_annotate_fungi_lgb50.sh"
    exit 0
fi

# 4) gffcompare (tmap)
TMAP=${ANN_DIR}/gffcmp.orfs.filtered.gtf.tmap
if [[ ! -s "${TMAP}" ]]; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate orffinder
    run_timed "${CLADE}" "${SPECIES}" gffcompare -- \
        bash -c "cd '${ANN_DIR}' && gffcompare -r '${REF}' -o gffcmp orfs.filtered.gtf"
fi

# 5) compute_orf_features with labels
FEAT=${ANN_DIR}/orf_features.tsv
if [[ ! -s "${FEAT}" ]]; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate orffinder
    run_timed "${CLADE}" "${SPECIES}" features -- \
        python "${SCRIPTS_DIR}/compute_orf_features.py" \
            --orfs-gtf "${ANN_DIR}/orfs.filtered.gtf" \
            --miniprot-gff "${MP_OUT}" \
            --hints-gff    "${MPH_DIR}/hc.gff" \
            --genome       "${GENOME}" \
            --ref-tmap     "${TMAP}" \
            --out          "${FEAT}"
fi

echo "[$(date -Iseconds)] done ${SPECIES}"
