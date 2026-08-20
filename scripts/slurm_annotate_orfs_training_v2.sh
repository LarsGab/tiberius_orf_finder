#!/bin/bash
# Step 3 of the training-species protein pipeline (runs in parallel with steps 1-2).
#
# For each of the 68 vertebrate training species:
#   1. Filter raw StringTie GTF (TPM≥1, coverage≥3, length≥300).
#   2. annotate.py → orfs.gtf + orfs.partial.gtf (with --lorf-class).
#   3. filter_subsequence_predictions.py → orfs.filtered.gtf.
#
# Output per species:
#   ${RESULTS_DIR}/${species}/annotate_train_lorf/
#     orfs.gtf
#     orfs.partial.gtf
#     orfs.filtered.gtf
#
#SBATCH --job-name=annot_train
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-67
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_train_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_train_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/training_vertebrates_v2
WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run006/epoch_74.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_run006.yaml
FILT_TAG=filt_tpm1cov3len300
CPUS=${SLURM_CPUS_PER_TASK:-4}

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    "Acipenser_ruthenus"
    "Alligator_mississippiensis"
    "Alosa_sapidissima"
    "Amblyraja_radiata"
    "Anguilla_anguilla"
    "Anolis_sagrei"
    "Apus_apus"
    "Bombina_bombina"
    "Carcharodon_carcharias"
    "Corvus_hawaiiensis"
    "Dendropsophus_ebraccatus"
    "Denticeps_clupeoides"
    "Elgaria_multicarinata_webbii"
    "Emys_orbicularis"
    "Erpetoichthys_calabaricus"
    "Esox_lucius"
    "Eublepharis_macularius"
    "Falco_peregrinus"
    "Gavia_stellata"
    "Labrus_mixtus"
    "Latimeria_chalumnae"
    "Lepisosteus_oculatus"
    "Megalops_cyprinoides"
    "Mobula_birostris"
    "Narcine_bancroftii"
    "Osmerus_eperlanus"
    "Parambassis_ranga"
    "Pelobates_fuscus"
    "Podarcis_raffonei"
    "Rana_temporaria"
    "Rhinatrema_bivittatum"
    "Scleropages_formosus"
    "Solea_solea"
    "Sparus_aurata"
    "Aotus_nancymaae"
    "Camelus_bactrianus"
    "Canis_lupus_familiaris"
    "Cavia_porcellus"
    "Cebus_capucinus"
    "Ceratotherium_simum"
    "Chinchilla_lanigera"
    "Condylura_cristata"
    "Desmodus_rotundus"
    "Dipodomys_ordii"
    "Enhydra_lutris"
    "Eptesicus_fuscus"
    "Equus_caballus"
    "Heterocephalus_glaber"
    "Ictidomys_tridecemlineatus"
    "Jaculus_jaculus"
    "Loxodonta_africana"
    "Marmota_marmota"
    "Microcebus_murinus"
    "Microtus_ochrogaster"
    "Mus_musculus"
    "Mus_pahari"
    "Neomonachus_schauinslandi"
    "Ochotona_princeps"
    "Octodon_degus"
    "Odobenus_rosmarus"
    "Otolemur_garnettii"
    "Panthera_pardus"
    "Propithecus_coquereli"
    "Puma_concolor"
    "Rattus_norvegicus"
    "Saimiri_boliviensis"
    "Sorex_araneus"
    "Trichechus_manatus"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}
GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
STRINGTIE_RAW=${RESULTS_DIR}/${species}/stringtie/stringtie.gtf
STRINGTIE_FILT=${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.gtf
OUTDIR=${RESULTS_DIR}/${species}/annotate_train_lorf

echo "[$(date -Iseconds)] species=${species}"

test -s "${WEIGHTS}" || { echo "ERROR: missing weights: ${WEIGHTS}" >&2; exit 2; }
test -s "${CONFIG}"  || { echo "ERROR: missing config: ${CONFIG}"   >&2; exit 2; }

if [[ -s "${OUTDIR}/orfs.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: orfs.filtered.gtf already exists"
    exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

if [[ ! -s "${GENOME}" ]]; then
    echo "SKIP ${species}: missing genome.fa" >&2; exit 0
fi
if [[ ! -s "${STRINGTIE_RAW}" ]]; then
    echo "SKIP ${species}: missing stringtie.gtf" >&2; exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

mkdir -p "${OUTDIR}"

if [[ ! -s "${STRINGTIE_FILT}" ]]; then
    echo "[$(date -Iseconds)] filtering stringtie GTF …"
    python "${PROJDIR}/scripts/filter_stringtie_gtf.py" \
        --in-gtf       "${STRINGTIE_RAW}" \
        --out-gtf      "${STRINGTIE_FILT}" \
        --out-tsv      "${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.decisions.tsv" \
        --min-length   300 \
        --min-cov      3.0 \
        --min-tpm      1.0 \
        --long-length  3000 \
        --min-tpm-long 0.5
fi
test -s "${STRINGTIE_FILT}" || { echo "ERROR: filter produced no GTF" >&2; exit 2; }

N_TX=$(awk -F'\t' '$3=="transcript"' "${STRINGTIE_FILT}" | wc -l || true)
if [[ "${N_TX}" -eq 0 ]]; then
    echo "SKIP ${species}: 0 transcripts in filtered GTF — too few reads, skipping"
    exit 0
fi
echo "[$(date -Iseconds)] ${N_TX} transcripts after filtering"

echo "[$(date -Iseconds)] annotating ORFs …"
cd "${PROJDIR}"

python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${STRINGTIE_FILT}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    200 \
    --threads       "${CPUS}" \
    --lorf-class \
    --partial-out   "${OUTDIR}/orfs.partial.gtf"

echo "[$(date -Iseconds)] subseq collapse …"

python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
    --orfs-gtf   "${OUTDIR}/orfs.gtf" \
    --out-gtf    "${OUTDIR}/orfs.filtered.gtf" \
    --report-tsv "${OUTDIR}/dropped_subsequences.tsv"

echo "[$(date -Iseconds)] done → ${OUTDIR}/orfs.filtered.gtf"
N=$(awk -F'\t' '$3=="transcript"' "${OUTDIR}/orfs.filtered.gtf" | wc -l || true)
echo "[$(date -Iseconds)] ${N} transcripts in orfs.filtered.gtf"
