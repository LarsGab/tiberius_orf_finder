#!/bin/bash
# Score Tiberius ab initio predictions for all 68 vertebrate training species
# using the ORFfinder model (epoch_74, run006, 500 bp upstream context).
#
# Prerequisite: slurm_run_tiberius_train_vertebrates.sh must have finished.
#
# Input per species:
#   ${RESULTS_DIR}/<sp>/tiberius/tiberius_seqlen.gtf
#   ${RESULTS_DIR}/<sp>/assembly/genome.fa
#
# Output per species:
#   ${RESULTS_DIR}/<sp>/score_tiberius_epoch_74_up500/scores.tsv
#
#SBATCH --job-name=score_tib_train
#SBATCH --partition=vision-fast,storm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-67
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/score_tib_train_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/score_tib_train_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/training_vertebrates_v2

WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run006/epoch_74.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_run006.yaml
WEIGHTS_TAG=epoch_74_up500

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
TIB_GTF=${RESULTS_DIR}/${species}/tiberius/tiberius_seqlen.gtf
OUTDIR=${RESULTS_DIR}/${species}/score_tiberius_${WEIGHTS_TAG}
OUT_TSV=${OUTDIR}/scores.tsv

test -s "${WEIGHTS}" || { echo "missing weights: ${WEIGHTS}" >&2; exit 2; }
test -s "${CONFIG}"  || { echo "missing config: ${CONFIG}"   >&2; exit 2; }

if [[ -s "${OUT_TSV}" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: scores.tsv already exists"
    exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

if [[ ! -s "${GENOME}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: missing genome.fa" >&2
    exit 0
fi
if [[ ! -s "${TIB_GTF}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: missing tiberius_seqlen.gtf" >&2
    exit 0
fi

mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate gpu

cd "${PROJDIR}"

echo "[$(date -Iseconds)] species=${species} weights=${WEIGHTS_TAG}"
echo "[$(date -Iseconds)] gtf=${TIB_GTF}"
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] out=${OUT_TSV}"

python "${PROJDIR}/scripts/score_tiberius.py" \
    --gtf         "${TIB_GTF}" \
    --genome      "${GENOME}" \
    --weights     "${WEIGHTS}" \
    --config      "${CONFIG}" \
    --out-tsv     "${OUT_TSV}" \
    --batch-size  200 \
    --upstream-bp 500

echo "[$(date -Iseconds)] done -> ${OUT_TSV}"
