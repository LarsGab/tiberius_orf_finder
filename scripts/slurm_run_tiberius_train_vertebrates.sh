#!/bin/bash
# Run Tiberius ab initio gene prediction for all 68 vertebrate training species.
#
# Genomes are expected at:
#   ${RESULTS_DIR}/<Genus_species>/assembly/genome.fa
# (published there by the Nextflow short-read training pipeline).
#
# Output per species:
#   ${RESULTS_DIR}/<Genus_species>/tiberius/tiberius_seqlen.gtf
#
#SBATCH --job-name=tib_train_vert
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-67
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_train_vert_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_train_vert_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/training_vertebrates_v2
TIBERIUS=/home/gabriell/Tiberius/tiberius.py
TIBERIUS_CFG=vertebrates

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
OUTDIR=${RESULTS_DIR}/${species}/tiberius
OUT_GTF=${OUTDIR}/tiberius_seqlen.gtf

if [[ -s "${OUT_GTF}" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: tiberius_seqlen.gtf already exists"
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

[[ -f "${TIBERIUS}" ]] || { echo "Tiberius not found: ${TIBERIUS}" >&2; exit 2; }

mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate gpu

echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] out=${OUT_GTF}"

python "${TIBERIUS}" \
    --genome    "${GENOME}" \
    --model_cfg "${TIBERIUS_CFG}" \
    --out       "${OUT_GTF}"

[[ -s "${OUT_GTF}" ]] || { echo "ERROR: Tiberius produced no GTF" >&2; exit 2; }
N_TX=$(awk -F'\t' '$3=="transcript"' "${OUT_GTF}" | wc -l || true)
echo "[$(date -Iseconds)] done -> ${OUT_GTF}  (${N_TX} transcripts)"
