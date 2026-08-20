#!/bin/bash
# Step 2 of the training-species protein pipeline.
#
# For each of the 68 vertebrate training species:
#   1. miniprot --aln (genome vs protein_top4.fa) piped into
#      miniprot_boundary_scorer → miniprot_scored.gff
#   2. miniprothint.py → miniprothint/hc.gff (high-confidence hints)
#
# Prerequisite: slurm_prepare_proteins_training_v2.sh (protein_top4.fa)
#
# Output per species (${RESULTS_DIR}/${species}/proteins/):
#   miniprot_scored.gff
#   miniprothint/hc.gff
#
#SBATCH --job-name=miniprot_train
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-67
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/miniprot_train_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/miniprot_train_%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
module load singularity/3.11.3

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/training_vertebrates_v2
SCORING_MATRIX=/home/gabriell/Tiberius/conf/blosum62.csv
SIF=${PROJDIR}/sif/tiberius_2.0.2.sif
CPUS=${SLURM_CPUS_PER_TASK:-16}

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
PROTEIN_FA=${RESULTS_DIR}/${species}/proteins/protein_top4.fa
OUT_DIR=${RESULTS_DIR}/${species}/proteins
SCORED_GFF=${OUT_DIR}/miniprot_scored.gff
MINIPROTHINT_DIR=${OUT_DIR}/miniprothint

echo "[$(date -Iseconds)] species=${species}"

[[ -s "${SCORING_MATRIX}" ]] || { echo "ERROR: missing ${SCORING_MATRIX}" >&2; exit 2; }

if [[ ! -s "${GENOME}" ]]; then
    echo "SKIP ${species}: missing genome.fa" >&2; exit 0
fi
if [[ ! -s "${PROTEIN_FA}" ]]; then
    echo "SKIP ${species}: missing protein_top4.fa (run step 1 first)" >&2; exit 0
fi

if [[ -s "${MINIPROTHINT_DIR}/hc.gff" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: miniprothint/hc.gff already exists"
    exit 0
fi

mkdir -p "${OUT_DIR}" "${MINIPROTHINT_DIR}"

run_tool() {
    singularity exec \
        --bind /projects/AI-GUSTUS,/home/gabriell \
        "${SIF}" "$@"
}

# ── Step 1: miniprot --aln | miniprot_boundary_scorer → miniprot_scored.gff ──
if [[ ! -s "${SCORED_GFF}" ]]; then
    echo "[$(date -Iseconds)] Running miniprot --aln | miniprot_boundary_scorer …"
    run_tool miniprot \
        --aln \
        -t "${CPUS}" \
        "${GENOME}" \
        "${PROTEIN_FA}" \
    | run_tool miniprot_boundary_scorer \
        -s "${SCORING_MATRIX}" \
        -o "${SCORED_GFF}"
    echo "[$(date -Iseconds)] miniprot_scored.gff: $(wc -l < "${SCORED_GFF}") lines"
else
    echo "[$(date -Iseconds)] Reusing ${SCORED_GFF}"
fi

# ── Step 2: miniprothint ──────────────────────────────────────────────────────
echo "[$(date -Iseconds)] Running miniprothint.py …"
run_tool miniprothint.py \
    "${SCORED_GFF}" \
    --workdir          "${MINIPROTHINT_DIR}" \
    --ignoreCoverage \
    --topNperSeed      10 \
    --minScoreFraction 0.5

[[ -s "${MINIPROTHINT_DIR}/hc.gff" ]] || {
    echo "ERROR: miniprothint produced no hc.gff" >&2; exit 2
}
echo "[$(date -Iseconds)] done → ${MINIPROTHINT_DIR}/hc.gff"
wc -l "${MINIPROTHINT_DIR}/hc.gff"
