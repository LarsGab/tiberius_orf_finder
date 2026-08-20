#!/bin/bash
# Step 4 of the training-species protein pipeline.
#
# For each of the 68 vertebrate training species:
#   1. gffcompare orfs.filtered.gtf vs annot_cds.gff → .tracking
#   2. compute_orf_features.py → orf_features.tsv (protein/structural/hint features)
#   3. join_tmap_labels.py → add gffcompare_class column to orf_features.tsv
#
# Prerequisite:
#   Step 2 (miniprot_scored.gff + miniprothint/hc.gff)
#   Step 3 (orfs.filtered.gtf)
#
# Output per species (${RESULTS_DIR}/${species}/annotate_train_lorf/):
#   orf_features.tsv          (with gffcompare_class column)
#   gffcompare/orfs.tracking
#
#SBATCH --job-name=orf_feat_train
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-67
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/orf_feat_train_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/orf_feat_train_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/training_vertebrates_v2

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

ANNOT_DIR=${RESULTS_DIR}/${species}/annotate_train_lorf
ORFS=${ANNOT_DIR}/orfs.filtered.gtf
MINIPROT=${RESULTS_DIR}/${species}/proteins/miniprot_scored.gff
HINTS=${RESULTS_DIR}/${species}/proteins/miniprothint/hc.gff
GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
REF=${RESULTS_DIR}/${species}/assembly/annot_cds.gff
OUT_TSV=${ANNOT_DIR}/orf_features.tsv
GC_DIR=${ANNOT_DIR}/gffcompare

echo "[$(date -Iseconds)] species=${species}"

for f in "${ORFS}" "${MINIPROT}" "${HINTS}" "${GENOME}" "${REF}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

if [[ -s "${OUT_TSV}" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: orf_features.tsv already exists"
    exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

mkdir -p "${GC_DIR}"

# ── Step 1: compute orf features ──────────────────────────────────────────────
echo "[$(date -Iseconds)] compute_orf_features.py …"
python "${PROJDIR}/scripts/compute_orf_features.py" \
    --orfs-gtf       "${ORFS}" \
    --miniprot-gff   "${MINIPROT}" \
    --hints-gff      "${HINTS}" \
    --genome         "${GENOME}" \
    --out            "${OUT_TSV}"
echo "[$(date -Iseconds)] orf_features rows: $(wc -l < "${OUT_TSV}")"

# ── Step 2: gffcompare ────────────────────────────────────────────────────────
GC_PREFIX="${GC_DIR}/orfs"
echo "[$(date -Iseconds)] gffcompare …"
gffcompare --strict-match -e 3 -T \
    -r "${REF}" \
    -o "${GC_PREFIX}" \
    "${ORFS}"

TRACKING="${GC_PREFIX}.tracking"
[[ -s "${TRACKING}" ]] || {
    echo "ERROR: gffcompare produced no .tracking file" >&2
    ls "${GC_DIR}" >&2
    exit 2
}
echo "[$(date -Iseconds)] tracking: $(wc -l < "${TRACKING}") rows"

# ── Step 3: join class labels ─────────────────────────────────────────────────
echo "[$(date -Iseconds)] join_tmap_labels.py …"
python "${PROJDIR}/scripts/join_tmap_labels.py" \
    --tracking "${TRACKING}" \
    --features "${OUT_TSV}"

echo "[$(date -Iseconds)] done → ${OUT_TSV}"
head -1 "${OUT_TSV}" | tr '\t' '\n' | tail -3 | nl
