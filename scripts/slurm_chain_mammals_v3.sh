#!/bin/bash
# Chain orchestrator v3: submits miniprothint array=2 (Homo_sapiens only),
# then chains compute_orf_features + apply_lgb_tiberius → apply_lgb_orfs → eval_lgb_comparison.
# Bos/Delphin already have hc.gff and will skip miniprothint.
#
#SBATCH --job-name=chain_mammals_v3
#SBATCH --partition=snowball
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=36:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/chain_mammals_v3_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/chain_mammals_v3_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

poll_until_done() {
    local job=$1 label=$2
    sleep 60
    while true; do
        local status
        status=$(sacct -u gabriell -j "$job" --format=JobID,State,ExitCode --noheader 2>/dev/null \
                 | grep -E "^${job}(_[0-9]+)?[[:space:]]" || true)
        local n_active
        n_active=$(echo "$status" | grep -cE 'RUNNING|PENDING' || true)
        echo "[$(date)] $label active tasks: $n_active"
        if [ "$n_active" -eq 0 ]; then
            echo "$status"
            return 0
        fi
        sleep 300
    done
}

check_completed() {
    local job=$1 label=$2
    local status
    status=$(sacct -u gabriell -j "$job" --format=JobID,State,ExitCode --noheader 2>/dev/null \
             | grep -E "^${job}(_[0-9]+)?[[:space:]]" || true)
    if echo "$status" | grep -qvE 'COMPLETED[[:space:]]+0:0'; then
        echo "[$(date)] $label FAILED:"
        echo "$status"
        return 1
    fi
    echo "[$(date)] $label: all tasks COMPLETED"
    return 0
}

echo "[$(date)] Chain v3 started."

# ── Step 1: miniprothint array=2 (Homo_sapiens only) ─────────────────────────
echo "[$(date)] Submitting miniprothint array=2 (Homo_sapiens)..."
MPHINT_JOB=$(sbatch --array=2 $PROJDIR/scripts/slurm_miniprothint_mammals_vertebrates_test.sh \
             2>&1 | grep -oP '(?<=job )\d+')
echo "[$(date)] miniprothint job: $MPHINT_JOB"
if [ -z "$MPHINT_JOB" ]; then
    echo "[$(date)] sbatch for miniprothint failed. Aborting."; exit 1
fi

poll_until_done "$MPHINT_JOB" "miniprothint"
check_completed "$MPHINT_JOB" "miniprothint" || { echo "[$(date)] Aborting."; exit 1; }

# Verify hc.gff for all 3 species
MISSING=0
for SPECIES in Bos_taurus Delphinapterus_leucas Homo_sapiens; do
    F=$PROJDIR/results/vertebrates_test/$SPECIES/fix_stop/miniprothint/hc.gff
    if [ -s "$F" ]; then
        echo "[$(date)] OK hc.gff: $SPECIES ($(du -h "$F" | cut -f1))"
    else
        echo "[$(date)] MISSING hc.gff: $SPECIES"
        MISSING=$((MISSING+1))
    fi
done
if [ $MISSING -gt 0 ]; then
    echo "[$(date)] $MISSING hc.gff file(s) missing. Aborting."; exit 1
fi

# ── Step 2: compute_orf_features + apply_lgb_tiberius in parallel ─────────────
echo "[$(date)] Submitting compute_orf_features and apply_lgb_tiberius in parallel..."
FEAT_JOB=$(sbatch --array=0-2 $PROJDIR/scripts/slurm_compute_orf_features_mammals_vertebrates_test.sh \
           2>&1 | grep -oP '(?<=job )\d+')
TIBLGB_JOB=$(sbatch --array=0-2 $PROJDIR/scripts/slurm_apply_lgb_tiberius_mammals_vertebrates_test.sh \
             2>&1 | grep -oP '(?<=job )\d+')
echo "[$(date)] compute_orf_features job: $FEAT_JOB"
echo "[$(date)] apply_lgb_tiberius job: $TIBLGB_JOB"
if [ -z "$FEAT_JOB" ] || [ -z "$TIBLGB_JOB" ]; then
    echo "[$(date)] sbatch failed for feat/tiblgb. Aborting."; exit 1
fi

poll_until_done "$FEAT_JOB" "compute_orf_features"
check_completed "$FEAT_JOB" "compute_orf_features" || { echo "[$(date)] Aborting."; exit 1; }

# ── Step 3: apply_lgb_orfs ────────────────────────────────────────────────────
echo "[$(date)] Submitting apply_lgb_orfs..."
ORFLGB_JOB=$(sbatch $PROJDIR/scripts/slurm_apply_lgb_orfs_mammals_vertebrates_test.sh \
             2>&1 | grep -oP '(?<=job )\d+')
echo "[$(date)] apply_lgb_orfs job: $ORFLGB_JOB"
if [ -z "$ORFLGB_JOB" ]; then
    echo "[$(date)] sbatch for apply_lgb_orfs failed. Aborting."; exit 1
fi

# ── Step 4: wait for apply_lgb_tiberius + apply_lgb_orfs ─────────────────────
for WAIT_JOB in $TIBLGB_JOB $ORFLGB_JOB; do
    poll_until_done "$WAIT_JOB" "job $WAIT_JOB"
    check_completed "$WAIT_JOB" "job $WAIT_JOB" || { echo "[$(date)] Aborting."; exit 1; }
done

# Verify LGB outputs
MISSING=0
for SPECIES in Bos_taurus Delphinapterus_leucas Homo_sapiens; do
    F1=$PROJDIR/results/vertebrates_test/$SPECIES/tiberius_lgb_filtered/tiberius_lgb_filtered.gtf
    F2=$PROJDIR/results/vertebrates_test/$SPECIES/annotate_epoch_74_filt_tpm1cov3len300_lorf/orfs_lgb3_filtered.gtf
    for F in "$F1" "$F2"; do
        if [ -s "$F" ]; then
            echo "[$(date)] OK: $F"
        else
            echo "[$(date)] MISSING: $F"
            MISSING=$((MISSING+1))
        fi
    done
done
if [ $MISSING -gt 0 ]; then
    echo "[$(date)] $MISSING LGB output(s) missing. Aborting."; exit 1
fi

# ── Step 5: eval_lgb_comparison ──────────────────────────────────────────────
echo "[$(date)] All LGB outputs present. Submitting eval_lgb_comparison..."
EVAL_JOB=$(sbatch $PROJDIR/scripts/slurm_eval_lgb_comparison.sh \
           2>&1 | grep -oP '(?<=job )\d+')
echo "[$(date)] eval_lgb_comparison job: $EVAL_JOB"
echo "[$(date)] === PIPELINE CHAIN COMPLETE. Final eval job: $EVAL_JOB ==="
