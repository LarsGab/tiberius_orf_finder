#!/bin/bash
# Helper: resolve a per-species genome path, decompressing from .gz into
# node-local scratch when only the gzipped form is present.
#
# Usage (in a SLURM job script):
#
#   source "$(dirname "$0")/_stage_genome.sh"
#   GENOME=$(stage_genome "${RESULTS_DIR}/${species}/assembly")
#
# The decompressed copy is written under ${SLURM_TMPDIR:-/tmp}, so it lives
# only on the compute node and is cleaned up by SLURM at job end. If you
# want explicit cleanup on early exit, also do:
#   trap 'stage_genome_cleanup' EXIT

# ---------------- internals ----------------
__STAGED_GENOME_TMPDIRS=()

stage_genome_cleanup() {
    local d
    for d in "${__STAGED_GENOME_TMPDIRS[@]:-}"; do
        [[ -n "$d" && -d "$d" ]] && rm -rf "$d"
    done
    __STAGED_GENOME_TMPDIRS=()
}

# stage_genome <assembly_dir>
#   Prints the resolved genome path on stdout.
#   If only genome.fa.gz exists, decompresses it into a fresh
#   ${SLURM_TMPDIR:-/tmp}/orffinder_${SLURM_JOB_ID:-$$}_<rand>/ dir.
stage_genome() {
    local src_dir="$1"
    if [[ -z "$src_dir" ]]; then
        echo "stage_genome: missing assembly dir argument" >&2
        return 2
    fi
    local plain="${src_dir}/genome.fa"
    local gz="${src_dir}/genome.fa.gz"

    if [[ -s "$plain" ]]; then
        printf '%s\n' "$plain"
        return 0
    fi
    if [[ ! -s "$gz" ]]; then
        echo "stage_genome: no genome at ${plain} or ${gz}" >&2
        return 2
    fi

    local root="${SLURM_TMPDIR:-/tmp}"
    local job="${SLURM_JOB_ID:-$$}"
    local task="${SLURM_ARRAY_TASK_ID:-0}"
    local tmp
    tmp=$(mktemp -d "${root}/orffinder_${job}_${task}_XXXXXX")
    __STAGED_GENOME_TMPDIRS+=("$tmp")

    local out="${tmp}/genome.fa"
    echo "[stage_genome] $(date -Iseconds) decompressing ${gz} -> ${out}" >&2
    gunzip -c "$gz" > "$out"
    printf '%s\n' "$out"
}
