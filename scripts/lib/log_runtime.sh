#!/bin/bash
# Runtime logger for benchmark pipeline SLURM scripts.
#
# Usage:
#   source "$(dirname "$0")/lib/log_runtime.sh"
#   RUNTIME_TSV=/path/to/runtimes.tsv
#   run_timed <clade> <species> <tool> <phase> -- <cmd> [args...]
#
# Appends one row per invocation to $RUNTIME_TSV:
#   clade species tool phase seconds wallclock host timestamp exit_status
# Non-zero exit propagates.

_runtime_init_tsv() {
    local tsv="$1"
    local dir; dir=$(dirname "$tsv")
    mkdir -p "$dir"
    # Header line only if file is new/empty
    if [[ ! -s "$tsv" ]]; then
        printf 'clade\tspecies\ttool\tphase\tseconds\twallclock\thost\ttimestamp\texit\n' > "$tsv"
    fi
}

_fmt_wallclock() {
    local s=$1
    printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

run_timed() {
    local clade=$1; local species=$2; local tool=$3; local phase=$4; shift 4
    if [[ "$1" != "--" ]]; then
        echo "run_timed: expected '--' separator before command" >&2
        return 2
    fi
    shift
    local tsv=${RUNTIME_TSV:?RUNTIME_TSV env var must be set}
    _runtime_init_tsv "$tsv"

    local t0; t0=$(date +%s)
    local ts; ts=$(date -Iseconds)
    local host; host=$(hostname -s)
    local rc=0
    "$@" || rc=$?
    local t1; t1=$(date +%s)
    local dt=$((t1 - t0))
    local wc; wc=$(_fmt_wallclock "$dt")

    # Locking append: flock if available, else best-effort
    if command -v flock >/dev/null 2>&1; then
        (
            flock -w 30 9
            printf '%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%d\n' \
                "$clade" "$species" "$tool" "$phase" "$dt" "$wc" "$host" "$ts" "$rc" >> "$tsv"
        ) 9>>"$tsv"
    else
        printf '%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%d\n' \
            "$clade" "$species" "$tool" "$phase" "$dt" "$wc" "$host" "$ts" "$rc" >> "$tsv"
    fi
    return $rc
}
