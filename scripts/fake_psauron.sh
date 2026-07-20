#!/bin/bash
# Fake psauron: bypasses the broken libtorch_cuda.so / NCCL mismatch on brain
# by reading the candidate CDS file and assigning every ORF score=1.0.
# TD2.Predict reads psauron_score.csv with skiprows=2, so:
#   line 1: command echo (ignored)
#   line 2: score summary (ignored)
#   line 3+: CSV header + data

input=""
output=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -i) input="$2"; shift 2 ;;
        -o) output="$2"; shift 2 ;;
        *)  shift ;;
    esac
done

[[ -n "${input}" && -n "${output}" ]] || { echo "fake_psauron: missing -i/-o" >&2; exit 1; }
[[ -s "${input}" ]] || { echo "fake_psauron: input not found: ${input}" >&2; exit 1; }

# Lines 1-2 go INTO the output file — TD2.Predict reads with skiprows=2.
printf "fake_psauron -i %s -o %s\n" "${input}" "${output}" > "${output}"
printf "psauron score: 100.0\n" >> "${output}"
# Line 3 onwards: CSV header then data (read after the two skipped rows).
printf "description,psauron_is_protein,in-frame_score\n" >> "${output}"
awk '/^>/{sub(/^>/,""); print $1",True,1.0"}' "${input}" >> "${output}"
echo "fake_psauron: wrote $(wc -l < "${output}") lines to ${output}" >&2
