# Eval observations from JBrowse2 review

Running list of model-behaviour observations to quantify and (where
warranted) address. Source: manual JBrowse2 review of one species at a
time. First pass: Bos_taurus, 2026-06-26.

## Observations

### O1. Truncated starts when ref TSS is near assembled-tx start
Real (reference) transcripts whose start codon sits near or at the very
beginning of the assembled StringTie transcript are predicted too short
on the 5' side. Worst when the ref ATG is exactly at position 0 of the
assembled sequence.

Hypothesis-not: HMM forced to start in IR.
Verified: `state_start_dist()` in
`src/tiberius_orf/hmm/annotation_hmm.py` uses a uniform initial
distribution over all 6 states, so the HMM can start in any state. Not
the cause.

Remaining candidates:
- CNN receptive field at the left edge (no left context).
- Codon emitter needs left-flanking nucleotides for a 3-base frame; at
  position 0 these don't exist.
- StringTie itself frequently drops the first few bases of the 5' UTR,
  so the true ATG may never reach the model.

Possible mitigation to test: pad the input sequence with extra N's at
the start (and possibly end) so the model always sees flanking context.

To quantify: distance from assembled-tx start to predicted start vs. the
same distance for the matched ref transcript; bin by "distance from
ref-ATG to tx-start".

### O2. Tiny isoforms that are substrings of a larger isoform
Within a gene, predictions sometimes include very small isoforms whose
ORF is a subsequence of a longer predicted isoform. These short ones
are usually wrong.

To quantify: per-gene count of predicted isoforms; flag any whose CDS
interval is a strict subinterval of another isoform's CDS in the same
gene; check ref support rate vs. the rest.

### O3. StringTie often splits one gene into two transcripts
Assembly artefact, upstream of the model, but it shapes what the model
gets to see (and what predictions look like).

To quantify: per gene, count distinct StringTie loci that overlap the
ref gene; rate at which this happens.

### O4. Many predicted isoforms are exact duplicates
Same CDS structure predicted from multiple input transcripts. After
prediction we should collapse duplicates to one transcript, keeping the
longest UTR.

To quantify: per gene, number of distinct predicted-CDS structures vs.
number of predicted transcripts.

Action: add a post-processing collapse step (CDS-identity → keep
longest-UTR representative).

### O5. Pre-prediction filter for StringTie subsequence transcripts
Before prediction, consider filtering StringTie transcripts that have a
longer alternative in the same locus when the shorter one is a true
sub-sequence (same exon structure and same nucleotide content, just
truncated). Should reduce O4 and O2.

### O6. HMM start distribution should be {IR, START} only
Design decision (2026-06-29): the initial-state distribution should put
all mass on `IR` and `START`, none on `E1 / E2 / E0 / STOP`. A transcript
can only begin in 5' UTR (`IR`) or with the ATG itself (`START`);
beginning mid-codon or mid-stop is not biological.

Current behaviour: `state_start_dist()` in
`src/tiberius_orf/hmm/annotation_hmm.py:108-117` returns uniform
`1/N_STATES` over all 6 states. STATE_NAMES is
`("IR", "START", "E1", "E2", "E0", "STOP")` (indices 0..5).

Change: restrict `allow_start` to `[0, 1]` and set `init` to a
two-element distribution on `IR` / `START` (proportions TBD — start with
the prior implied by `initial_ir_len` vs an expected 5'-UTR-then-ATG
mass, or just 50/50 as a first cut).

Likely interaction with O1: if the truncated-start symptom is partly
driven by uniform mass leaking into E0/E1/E2 at position 0, this change
should reduce O1 on its own. Worth re-running on Bos_taurus immediately
after the change to see whether O1 shrinks.

### Confirming case for O5 (PDIA5, Bos_taurus, NC_037328.1 ~67.4-67.5 Mb)
StringTie reports 4 isoforms (STRG.453.1-4); the predictions on them
yield 4 ORFs where `STRG.453.2` matches the ref PDIA5 transcript and
the other three predicted ORFs are strict sub-sequences. Concrete
illustration of why a pre-prediction subsequence filter (O5) would also
collapse O4 duplicates in this case.

## Quantification scripts
- O2 + O4 + O5: `scripts/filter_subsequence_predictions.py`
  Drops predicted isoforms whose CDS is a true sub-sequence of another
  predicted isoform in the same locus. Produces a filtered GTF + a TSV
  report of (dropped_tid -> keeper_tid).
- O1: `scripts/quantify_start_truncation.py`
  For ref ATGs that project to transcript-position <= 3 on a StringTie
  tx, categorises the prediction's start as correct / late / early /
  missing. Stdout summary + per-case TSV.
- O3: `scripts/quantify_stringtie_gene_split.py`
  Per ref protein-coding gene, counts overlapping same-strand StringTie
  loci. Prints the distribution and writes a per-gene TSV.

## Results from vertebrates_test (run 2026-06-29, job 7520931)
Excludes Homo_sapiens (data prep failed).

| Species | Sub-seq drop % | Near-start (<=3bp) correct % | At ref ATG=0 correct | Split rate % |
|---|---|---|---|---|
| Gallus_gallus | 28.7 (8793/30624) | 22.0 (40/182) | 0 / many | 12.7 (2287/18016) |
| Pristiophorus_japonicus | 25.6 (8176/31961) | 29.8 (139/467) | 0 / many | 14.3 (3183/22273) |
| Bos_taurus | 29.3 (11271/38483) | 46.5 (126/271) | 0 / many | 9.7 (2103/21695) |
| Delphinapterus_leucas | 27.4 (6182/22574) | 41.1 (157/382) | 0 / many | 13.1 (2420/18469) |

Headlines:
- O2/O4/O5: ~1 in 4 predicted transcripts is redundant; a sub-sequence
  collapse step is high-value low-effort post-processing.
- O1: confirmed and stark. Predicted-correct rate at `ref_atg_pos=0`
  is **0% in every species** -- the model never emits a START at the
  absolute first base. Strong support for O6 (restrict HMM start to
  {IR, START}) and for input left-padding with N's.
- O3: bounded -- 10-14% of ref protein-coding genes are split across
  multiple StringTie loci. Real but lower priority.

Raw outputs:
- combined: `results/vertebrates_test/eval_observations_local/eval_observations_summary.txt`
- per species: `results/vertebrates_test/eval_observations_local/<sp>/eval/*.tsv`

## Status
- 2026-06-26: list opened from Bos_taurus JBrowse review.
- 2026-06-29: added O6 (HMM start dist) and PDIA5 example for O5. Wrote
  and ran quantification scripts on 4 vertebrates_test species. Job
  7520931, 9 min wall. Numbers above.
