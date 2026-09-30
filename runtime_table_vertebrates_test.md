# Runtime table — vertebrates_test benchmark

All times inferred from SLURM logs on brain
(`/projects/AI-GUSTUS/tiberius_orf_finder/logs/`) and file mtimes on
`/projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test/`.
Times are approximate (rounded to the nearest minute where sensible).

## Species order (SLURM array indices for `_lorf` + `filt_tpm1cov3len300` scripts)

| Idx | Species                    |
|-----|----------------------------|
| 0   | Gallus_gallus              |
| 1   | Pristiophorus_japonicus    |
| 2   | Bos_taurus                 |
| 3   | Delphinapterus_leucas      |
| 4   | Takifugu_rubripes          |
| 5   | Zootoca_vivipara           |
| 6   | Archocentrus_centrarchus   |
| 7   | Betta_splendens            |
| 8   | Homo_sapiens               |

## 1. Annotation — Tiberius epoch_74 + lorf post-process

Source: `annot_lorf` job 7798125 (species 0–7) and `annot_homo` job 7902250 (Homo).

| Species                    | Elapsed |
|----------------------------|---------|
| Gallus_gallus              | 11m 43s |
| Pristiophorus_japonicus    |  9m 02s |
| Bos_taurus                 |  7m 14s |
| Delphinapterus_leucas      |  5m 32s |
| Takifugu_rubripes          |  6m 42s |
| Zootoca_vivipara           | 10m 13s |
| Archocentrus_centrarchus   |  7m 00s |
| Betta_splendens            |  7m 31s |
| Homo_sapiens               | 10m 30s |

Note: reflects the ORF-finder + lorf post-process pass (predict cache reused).
The cold Tiberius (predict + ORF finder) first-pass in
`annot_vert_test_7527341` was 30–55 min per species.

## 2. Protein handling

### 2a. Non-mammal species — `fix_stop_vert_7798088` (Aug 12, reused `protein_top5.fa`)

| Species                    | miniprot --aln | boundary_scorer | miniprothint | fix_stop | **Total** |
|----------------------------|----------------|-----------------|--------------|----------|-----------|
| Gallus_gallus              |  7m 08s        | 1m 35s          | 1m 41s       | 40s      | **~11 min** |
| Pristiophorus_japonicus    | 33m 03s        | 1m 50s          | 1m 56s       | 37s      | **~37 min** |
| Takifugu_rubripes          | 16m 27s        | 2m 01s          | 2m 12s       | 53s      | **~22 min** |
| Zootoca_vivipara           | 22m 09s        | 2m 17s          | 2m 08s       | 1m 07s   | **~28 min** |
| Archocentrus_centrarchus   | 29m 04s        | 2m 11s          | 1m 58s       | 42s      | **~34 min** |
| Betta_splendens            | 16m 48s        | 2m 36s          | 2m 37s       | 1m 11s   | **~23 min** |

`protein_top5.fa` (top 5 close-relative species from ODB) was pre-computed
Aug 11 (~50–100 MB per species, ~200k proteins). Diamond pre-filter is a
separate step not included above — order of ~1–2 h.

### 2b. Mammal species — `mphint_mammals_` + `prep_prot_mammals_`

Final numbers (all using `protein_top4.fa`, ~80k proteins):

| Species                    | Diamond prep | miniprot + boundary_scorer | miniprothint | **Total (excl. Diamond)** |
|----------------------------|--------------|----------------------------|--------------|---------------------------|
| Bos_taurus (top4, 7940125_0)     | 2h 46m | 10m 53s | 2m 05s | **~13 min** |
| Delphinapterus_leucas (top4, 7940125_1) | 1h 18m | 11m 57s | 1m 41s | **~14 min** |
| Homo_sapiens (top4, 7918421_2)   | 1h 39m | ~1–2h (no SLURM log for miniprot) | 2m 28s | **~1–2h + 2m** |

Old botched numbers for reference (full-ODB, killed at 8h wall):

| Species                    | Full-ODB attempt          |
|----------------------------|---------------------------|
| Bos_taurus (old, 7902222_0)      | ≥ 8h — TIMEOUT (partial scored.gff 8.8 GB) |
| Delphinapterus_leucas (old, 7902222_1) | ≥ 8h — TIMEOUT (partial scored.gff 9.3 GB) |

## 3. Filtering — `compute_orf_features` + LGB (apply on Tiberius + on ORFs)

### 3a. Non-mammal species

| Species                    | orf_features (7820672) | lgb_tib (7851717) | lgb_orfs (shared) | **Total** |
|----------------------------|------------------------|-------------------|-------------------|-----------|
| Gallus_gallus              | 1m 24s                 | 2m 32s            | ~1m 47s           | **~6 min** |
| Pristiophorus_japonicus    | 1m 21s                 | 4m 33s            | ~1m 47s           | **~8 min** |
| Takifugu_rubripes          | 1m 52s                 | 1m 48s            | ~1m 47s           | **~6 min** |
| Zootoca_vivipara           | 2m 23s                 | 2m 00s            | ~1m 47s           | **~6 min** |
| Archocentrus_centrarchus   | 1m 41s                 | 1m 49s            | ~1m 47s           | **~6 min** |
| Betta_splendens            | 2m 31s                 | 2m 10s            | ~1m 47s           | **~6 min** |

### 3b. Mammal species (includes protein-hint features from hc.gff)

| Species                    | orf_feat_mammals (7918456) | lgb_tib_mammals (7918457) | lgb_orfs (shared) | **Total** |
|----------------------------|----------------------------|---------------------------|-------------------|-----------|
| Bos_taurus                 | 32m 56s                    | 30m 33s                   | ~3m 54s           | **~67 min** |
| Delphinapterus_leucas      | 18m 31s                    | 23m 31s                   | ~3m 54s           | **~46 min** |
| Homo_sapiens               |  2m 58s                    |  6m 46s                   | ~3m 54s           | **~14 min** |

`lgb_orfs` is a single non-array job that covers all species in the batch;
the per-species share is nominal.

## Overall summary (rounded)

| Species                    | Annotation | Protein handling | Filtering |
|----------------------------|------------|------------------|-----------|
| Gallus_gallus              | ~12 min    | ~11 min          | ~6 min    |
| Pristiophorus_japonicus    | ~9 min     | ~37 min          | ~8 min    |
| Bos_taurus                 | ~7 min     | ~13 min (top4)   | ~67 min   |
| Delphinapterus_leucas      | ~6 min     | ~14 min (top4)   | ~46 min   |
| Takifugu_rubripes          | ~7 min     | ~22 min          | ~6 min    |
| Zootoca_vivipara           | ~10 min    | ~28 min          | ~6 min    |
| Archocentrus_centrarchus   | ~7 min     | ~34 min          | ~6 min    |
| Betta_splendens            | ~8 min     | ~23 min          | ~6 min    |
| Homo_sapiens               | ~11 min    | ~2–4h            | ~14 min   |

Mammal filtering is longer than non-mammal because it also computes
protein-hint features from `hc.gff` (which the non-mammals also have,
but the current LGB feature set for non-mammals doesn't use them).

## Job ID → step lookup

| Step                        | Job ID(s)                                    |
|-----------------------------|----------------------------------------------|
| Annotation (0–7)            | annot_lorf 7798125                           |
| Annotation (Homo)           | annot_homo 7902250                           |
| Protein handling (0–7 non-mammal + Homo skip) | fix_stop_vert 7798088 |
| Protein handling mammals (prep_prot) | prep_prot_mammals 7909646             |
| Protein handling mammals (old mphint) | mphint_mammals 7902222, 7909352 (Bos/Delphin), 7918421 (Homo) |
| Protein handling mammals (Bos/Delphin re-run, top4) | mphint_mammals 7940125 (COMPLETED 2026-08-24) |
| Filtering non-mammals — features | orf_features 7820672                    |
| Filtering non-mammals — lgb_tib | lgb_tib_test 7851717                     |
| Filtering non-mammals — lgb_orfs | lgb_orfs_test 7851716                    |
| Filtering mammals — features | orf_feat_mammals 7918456                    |
| Filtering mammals — lgb_tib | lgb_tib_mammals 7918457                      |
| Filtering mammals — lgb_orfs | lgb_orfs_mammals 7918632                    |
