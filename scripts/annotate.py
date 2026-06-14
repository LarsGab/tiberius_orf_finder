"""End-to-end ORF annotation from raw inputs (StringTie GTF or BAM).

Two input modes:
    --stringtie-gtf <gtf> --genome <fa>      StringTie has already been run
    --bam <bam>          --genome <fa>       run StringTie internally first

Output: a single genomic GTF with predicted CDS lines per transcript
(1-based inclusive), source = "tiberius_orf".

The inference path delegates chunking, boundary-mismatch detection, and
chunk-boundary repredictions to ``b2m.tools.annotate.annotate_genome`` with
the transcript FASTA used as the "genome" input — same pattern Tiberius
uses in ``main.py`` / ``eval_model_class.py``. Per-transcript CDS intervals
are then projected back to genomic coordinates via the StringTie exon
structure.

External tools required on PATH:
    gffread   (always — used to extract transcript sequences from the genome)
    stringtie (only when --bam is given)

CLI::

    python scripts/annotate.py \\
      --stringtie-gtf path/to/stringtie.gtf \\
      --genome        path/to/genome.fa \\
      --weights       results/models/run_002/epoch_41.weights.h5 \\
      --config        configs/default.yaml \\
      --out           results/annotation.gtf \\
      --batch-size    200
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml


def _enable_gpu_memory_growth() -> None:
    """Switch TF off its default preallocate-everything mode.

    Without growth enabled, an oversized intermediate workspace surfaces as
    'illegal memory access' on whichever kernel happens to be running
    (typically softmax / RowReduceKernel) instead of as a clean OOM.
    """
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="End-to-end ORF annotation.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stringtie-gtf", type=Path,
                     help="StringTie GTF (skip running StringTie).")
    ap.add_argument("--transcripts-fa", type=Path, default=None,
                     help="FASTA of assembled transcripts.")
    src.add_argument("--bam", type=Path,
                     help="Sorted alignment BAM (run StringTie internally).")
    ap.add_argument("--genome", type=Path, required=True,
                    help="Genome FASTA matching the GTF/BAM contig names.")
    ap.add_argument("--weights", type=Path, required=True,
                    help="Trained model weights (.h5).")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output GTF path.")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="Inference batch size (default 200).")
    ap.add_argument("--parallel", type=int, default=100,
                    help="HMM parallel scan factor (default 100). Per-tx "
                         "sequences are right-padded to a multiple of this.")
    ap.add_argument("--no-codon-emitter", action="store_true",
                    help="Disable hard ATG/STOP/in-frame-stop codon "
                         "constraints in the HMM (debugging variant).")
    ap.add_argument("--threads", type=int, default=4,
                    help="Threads for stringtie/gffread.")
    ap.add_argument("--longread", action="store_true",
                    help="Pass -L to StringTie (long-read assembly mode). "
                         "Use when --bam contains PacBio Iso-Seq or ONT "
                         "spliced alignments.")
    ap.add_argument("--tmp-dir", type=Path, default=None,
                    help="Workdir for intermediate files. Defaults to a tempdir.")
    ap.add_argument("--keep-tmp", action="store_true",
                    help="Don't delete the tempdir on exit.")
    # reference-free StringTie pre-filters (approximate the categories that
    # the TFRecord-build pipeline drops via project_labels)
    ap.add_argument("--min-cov", type=float, default=None,
                    help="Drop StringTie transcripts whose 'cov' attribute "
                         "is below this value. Typical: 1.0.")
    ap.add_argument("--min-tpm", type=float, default=None,
                    help="Drop StringTie transcripts whose 'TPM' attribute "
                         "is below this value.")
    ap.add_argument("--drop-unstranded", action="store_true",
                    help="Drop transcripts whose StringTie strand is '.'.")
    ap.add_argument("--drop-single-exon", action="store_true",
                    help="Drop transcripts assembled from a single exon "
                         "(often genomic/intronic noise).")
    ap.add_argument("--min-coding-length", type=int, default=200,
                    help="Drop predicted ORFs whose total coding length is "
                         "below this many nt (Tiberius default: 200). "
                         "Set to 0 to disable.")
    ap.add_argument("--single-isoform", action="store_true",
                    help="Keep at most one transcript per StringTie gene_id "
                         "(the one with the longest predicted CDS, ties "
                         "broken by transcript_id). Approximates the "
                         "single-isoform reduction Tiberius/BRAKER do.")
    return ap.parse_args(argv)


def _run_cmd(cmd: list[str]) -> None:
    print(f"  $ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True)


def _gtf_attr(attr_col: str, key: str) -> str | None:
    """Pull ``key`` out of a GTF attribute column (``key "value";`` form)."""
    for chunk in attr_col.strip().strip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(None, 1)
        if len(parts) == 2 and parts[0] == key:
            return parts[1].strip().strip('"')
    return None


def _parse_tx_to_gene(gtf_path: Path) -> dict[str, str]:
    """Build a ``{transcript_id: gene_id}`` map from a StringTie GTF."""
    out: dict[str, str] = {}
    for raw in Path(gtf_path).read_text().splitlines():
        if not raw or raw.startswith("#"):
            continue
        f = raw.split("\t")
        if len(f) < 9:
            continue
        tid = _gtf_attr(f[8], "transcript_id")
        gid = _gtf_attr(f[8], "gene_id")
        if tid is not None and gid is not None and tid not in out:
            out[tid] = gid
    return out


def _filter_stringtie_gtf(
    in_path: Path,
    out_path: Path,
    min_cov: float | None,
    min_tpm: float | None,
    drop_unstranded: bool,
    drop_single_exon: bool,
) -> tuple[int, int]:
    """Filter a StringTie GTF by per-transcript metadata.

    The TFRecord-build pipeline drops ``dropped_antisense_only`` /
    ``dropped_ref_partial`` / ``dropped_not_contained`` using the reference
    annotation. These flags approximate that filter without a reference:
    StringTie noise (low coverage, unstranded, single-exon) is the main
    source of FP ORFs once annotate.py is unrestricted.

    Returns ``(n_kept, n_dropped)``.
    """
    from collections import defaultdict

    tx_strand: dict[str, str] = {}
    tx_cov: dict[str, float] = {}
    tx_tpm: dict[str, float] = {}
    tx_exon_count: dict[str, int] = defaultdict(int)
    lines_by_tid: dict[str, list[str]] = defaultdict(list)
    header_lines: list[str] = []

    for raw in Path(in_path).read_text().splitlines():
        if not raw or raw.startswith("#"):
            header_lines.append(raw)
            continue
        f = raw.split("\t")
        if len(f) < 9:
            continue
        tid = _gtf_attr(f[8], "transcript_id")
        if tid is None:
            continue
        lines_by_tid[tid].append(raw)
        if f[2] == "transcript":
            tx_strand[tid] = f[6]
            cov = _gtf_attr(f[8], "cov")
            tpm = _gtf_attr(f[8], "TPM")
            if cov is not None:
                try:
                    tx_cov[tid] = float(cov)
                except ValueError:
                    pass
            if tpm is not None:
                try:
                    tx_tpm[tid] = float(tpm)
                except ValueError:
                    pass
        elif f[2] == "exon":
            tx_exon_count[tid] += 1
            # Fall back to strand from exon lines if no transcript record.
            tx_strand.setdefault(tid, f[6])

    keep: set[str] = set()
    for tid in lines_by_tid:
        if drop_unstranded and tx_strand.get(tid, ".") == ".":
            continue
        if min_cov is not None and tx_cov.get(tid, 0.0) < min_cov:
            continue
        if min_tpm is not None and tx_tpm.get(tid, 0.0) < min_tpm:
            continue
        if drop_single_exon and tx_exon_count.get(tid, 0) <= 1:
            continue
        keep.add(tid)

    with Path(out_path).open("w") as fh:
        for h in header_lines:
            fh.write(h + "\n")
        for tid, lines in lines_by_tid.items():
            if tid in keep:
                for ln in lines:
                    fh.write(ln + "\n")

    return len(keep), len(lines_by_tid) - len(keep)


# tiberius_orf label index -> bricks2marble's 15-state index. The b2m
# `_split_regions` helper aggregates IR/I0..I2 to 0/1 (intergenic / intron)
# and any other state to 2 (CDS); the boundary-mismatch detector in
# `_find_mismatches` treats {0,1,2,3} as "safe trailing states" and
# {4,5,6} as "exon at boundary". Mapping coding states onto E0/E1/E2 (and
# START/STOP onto Start/Stop) makes those checks behave correctly.
_TIB_TO_B2M = np.array([
    0,   # IR    -> IR     (intergenic)
    7,   # START -> Start  (CDS)
    5,   # E1    -> E1     (CDS, exon-at-boundary)
    6,   # E2    -> E2     (CDS, exon-at-boundary)
    4,   # E0    -> E0     (CDS, exon-at-boundary)
    14,  # STOP  -> Stop   (CDS)
], dtype=np.int32)


def _encode_b2m_to_model(
    nuc_int: np.ndarray, chunk_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a b2m chunked nucleotide array ``(N, T)`` to the model's
    ``(N, chunk_len, 6)`` input plus a ``(N, chunk_len)`` bool pad mask.

    Channels are ``A, C, G, T, N, PAD``. b2m's pad sentinel ``-1`` and
    lowercase letters ``5..8`` are normalised: lowercase -> uppercase (we
    don't track softmasking), and pad positions are set to PAD only. If
    ``T < chunk_len`` the trailing suffix is also marked PAD so the HMM's
    codon emitter sees ``N`` for those positions (uniform contribution).
    """
    N, T = nuc_int.shape
    if T > chunk_len:
        raise ValueError(
            f"b2m chunk T={T} exceeds model chunk_len={chunk_len}; "
            "lower T_max in annotate_genome or rebuild the model with a "
            "larger chunk_len."
        )
    nuc = nuc_int.copy()
    pad_T = (nuc == -1)
    nuc[pad_T] = 4  # placeholder; overwritten as PAD below
    # collapse softmasked lowercase letters (5..8) to uppercase (0..3)
    mask_lc = nuc > 4
    nuc[mask_lc] = nuc[mask_lc] - 5

    rows, cols = np.indices((N, T))
    oh_T = np.zeros((N, T, 6), dtype=np.float32)
    oh_T[rows, cols, nuc] = 1.0
    oh_T[pad_T, 4] = 0.0
    oh_T[pad_T, 5] = 1.0

    if T < chunk_len:
        tail = np.zeros((N, chunk_len - T, 6), dtype=np.float32)
        tail[..., 5] = 1.0
        oh = np.concatenate([oh_T, tail], axis=1)
        pad_mask = np.zeros((N, chunk_len), dtype=bool)
        pad_mask[:, :T] = pad_T
        pad_mask[:, T:] = True
    else:
        oh = oh_T
        pad_mask = pad_T

    return oh, pad_mask


def _make_predict_func(
    model,
    hmm,
    chunk_len: int,
    batch_size: int,
) -> Callable:
    """Build a ``predict_func`` for ``b2m.tools.annotate.annotate_genome``.

    Each call:
      1. Converts the b2m ``Fasta`` chunks to the model's 6-channel input.
      2. Runs the NN in mini-batches to get per-position logits.
      3. Decodes each chunk with the HMM Viterbi (via ``viterbi_decode_batch``,
         which pads internally to a multiple of ``hmm.parallel``).
      4. Truncates labels back to the b2m chunk size ``T`` and remaps the
         6-state tiberius_orf labels into b2m's 15-state convention so the
         downstream ``_split_regions`` / ``_find_mismatches`` helpers work.

    Returns ``(labels_fwd, None)`` because the transcript FASTA produced by
    gffread is already strand-oriented; there is no reverse strand to scan.
    The same function is used as the ``repredict_func`` argument: b2m's
    repredict step just re-runs us on shorter chunks centred on chunk
    boundaries, which our adapter handles identically (it doesn't care
    about evidence; ``concat_strand_to_reprediction`` is set to False).
    """
    from tiberius_orf.hmm.decode import viterbi_decode_batch

    def predict_func(fasta):
        N, T = fasta.N, fasta.T
        if N == 0:
            return np.zeros((0, T), dtype=np.int32), None
        x, pad_mask = _encode_b2m_to_model(fasta.nuc, chunk_len)
        nuc_one_hot = x[..., :5]

        # Chunk NN forward *and* HMM decode by batch_size. b2m can pack
        # tens of thousands of chunks into a single predict call; running
        # softmax + tf.scan over all of them at once was overflowing GPU
        # workspace and surfacing as an illegal memory access on the
        # softmax kernel.
        labels_out = np.empty((N, T), dtype=np.int32)
        for i in range(0, N, batch_size):
            sl = slice(i, i + batch_size)
            logits_b = model(x[sl], training=False).numpy()
            labels_b = viterbi_decode_batch(
                hmm, logits_b, nuc_one_hot[sl], pad_mask=pad_mask[sl],
            )
            labels_out[sl] = labels_b[:, :T]

        return _TIB_TO_B2M[labels_out].astype(np.int32), None

    return predict_func


def _project_tx_intervals_to_genomic(
    tx_id: str,
    cds_intervals: list[tuple[int, int]],
    tx,
    source: str,
) -> list[str]:
    """Project transcript-coord half-open CDS intervals onto the genome
    using the StringTie exon structure and emit GTF CDS lines.

    Each ``(tx_start, tx_end)`` is treated as one ORF for phase
    computation (phase = bases to skip to reach the next codon start;
    see ``tiberius_orf.data.gtf_writer.labels_to_gtf_lines`` for the
    same convention).
    """
    out: list[str] = []
    for orf_tx_start, orf_tx_end in cds_intervals:
        if orf_tx_start >= orf_tx_end:
            continue
        exons_in_tx_order = (
            list(tx.exons) if tx.strand == "+" else list(reversed(tx.exons))
        )
        cumulative = 0
        per_segment: list[tuple[int, int, int]] = []
        for g_start, g_end in exons_in_tx_order:
            exon_len = g_end - g_start
            lo = max(orf_tx_start, cumulative)
            hi = min(orf_tx_end, cumulative + exon_len)
            if lo < hi:
                off_lo = lo - cumulative
                off_hi = hi - cumulative
                if tx.strand == "+":
                    g_lo, g_hi = g_start + off_lo, g_start + off_hi
                else:
                    g_lo, g_hi = g_end - off_hi, g_end - off_lo
                per_segment.append((g_lo, g_hi, lo))
            cumulative += exon_len
            if cumulative >= orf_tx_end:
                break
        per_segment.sort()
        for g_lo, g_hi, tx_pos in per_segment:
            phase = (3 - (tx_pos - orf_tx_start) % 3) % 3
            out.append("\t".join([
                tx.contig, source, "CDS",
                str(g_lo + 1), str(g_hi),
                ".", tx.strand, str(phase),
                f'transcript_id "{tx_id}"; gene_id "{tx_id}";',
            ]))
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    dc, mc = cfg["data"], cfg["model"]
    chunk_len = dc["chunk_len"]

    # workdir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cleanup = False
    print(f"Workdir: {args.out_dir}", flush=True)

    # 1. resolve StringTie GTF
    if args.bam is not None:
        stringtie_gtf = args.out_dir / "stringtie.gtf"
        print(f"Running StringTie on {args.bam} "
              f"(longread={args.longread})", flush=True)
        st_cmd = ["stringtie", str(args.bam),
                  "-o", str(stringtie_gtf),
                  "-p", str(args.threads)]
        if args.longread:
            st_cmd.append("-L")
        _run_cmd(st_cmd)
    else:
        stringtie_gtf = args.stringtie_gtf

    # 1b. optional reference-free filtering of the StringTie GTF
    if (args.min_cov is not None or args.min_tpm is not None
            or args.drop_unstranded or args.drop_single_exon):
        filtered_gtf = args.out_dir / "stringtie.filtered.gtf"
        kept, dropped = _filter_stringtie_gtf(
            stringtie_gtf, filtered_gtf,
            min_cov=args.min_cov,
            min_tpm=args.min_tpm,
            drop_unstranded=args.drop_unstranded,
            drop_single_exon=args.drop_single_exon,
        )
        print(
            f"Filtered StringTie GTF "
            f"(min_cov={args.min_cov}, min_tpm={args.min_tpm}, "
            f"drop_unstranded={args.drop_unstranded}, "
            f"drop_single_exon={args.drop_single_exon}): "
            f"kept {kept}, dropped {dropped} -> {filtered_gtf}",
            flush=True,
        )
        stringtie_gtf = filtered_gtf

    # 2. extract transcripts FASTA via gffread
    if args.transcripts_fa is None:
        transcripts_fa = args.out_dir / "transcripts.fa"
        print(f"Extracting transcripts -> {transcripts_fa}", flush=True)
        _run_cmd(["gffread", "-w", str(transcripts_fa),
                "-g", str(args.genome), str(stringtie_gtf)])
    else:
        transcripts_fa = args.transcripts_fa

    # 3. load model + HMM
    import tensorflow as tf  # noqa: F401
    _enable_gpu_memory_growth()
    import bricks2marble as b2m
    from tiberius_orf.data.label_transcripts import parse_stringtie_gtf
    from tiberius_orf.hmm.decode import build_decoder_hmm
    from tiberius_orf.model.model import build_model_from_config

    model = build_model_from_config(cfg, chunk_len=chunk_len)
    model.load_weights(str(args.weights))
    print(f"Loaded {mc['type']} weights from {args.weights}", flush=True)

    hmm = build_decoder_hmm(
        parallel=args.parallel,
        use_codon_emitter=not args.no_codon_emitter,
    )
    print(
        f"Built OrfAnnotationHMM (parallel={args.parallel}, "
        f"codon_emitter={not args.no_codon_emitter})",
        flush=True,
    )

    # 4. exon structure for tx-coord -> genomic-coord projection
    transcripts = parse_stringtie_gtf(stringtie_gtf)
    print(f"  transcripts (exon structure): {len(transcripts)}", flush=True)

    # 5. b2m predict / repredict adapters. The same function is used for
    # both — repredict only differs in being called on shorter chunks
    # centred on chunk boundaries, which the adapter handles identically.
    predict_func = _make_predict_func(model, hmm, chunk_len, args.batch_size)
    repredict_func = predict_func

    # 6. annotate transcripts as a "genome" and project each detected
    # transcript-coord ORF back to genomic CDS lines on the fly via the
    # postprocess hook. The b2m intermediate GTF (under tmp_root) is a
    # by-product we ignore — the final genomic GTF is args.out.
    intermediate_gtf = args.out_dir / "orfs_local.gtf"
    out_gtf = args.out_dir / "orfs.gtf"
    out_log = args.out_dir / "orfs.log"

    # tx -> StringTie gene_id (only needed for --single-isoform)
    tx_to_gene: dict[str, str] = (
        _parse_tx_to_gene(stringtie_gtf) if args.single_isoform else {}
    )

    # Buffer per-tx output so single-isoform selection can run after
    # annotate_genome finishes (postprocess may be called once per b2m
    # group). tid -> (coding_length, [gtf_lines])
    per_tx_output: dict[str, tuple[int, list[str]]] = {}

    def postprocess(_fasta, annot):
        # Same short-ORF filter Tiberius uses (b2m.tools.check_min_coding_length
        # at 200 nt). Removed in place; nothing downstream sees the dropped tx.
        if args.min_coding_length > 0:
            b2m.tools.check_min_coding_length(
                annot, args.min_coding_length, remove=True,
            )
        for seq_ann in annot:
            for tx in seq_ann:
                tid = tx.sequence
                if tid not in transcripts:
                    continue
                cds_intervals = [(c.start, c.end) for c in tx.cds]
                if not cds_intervals:
                    continue
                coding_length = sum(e - s for s, e in cds_intervals)
                lines = _project_tx_intervals_to_genomic(
                    tid, cds_intervals, transcripts[tid], "tiberius_orf",
                )
                per_tx_output[tid] = (coding_length, lines)
        return annot



    if intermediate_gtf.exists():
        intermediate_gtf.unlink()
    # b2m's _merge_replace_center assumes the chunk length T is even
    # (it splits at T//2 and compares the two halves elementwise). Pass
    # an even T_max so the repred-merge broadcast lines up; the model
    # still sees chunks padded to chunk_len via _encode_b2m_to_model.
    t_max_b2m = chunk_len - (chunk_len % 2)
    b2m.tools.annotate.annotate_genome(
        fasta=transcripts_fa,
        predict_func=predict_func,
        repredict_func=repredict_func,
        output=intermediate_gtf,
        log_file=out_log,
        model_name="tiberius_orf",
        T_max=t_max_b2m,
        T_delta=0.1,
        min_sequence_size=1,
        reprediction_factor=0.5,
        concat_strand_to_reprediction=False,
        postprocess=postprocess,
        group_size_limit=1_000_000_000,
    )

    # Pick which transcripts to emit. With --single-isoform, group by the
    # StringTie gene_id and keep the transcript with the longest predicted
    # CDS (ties broken by transcript_id). Otherwise keep every transcript
    # that produced an ORF.
    if args.single_isoform:
        best_per_gene: dict[str, tuple[int, str]] = {}
        for tid, (length, _lines) in per_tx_output.items():
            gid = tx_to_gene.get(tid, tid)
            cur = best_per_gene.get(gid)
            if cur is None or length > cur[0] or (length == cur[0] and tid < cur[1]):
                best_per_gene[gid] = (length, tid)
        keep_tids: set[str] = {tid for _, tid in best_per_gene.values()}
        print(
            f"single-isoform: kept {len(keep_tids)} of {len(per_tx_output)} "
            f"transcripts ({len(best_per_gene)} StringTie genes)",
            flush=True,
        )
    else:
        keep_tids = set(per_tx_output)

    n_lines = 0
    with open(out_gtf, "w") as fh:
        for tid in sorted(keep_tids):
            for line in per_tx_output[tid][1]:
                fh.write(line + "\n")
                n_lines += 1

    print(
        f"Wrote {n_lines} CDS lines from {len(keep_tids)} transcripts "
        f"to {out_gtf}",
        flush=True,
    )

    if cleanup:
        shutil.rm(intermediate_gtf, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
