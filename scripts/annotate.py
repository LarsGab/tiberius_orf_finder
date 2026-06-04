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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="End-to-end ORF annotation.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stringtie-gtf", type=Path,
                     help="StringTie GTF (skip running StringTie).")
    src.add_argument("--bam", type=Path,
                     help="Sorted alignment BAM (run StringTie internally).")
    ap.add_argument("--genome", type=Path, required=True,
                    help="Genome FASTA matching the GTF/BAM contig names.")
    ap.add_argument("--weights", type=Path, required=True,
                    help="Trained model weights (.h5).")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out", type=Path, required=True,
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
    return ap.parse_args(argv)


def _run_cmd(cmd: list[str]) -> None:
    print(f"  $ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True)


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

        logits = np.empty((N, chunk_len, 6), dtype=np.float32)
        for i in range(0, N, batch_size):
            logits[i:i + batch_size] = model(
                x[i:i + batch_size], training=False
            ).numpy()

        nuc_one_hot = x[..., :5]
        labels = viterbi_decode_batch(
            hmm, logits, nuc_one_hot, pad_mask=pad_mask
        )
        labels_b2m = _TIB_TO_B2M[labels[:, :T]]
        return labels_b2m.astype(np.int32), None

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
    if args.tmp_dir is None:
        tmp_root = Path(tempfile.mkdtemp(prefix="annotate_"))
        cleanup_default = True
    else:
        tmp_root = args.tmp_dir
        tmp_root.mkdir(parents=True, exist_ok=True)
        cleanup_default = False
    cleanup = cleanup_default and not args.keep_tmp
    print(f"Workdir: {tmp_root}", flush=True)

    # 1. resolve StringTie GTF
    if args.bam is not None:
        stringtie_gtf = tmp_root / "stringtie.gtf"
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

    # 2. extract transcripts FASTA via gffread
    transcripts_fa = tmp_root / "transcripts.fa"
    print(f"Extracting transcripts -> {transcripts_fa}", flush=True)
    _run_cmd(["gffread", "-w", str(transcripts_fa),
              "-g", str(args.genome), str(stringtie_gtf)])

    # 3. load model + HMM
    import tensorflow as tf  # noqa: F401
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("")
    line_counter = [0]

    def postprocess(_fasta, annotation):
        with open(args.out, "a") as fh:
            for seq_ann in annotation:
                for tx in seq_ann:
                    tid = tx.sequence
                    if tid not in transcripts:
                        continue
                    cds_intervals = [(c.start, c.end) for c in tx.cds]
                    if not cds_intervals:
                        continue
                    lines = _project_tx_intervals_to_genomic(
                        tid, cds_intervals, transcripts[tid], "tiberius_orf",
                    )
                    for line in lines:
                        fh.write(line + "\n")
                        line_counter[0] += 1
        return annotation

    intermediate_gtf = tmp_root / "tiberius_orf_tx.gtf"
    intermediate_log = tmp_root / "tiberius_orf_tx.log"
    if intermediate_gtf.exists():
        intermediate_gtf.unlink()

    b2m.tools.annotate.annotate_genome(
        fasta=transcripts_fa,
        predict_func=predict_func,
        repredict_func=repredict_func,
        output=intermediate_gtf,
        log_file=intermediate_log,
        model_name="tiberius_orf",
        T_max=chunk_len,
        T_delta=0.1,
        min_sequence_size=1,
        reprediction_factor=0.5,
        concat_strand_to_reprediction=False,
        postprocess=postprocess,
        group_size_limit=1_000_000_000,
    )

    print(f"Wrote {line_counter[0]} CDS lines to {args.out}", flush=True)

    if cleanup:
        shutil.rmtree(tmp_root, ignore_errors=True)
    else:
        print(f"Tempdir kept: {tmp_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
