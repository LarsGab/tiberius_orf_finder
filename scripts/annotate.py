"""End-to-end ORF annotation from raw inputs (StringTie GTF or BAM).

Two input modes:
    --stringtie-gtf <gtf> --genome <fa>      StringTie has already been run
    --bam <bam>          --genome <fa>       run StringTie internally first

Output: a single genomic GTF with predicted CDS lines per transcript
(1-based inclusive), source = "tiberius_orf".

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
from collections import defaultdict
from pathlib import Path

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
    ap.add_argument("--threads", type=int, default=4,
                    help="Threads for stringtie/gffread.")
    ap.add_argument("--tmp-dir", type=Path, default=None,
                    help="Workdir for intermediate files. Defaults to a tempdir.")
    ap.add_argument("--keep-tmp", action="store_true",
                    help="Don't delete the tempdir on exit.")
    return ap.parse_args(argv)


def _run_cmd(cmd: list[str]) -> None:
    print(f"  $ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True)


def _chunk_sequence(seq: str, chunk_len: int) -> list[np.ndarray]:
    """Tile a transcript sequence into [chunk_len, 6] one-hot chunks.

    The last chunk is right-padded; padded positions have channel 5 (PAD) = 1.
    """
    from tiberius_orf.data.chunk_tfrecord import encode_nucleotides
    chunks: list[np.ndarray] = []
    L = len(seq)
    if L == 0:
        return chunks
    n_chunks = (L + chunk_len - 1) // chunk_len
    for ci in range(n_chunks):
        start = ci * chunk_len
        end   = start + chunk_len
        sub   = seq[start:end]
        real  = len(sub)
        chunk = np.zeros((chunk_len, 6), dtype=np.uint8)
        chunk[:real, :5] = encode_nucleotides(sub)
        chunk[real:, 5]  = 1   # PAD channel
        chunks.append(chunk)
    return chunks


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    dc, mc = cfg["data"], cfg["model"]

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
        print(f"Running StringTie on {args.bam}", flush=True)
        _run_cmd(["stringtie", str(args.bam),
                  "-o", str(stringtie_gtf),
                  "-p", str(args.threads)])
    else:
        stringtie_gtf = args.stringtie_gtf

    # 2. extract transcripts FASTA via gffread
    transcripts_fa = tmp_root / "transcripts.fa"
    print(f"Extracting transcripts -> {transcripts_fa}", flush=True)
    _run_cmd(["gffread", "-w", str(transcripts_fa),
              "-g", str(args.genome), str(stringtie_gtf)])

    # 3. load model
    import tensorflow as tf  # noqa: F401
    from tiberius_orf.model.model import build_model_from_config
    from tiberius_orf.hmm.viterbi import viterbi_decode
    from tiberius_orf.data.chunk_tfrecord import read_fasta
    from tiberius_orf.data.label_transcripts import parse_stringtie_gtf
    from tiberius_orf.data.gtf_writer import write_gtf

    model = build_model_from_config(cfg, chunk_len=dc["chunk_len"])
    model.load_weights(str(args.weights))
    print(f"Loaded {mc['type']} weights from {args.weights}", flush=True)

    # 4. load sequences + exon structure
    sequences   = read_fasta(transcripts_fa)
    transcripts = parse_stringtie_gtf(stringtie_gtf)
    print(f"  sequences:   {len(sequences)}", flush=True)
    print(f"  transcripts: {len(transcripts)} (with exon structure)", flush=True)

    # 5. chunk all sequences and run inference in batches
    chunk_len = dc["chunk_len"]
    chunks_per_tx: dict[str, list[np.ndarray]] = {}
    flat: list[tuple[str, int, np.ndarray]] = []
    for tx_id, seq in sequences.items():
        if tx_id not in transcripts:
            continue
        cs = _chunk_sequence(seq, chunk_len)
        if not cs:
            continue
        chunks_per_tx[tx_id] = cs
        for ci, c in enumerate(cs):
            flat.append((tx_id, ci, c))

    print(f"Running inference on {len(flat)} chunks "
          f"(batch_size={args.batch_size})", flush=True)

    logits_per_tx: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    bs = args.batch_size
    for i in range(0, len(flat), bs):
        batch = flat[i : i + bs]
        x_batch = np.stack([c.astype(np.float32) for _, _, c in batch])
        out = model(x_batch, training=False).numpy()      # [b, L, 6]
        for (tid, ci, _), lg in zip(batch, out):
            logits_per_tx[tid][ci] = lg

    # 6. stitch logits per transcript, log-softmax, Viterbi, truncate
    print("Viterbi decoding per transcript", flush=True)
    pred_labels: dict[str, np.ndarray] = {}
    for tx_id, by_ci in logits_per_tx.items():
        n = len(chunks_per_tx[tx_id])
        ordered_logits = np.concatenate([by_ci[ci] for ci in range(n)], axis=0)
        m = ordered_logits.max(axis=-1, keepdims=True)
        log_probs = ordered_logits - m - np.log(
            np.exp(ordered_logits - m).sum(axis=-1, keepdims=True)
        )
        pred_seq = viterbi_decode(log_probs)              # [n*L]
        true_len = len(sequences[tx_id])
        pred_labels[tx_id] = pred_seq[:true_len].astype(np.int32)

    # 7. write genomic GTF
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_lines = write_gtf(args.out, pred_labels, transcripts, source="tiberius_orf")
    print(f"Wrote {n_lines} CDS lines to {args.out}", flush=True)

    if cleanup:
        shutil.rmtree(tmp_root, ignore_errors=True)
    else:
        print(f"Tempdir kept: {tmp_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
