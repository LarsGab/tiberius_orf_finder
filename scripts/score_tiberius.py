"""Score Tiberius ab initio predictions with the ORFfinder model.

Extracts the spliced CDS sequence for each Tiberius-predicted transcript,
prepends real upstream genomic sequence as context (so the model sees the
natural Kozak/5'-UTR signal before the ATG), runs the CNN-LSTM forward
pass, and reports per-transcript coding-probability scores.

Why upstream context matters
----------------------------
The model was trained on transcript sequences where the ORF sits inside
a 5'-UTR region. It learned to fire START in the context of real upstream
sequence (Kozak composition, nucleotide bias). Replacing that region with
Ns destroys the IR->START signal; start_prob drops to near zero. Using
real genomic upstream sequence restores it.

Usage
-----
python scripts/score_tiberius.py \\
  --gtf         /path/to/tiberius_seqlen.gtf \\
  --genome      /path/to/genome.fa \\
  --weights     /path/to/epoch_74.weights.h5 \\
  --config      configs/cnn_lstm_run006.yaml \\
  --out-tsv     /path/to/scores.tsv \\
  --upstream-bp 500 \\
  --batch-size  200

Output TSV columns
------------------
transcript_id   gene_id   contig   strand   n_exons   cds_len   cds_len_scored
mean_coding_prob   start_prob   stop_prob   mean_ir_prob   frac_argmax_coding

cds_len_scored == cds_len unless upstream_bp + cds_len > chunk_len, in
which case only the first (chunk_len - upstream_bp) CDS bases are scored.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml

# Label indices (tiberius_orf convention)
_IR, _START, _E1, _E2, _E0, _STOP = 0, 1, 2, 3, 4, 5
_CODING_STATES = np.array([_E0, _E1, _E2])  # indices 4, 2, 3

_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _rev_comp(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def _attr(attr_col: str, key: str) -> str | None:
    for chunk in attr_col.strip().strip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(None, 1)
        if len(parts) == 2 and parts[0] == key:
            return parts[1].strip().strip('"')
    return None


def _parse_tiberius_gtf(gtf_path: Path) -> dict:
    """Return {tid: {gene_id, contig, strand, cds: [(s, e), ...]}} from CDS lines.

    Coordinates are 0-based half-open (converted from GTF 1-based inclusive).
    CDS intervals are sorted by genomic start position (ascending).
    """
    txs: dict = {}
    for line in Path(gtf_path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "CDS":
            continue
        tid = _attr(f[8], "transcript_id")
        if tid is None:
            continue
        gid = _attr(f[8], "gene_id") or tid
        if tid not in txs:
            txs[tid] = {
                "gene_id": gid,
                "contig":  f[0],
                "strand":  f[6],
                "cds":     [],
            }
        txs[tid]["cds"].append((int(f[3]) - 1, int(f[4])))  # 0-based half-open
    for tx in txs.values():
        tx["cds"].sort(key=lambda x: x[0])
    return txs


def _get_upstream_prefix(tx: dict, genome, upstream_bp: int) -> str:
    """Return upstream_bp bases of real genomic sequence upstream of the CDS 5' end.

    + strand: take the window ending just before the first CDS exon start.
    - strand: take the window starting just after the last CDS exon end, then RC.
    Pads with N where the chromosome edge cuts the window short.
    """
    contig = tx["contig"]
    if contig not in genome:
        return "N" * upstream_bp

    chrom_len = len(genome[contig])

    if tx["strand"] == "+":
        cds_start = tx["cds"][0][0]            # 0-based start of first exon
        lo = max(0, cds_start - upstream_bp)
        seq = str(genome[contig][lo:cds_start]).upper()
        seq = "N" * (upstream_bp - len(seq)) + seq   # left-pad if near chr start
    else:
        cds_end = tx["cds"][-1][1]             # 0-based half-open end of last exon
        hi = min(chrom_len, cds_end + upstream_bp)
        seq = str(genome[contig][cds_end:hi]).upper()
        seq = seq + "N" * (upstream_bp - len(seq))   # right-pad if near chr end
        seq = _rev_comp(seq)                   # flip to 5'→3' of transcript

    return "".join(c if c in "ACGTN" else "N" for c in seq)


def _extract_cds_seq(tx: dict, genome) -> tuple[str, int]:
    """Return (spliced_cds_sequence, cds_len). Empty string if contig missing."""
    contig = tx["contig"]
    if contig not in genome:
        return "", 0
    parts: list[str] = []
    for s, e in tx["cds"]:
        seg = str(genome[contig][s:e]).upper()
        parts.append(seg)
    cds_seq = "".join(parts)
    if tx["strand"] == "-":
        cds_seq = _rev_comp(cds_seq)
    cds_seq = "".join(c if c in "ACGTN" else "N" for c in cds_seq)
    return cds_seq, len(cds_seq)


# ASCII byte → nucleotide index (A=0, C=1, G=2, T=3, N/other=4)
_NUC_ORD = np.full(256, 4, dtype=np.int32)
for _c, _i in zip("ACGT", range(4)):
    _NUC_ORD[ord(_c)] = _i


def _encode_batch(seqs: list[str], chunk_len: int) -> np.ndarray:
    """Encode sequences to (B, chunk_len, 6) float32.

    Channels: A=0 C=1 G=2 T=3 N=4 PAD=5.
    Sequences are truncated to chunk_len and right-padded with PAD.
    """
    B = len(seqs)
    arr = np.zeros((B, chunk_len, 6), dtype=np.float32)
    arr[..., 5] = 1.0  # default: PAD
    for b, seq in enumerate(seqs):
        L = min(len(seq), chunk_len)
        if L == 0:
            continue
        seq_bytes = np.frombuffer(seq[:L].encode("ascii"), dtype=np.uint8)
        nuc_idx   = _NUC_ORD[seq_bytes]
        pos       = np.arange(L)
        arr[b, pos, 5]       = 0.0
        arr[b, pos, nuc_idx] = 1.0
    return arr


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


_SCORE_NAN_KEYS = [
    "mean_coding_prob", "max_coding_prob", "min_coding_prob",
    "p10_coding_prob", "p90_coding_prob", "std_coding_prob",
    "coding_first10pct", "coding_mid80pct", "coding_last10pct",
    "codon1_mean_coding", "codon2_mean_coding",
    "start_prob", "stop_prob",
    "mean_ir_prob", "max_ir_prob", "p90_ir_prob",
    "frac_argmax_coding", "frac_argmax_ir",
]


def _score_probs(probs: np.ndarray, upstream_bp: int, cds_len: int) -> dict:
    """Score (chunk_len, 6) softmax probs over positions [upstream_bp : upstream_bp+cds_len]."""
    end = min(upstream_bp + cds_len, probs.shape[0])
    cds_probs  = probs[upstream_bp:end]
    scored_len = len(cds_probs)
    if scored_len == 0:
        return {k: float("nan") for k in _SCORE_NAN_KEYS} | {"cds_len_scored": 0}

    p_coding = cds_probs[:, _E0] + cds_probs[:, _E1] + cds_probs[:, _E2]
    p_ir     = cds_probs[:, _IR]
    argmax   = np.argmax(cds_probs, axis=-1)
    n = scored_len
    i10 = max(1, n // 10)
    i90 = max(i10 + 1, n - n // 10)
    codon1 = float(p_coding[:3].mean()) if n >= 3 else float(p_coding.mean())
    codon2 = float(p_coding[3:6].mean()) if n >= 6 else codon1
    return {
        "mean_coding_prob":   float(p_coding.mean()),
        "max_coding_prob":    float(p_coding.max()),
        "min_coding_prob":    float(p_coding.min()),
        "p10_coding_prob":    float(np.percentile(p_coding, 10)),
        "p90_coding_prob":    float(np.percentile(p_coding, 90)),
        "std_coding_prob":    float(p_coding.std()),
        "coding_first10pct":  float(p_coding[:i10].mean()),
        "coding_mid80pct":    float(p_coding[i10:i90].mean() if i90 > i10 else p_coding.mean()),
        "coding_last10pct":   float(p_coding[i90:].mean() if n > i90 else p_coding[-i10:].mean()),
        "codon1_mean_coding": codon1,
        "codon2_mean_coding": codon2,
        "start_prob":         float(cds_probs[0, _START]),
        "stop_prob":          float(cds_probs[-1, _STOP]),
        "mean_ir_prob":       float(p_ir.mean()),
        "max_ir_prob":        float(p_ir.max()),
        "p90_ir_prob":        float(np.percentile(p_ir, 90)),
        "frac_argmax_coding": float(np.isin(argmax, _CODING_STATES).mean()),
        "frac_argmax_ir":     float((argmax == _IR).mean()),
        "cds_len_scored":     scored_len,
    }


def _enable_gpu_memory_growth() -> None:
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Score Tiberius ab initio predictions with the ORFfinder model."
    )
    ap.add_argument("--gtf",          type=Path, required=True,
                    help="Tiberius prediction GTF (must contain CDS features).")
    ap.add_argument("--genome",       type=Path, required=True,
                    help="Genome FASTA matching the GTF contig names.")
    ap.add_argument("--weights",      type=Path, required=True,
                    help="Trained model weights (.h5).")
    ap.add_argument("--config",       type=Path, default=Path("configs/cnn_lstm_run006.yaml"))
    ap.add_argument("--out-tsv",      type=Path, required=True,
                    help="Output TSV path.")
    ap.add_argument("--batch-size",   type=int,  default=200,
                    help="Inference batch size (default 200).")
    ap.add_argument("--upstream-bp",  type=int,  default=500,
                    help="Bases of real genomic upstream sequence to prepend "
                         "as context before the CDS (default 500). "
                         "Replaces the N-padding approach so the model sees "
                         "genuine Kozak / 5'-UTR signal.")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg       = yaml.safe_load(open(args.config))
    chunk_len: int = cfg["data"]["chunk_len"]

    import tensorflow as tf  # noqa: F401
    _enable_gpu_memory_growth()
    from pyfaidx import Fasta
    from tiberius_orf.model.model import build_model_from_config

    model = build_model_from_config(cfg, chunk_len=chunk_len)
    model.load_weights(str(args.weights))
    print(f"[score_tiberius] Loaded weights: {args.weights}", flush=True)

    print(f"[score_tiberius] Parsing GTF: {args.gtf}", flush=True)
    txs  = _parse_tiberius_gtf(args.gtf)
    tids = sorted(txs.keys())
    print(f"[score_tiberius]   {len(tids)} transcripts", flush=True)

    genome = Fasta(str(args.genome), as_raw=True, sequence_always_upper=True)

    # Build full input sequences: upstream_prefix + spliced_CDS, capped at chunk_len
    max_cds = chunk_len - args.upstream_bp
    seq_meta: list[tuple[str, int]] = []  # (full_input_seq, true_cds_len)
    skipped = 0
    for tid in tids:
        tx = txs[tid]
        cds_seq, cds_len = _extract_cds_seq(tx, genome)
        if not cds_seq:
            skipped += 1
            seq_meta.append(("", 0))
            continue
        prefix   = _get_upstream_prefix(tx, genome, args.upstream_bp)
        full_seq = (prefix + cds_seq)[:chunk_len]
        seq_meta.append((full_seq, cds_len))

    if skipped:
        print(f"[score_tiberius]   Skipped {skipped} (missing contig)", flush=True)
    n_truncated = sum(1 for _, cds_len in seq_meta
                      if cds_len > 0 and cds_len > max_cds)
    if n_truncated:
        print(f"[score_tiberius]   {n_truncated} CDS truncated to {max_cds} nt "
              f"(chunk_len={chunk_len}, upstream_bp={args.upstream_bp})", flush=True)

    # Batched inference
    all_scores: list[dict] = [{}] * len(tids)
    batch_seqs: list[str]  = []
    batch_idx:  list[int]  = []

    def _flush() -> None:
        if not batch_seqs:
            return
        x      = _encode_batch(batch_seqs, chunk_len)
        logits = model(x, training=False).numpy()
        probs  = _softmax(logits)
        for j, i in enumerate(batch_idx):
            _, cds_len = seq_meta[i]
            all_scores[i] = _score_probs(probs[j], args.upstream_bp, cds_len)
        batch_seqs.clear()
        batch_idx.clear()

    print(f"[score_tiberius] Inference (batch_size={args.batch_size}, "
          f"upstream_bp={args.upstream_bp})", flush=True)
    for i, (seq, _) in enumerate(seq_meta):
        if not seq:
            all_scores[i] = {k: float("nan") for k in _SCORE_NAN_KEYS}
            all_scores[i]["cds_len_scored"] = 0
            continue
        batch_seqs.append(seq)
        batch_idx.append(i)
        if len(batch_seqs) >= args.batch_size:
            _flush()
        if i > 0 and i % 5000 == 0:
            print(f"[score_tiberius]   {i}/{len(tids)}", flush=True)

    _flush()
    print(f"[score_tiberius] Inference complete.", flush=True)

    # Write TSV
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    _META_COLS = ["transcript_id", "gene_id", "contig", "strand", "n_exons", "cds_len"]
    _SCORE_COLS = ["cds_len_scored"] + _SCORE_NAN_KEYS
    cols = _META_COLS + _SCORE_COLS

    def _fmt(v) -> str:
        if isinstance(v, float):
            return "nan" if v != v else f"{v:.6f}"
        return str(v)

    with open(args.out_tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for i, tid in enumerate(tids):
            tx         = txs[tid]
            sc         = all_scores[i]
            _, cds_len = seq_meta[i]
            meta = [
                tid, tx["gene_id"], tx["contig"], tx["strand"],
                str(len(tx["cds"])), str(cds_len),
            ]
            scores = [_fmt(sc.get("cds_len_scored", 0))] + [
                _fmt(sc.get(k, float("nan"))) for k in _SCORE_NAN_KEYS
            ]
            fh.write("\t".join(meta + scores) + "\n")

    print(f"[score_tiberius] Wrote {len(tids)} rows -> {args.out_tsv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
