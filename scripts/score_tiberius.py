"""Score Tiberius ab initio predictions with the ORFfinder model.

Extracts the spliced CDS sequence for each Tiberius-predicted transcript,
prepends a short N-context block so the model sees IR before the ATG, runs
the CNN-LSTM forward pass, and reports per-transcript coding-probability
scores.

Usage
-----
python scripts/score_tiberius.py \\
  --gtf      /path/to/tiberius_seqlen.gtf \\
  --genome   /path/to/genome.fa \\
  --weights  /path/to/epoch_74.weights.h5 \\
  --config   configs/cnn_lstm_run006.yaml \\
  --out-tsv  /path/to/scores.tsv \\
  --batch-size 200

Output TSV columns
------------------
transcript_id   gene_id   contig   strand   n_exons   cds_len   cds_len_scored
mean_coding_prob   start_prob   stop_prob   mean_ir_prob   frac_argmax_coding

cds_len_scored equals cds_len unless the sequence exceeds chunk_len (9999 nt),
in which case only the first chunk_len - n_context bases are scored.
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

# Label indices (from tiberius_orf convention)
_IR, _START, _E1, _E2, _E0, _STOP = 0, 1, 2, 3, 4, 5
_CODING_STATES = np.array([_E0, _E1, _E2])  # 4, 2, 3

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
                "contig": f[0],
                "strand": f[6],
                "cds": [],
            }
        txs[tid]["cds"].append((int(f[3]) - 1, int(f[4])))  # 0-based half-open
    for tx in txs.values():
        tx["cds"].sort(key=lambda x: x[0])
    return txs


def _extract_cds_seq(tx: dict, genome, n_context: int) -> tuple[str, int]:
    """Return (context_prefix + spliced_cds, cds_len). Empty string on missing contig."""
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
    return "N" * n_context + cds_seq, len(cds_seq)


# Lookup table for ASCII byte → nucleotide index (0=A,1=C,2=G,3=T,4=N)
_NUC_ORD = np.full(256, 4, dtype=np.int32)
for _c, _i in zip("ACGT", range(4)):
    _NUC_ORD[ord(_c)] = _i


def _encode_batch(seqs: list[str], chunk_len: int) -> np.ndarray:
    """Encode a list of sequences to (B, chunk_len, 6) float32.

    Channels: A=0, C=1, G=2, T=3, N=4, PAD=5.
    Sequences shorter than chunk_len are right-padded with PAD.
    Sequences longer than chunk_len are truncated.
    """
    B = len(seqs)
    arr = np.zeros((B, chunk_len, 6), dtype=np.float32)
    arr[..., 5] = 1.0  # default: all PAD
    for b, seq in enumerate(seqs):
        L = min(len(seq), chunk_len)
        if L == 0:
            continue
        seq_bytes = np.frombuffer(seq[:L].encode("ascii"), dtype=np.uint8)
        nuc_idx = _NUC_ORD[seq_bytes]  # (L,) in 0..4
        pos = np.arange(L)
        arr[b, pos, 5] = 0.0          # clear PAD
        arr[b, pos, nuc_idx] = 1.0    # set nucleotide channel
    return arr


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def _score_probs(probs: np.ndarray, n_context: int, cds_len: int) -> dict:
    """Compute scores from (chunk_len, 6) softmax probs over the CDS region.

    Only positions [n_context : n_context + cds_len] are scored, capped at
    the actual array length (handles truncated sequences).
    """
    end = min(n_context + cds_len, probs.shape[0])
    cds_probs = probs[n_context:end]
    scored_len = len(cds_probs)
    if scored_len == 0:
        nan = float("nan")
        return {
            "mean_coding_prob":    nan,
            "start_prob":          nan,
            "stop_prob":           nan,
            "mean_ir_prob":        nan,
            "frac_argmax_coding":  nan,
            "cds_len_scored":      0,
        }
    p_coding = cds_probs[:, _E0] + cds_probs[:, _E1] + cds_probs[:, _E2]
    argmax = np.argmax(cds_probs, axis=-1)
    return {
        "mean_coding_prob":   float(p_coding.mean()),
        "start_prob":         float(cds_probs[0, _START]),
        "stop_prob":          float(cds_probs[-1, _STOP]),
        "mean_ir_prob":       float(cds_probs[:, _IR].mean()),
        "frac_argmax_coding": float(np.isin(argmax, _CODING_STATES).mean()),
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
    ap.add_argument("--gtf",       type=Path, required=True,
                    help="Tiberius prediction GTF (CDS features).")
    ap.add_argument("--genome",    type=Path, required=True,
                    help="Genome FASTA matching GTF contig names.")
    ap.add_argument("--weights",   type=Path, required=True,
                    help="Trained model weights (.h5).")
    ap.add_argument("--config",    type=Path, default=Path("configs/cnn_lstm_run006.yaml"))
    ap.add_argument("--out-tsv",   type=Path, required=True,
                    help="Output TSV path.")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="Inference batch size (default 200).")
    ap.add_argument("--n-context",  type=int, default=50,
                    help="N bases prepended before CDS to provide IR context "
                         "(default 50).")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    chunk_len: int = cfg["data"]["chunk_len"]

    import tensorflow as tf  # noqa: F401
    _enable_gpu_memory_growth()
    from pyfaidx import Fasta
    from tiberius_orf.model.model import build_model_from_config

    model = build_model_from_config(cfg, chunk_len=chunk_len)
    model.load_weights(str(args.weights))
    print(f"[score_tiberius] Loaded weights: {args.weights}", flush=True)

    print(f"[score_tiberius] Parsing GTF: {args.gtf}", flush=True)
    txs = _parse_tiberius_gtf(args.gtf)
    print(f"[score_tiberius]   {len(txs)} transcripts", flush=True)

    genome = Fasta(str(args.genome), as_raw=True, sequence_always_upper=True)

    # --- Collect sequences -----------------------------------------------
    tids = sorted(txs.keys())
    # (full_seq_or_empty, true_cds_len)
    seq_meta: list[tuple[str, int]] = []
    skipped = 0
    max_input = chunk_len  # truncate to one chunk
    for tid in tids:
        seq, cds_len = _extract_cds_seq(txs[tid], genome, args.n_context)
        if not seq or cds_len == 0:
            skipped += 1
            seq_meta.append(("", 0))
        else:
            seq_meta.append((seq[:max_input], cds_len))
    if skipped:
        print(f"[score_tiberius]   Skipped {skipped} transcripts (missing contig)",
              flush=True)

    # --- Batched inference ------------------------------------------------
    all_scores: list[dict] = [{}] * len(tids)

    batch_seqs: list[str] = []
    batch_idx:  list[int] = []

    def _flush() -> None:
        if not batch_seqs:
            return
        x = _encode_batch(batch_seqs, chunk_len)
        logits = model(x, training=False).numpy()          # (B, chunk_len, 6)
        probs  = _softmax(logits)                          # (B, chunk_len, 6)
        for j, i in enumerate(batch_idx):
            _, cds_len = seq_meta[i]
            all_scores[i] = _score_probs(probs[j], args.n_context, cds_len)
        batch_seqs.clear()
        batch_idx.clear()

    print(f"[score_tiberius] Inference (batch_size={args.batch_size}, "
          f"chunk_len={chunk_len}, n_context={args.n_context})", flush=True)
    for i, (seq, _cds_len) in enumerate(seq_meta):
        if not seq:
            all_scores[i] = {k: float("nan") for k in [
                "mean_coding_prob", "start_prob", "stop_prob",
                "mean_ir_prob", "frac_argmax_coding", "cds_len_scored",
            ]}
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

    # --- Write TSV --------------------------------------------------------
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "transcript_id", "gene_id", "contig", "strand", "n_exons", "cds_len",
        "cds_len_scored", "mean_coding_prob", "start_prob", "stop_prob",
        "mean_ir_prob", "frac_argmax_coding",
    ]

    def _fmt(v) -> str:
        if isinstance(v, float):
            return "nan" if v != v else f"{v:.6f}"
        return str(v)

    with open(args.out_tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for i, tid in enumerate(tids):
            tx  = txs[tid]
            sc  = all_scores[i]
            _, cds_len = seq_meta[i]
            row = [
                tid,
                tx["gene_id"],
                tx["contig"],
                tx["strand"],
                str(len(tx["cds"])),
                str(cds_len),
                _fmt(sc.get("cds_len_scored", 0)),
                _fmt(sc.get("mean_coding_prob", float("nan"))),
                _fmt(sc.get("start_prob",       float("nan"))),
                _fmt(sc.get("stop_prob",         float("nan"))),
                _fmt(sc.get("mean_ir_prob",      float("nan"))),
                _fmt(sc.get("frac_argmax_coding", float("nan"))),
            ]
            fh.write("\t".join(row) + "\n")

    print(f"[score_tiberius] Wrote {len(tids)} rows -> {args.out_tsv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
