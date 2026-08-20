"""Filter a gene prediction GTF using ORFfinder model scores.

Transcripts are dropped when EITHER of these conditions holds:
  mean_coding_prob < --min-coding   (most of the CDS scores as non-coding)
  start_prob       < --min-start    (ATG position looks non-canonical)

All GTF lines belonging to passing transcripts are written to the output.
Gene lines are kept only when at least one of their transcripts passes.
A TSV of per-transcript scores (before filtering) is written alongside the
output GTF so the thresholds can be adjusted post-hoc without re-running
inference.

Usage
-----
python scripts/filter_by_orf_score.py \\
  --gtf         /path/to/tiberius_seqlen.gtf \\
  --genome      /path/to/genome.fa \\
  --weights     /path/to/epoch_74.weights.h5 \\
  --config      configs/cnn_lstm_run006.yaml \\
  --out-gtf     /path/to/filtered.gtf \\
  --upstream-bp 500 \\
  --min-coding  0.2 \\
  --min-start   0.2 \\
  --batch-size  200
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml

# Label indices
_IR, _START, _E1, _E2, _E0, _STOP = 0, 1, 2, 3, 4, 5
_CODING_STATES = np.array([_E0, _E1, _E2])

_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# ASCII → nucleotide index (A=0 C=1 G=2 T=3 N/other=4)
_NUC_ORD = np.full(256, 4, dtype=np.int32)
for _c, _i in zip("ACGT", range(4)):
    _NUC_ORD[ord(_c)] = _i


# ---------------------------------------------------------------------------
# GTF helpers
# ---------------------------------------------------------------------------

def _attr(attr_col: str, key: str) -> str | None:
    for chunk in attr_col.strip().strip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(None, 1)
        if len(parts) == 2 and parts[0] == key:
            return parts[1].strip().strip('"')
    return None


def _parse_gtf(gtf_path: Path) -> tuple[dict, dict, dict]:
    """Two-purpose GTF parse in one pass.

    Returns
    -------
    cds_txs : {tid: {gene_id, contig, strand, cds: [(s,e),...]}}
        CDS intervals (0-based half-open) needed for scoring.
    tid_lines : {tid: [raw_lines]}
        All non-gene lines grouped by transcript_id.
    gene_info : {gene_id: {line: str, tids: [str]}}
        Gene-level lines and their child transcript_ids.
    """
    cds_txs:   dict = {}
    tid_lines: dict = defaultdict(list)
    gene_info: dict = {}

    for raw in Path(gtf_path).read_text().splitlines():
        if not raw or raw.startswith("#"):
            continue
        f = raw.split("\t")
        if len(f) < 9:
            continue
        feature = f[2]
        tid = _attr(f[8], "transcript_id")
        gid = _attr(f[8], "gene_id")

        if feature == "gene" or tid is None:
            if gid and gid not in gene_info:
                gene_info[gid] = {"line": raw, "tids": []}
            continue

        tid_lines[tid].append(raw)

        if gid:
            if gid not in gene_info:
                gene_info[gid] = {"line": None, "tids": []}
            if tid not in gene_info[gid]["tids"]:
                gene_info[gid]["tids"].append(tid)

        if feature == "CDS":
            if tid not in cds_txs:
                cds_txs[tid] = {
                    "gene_id": gid or tid,
                    "contig":  f[0],
                    "strand":  f[6],
                    "cds":     [],
                }
            cds_txs[tid]["cds"].append((int(f[3]) - 1, int(f[4])))

    for tx in cds_txs.values():
        tx["cds"].sort(key=lambda x: x[0])

    return cds_txs, dict(tid_lines), gene_info


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------

def _rev_comp(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def _get_upstream_prefix(tx: dict, genome, upstream_bp: int) -> str:
    contig = tx["contig"]
    if contig not in genome:
        return "N" * upstream_bp
    chrom_len = len(genome[contig])
    if tx["strand"] == "+":
        cds_start = tx["cds"][0][0]
        lo  = max(0, cds_start - upstream_bp)
        seq = str(genome[contig][lo:cds_start]).upper()
        seq = "N" * (upstream_bp - len(seq)) + seq
    else:
        cds_end = tx["cds"][-1][1]
        hi  = min(chrom_len, cds_end + upstream_bp)
        seq = str(genome[contig][cds_end:hi]).upper()
        seq = seq + "N" * (upstream_bp - len(seq))
        seq = _rev_comp(seq)
    return "".join(c if c in "ACGTN" else "N" for c in seq)


def _extract_cds_seq(tx: dict, genome) -> tuple[str, int]:
    contig = tx["contig"]
    if contig not in genome:
        return "", 0
    parts = [str(genome[contig][s:e]).upper() for s, e in tx["cds"]]
    cds_seq = "".join(parts)
    if tx["strand"] == "-":
        cds_seq = _rev_comp(cds_seq)
    cds_seq = "".join(c if c in "ACGTN" else "N" for c in cds_seq)
    return cds_seq, len(cds_seq)


# ---------------------------------------------------------------------------
# Encoding + inference
# ---------------------------------------------------------------------------

def _encode_batch(seqs: list[str], chunk_len: int) -> np.ndarray:
    B   = len(seqs)
    arr = np.zeros((B, chunk_len, 6), dtype=np.float32)
    arr[..., 5] = 1.0  # PAD
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


def _score_probs(probs: np.ndarray, upstream_bp: int, cds_len: int) -> dict:
    end        = min(upstream_bp + cds_len, probs.shape[0])
    cds_probs  = probs[upstream_bp:end]
    scored_len = len(cds_probs)
    if scored_len == 0:
        nan = float("nan")
        return {k: nan for k in [
            "mean_coding_prob", "start_prob", "stop_prob",
            "mean_ir_prob", "frac_argmax_coding",
        ]} | {"cds_len_scored": 0}
    p_coding = cds_probs[:, _E0] + cds_probs[:, _E1] + cds_probs[:, _E2]
    argmax   = np.argmax(cds_probs, axis=-1)
    return {
        "mean_coding_prob":   float(p_coding.mean()),
        "start_prob":         float(cds_probs[0, _START]),
        "stop_prob":          float(cds_probs[-1, _STOP]),
        "mean_ir_prob":       float(cds_probs[:, _IR].mean()),
        "frac_argmax_coding": float(np.isin(argmax, _CODING_STATES).mean()),
        "cds_len_scored":     scored_len,
    }


def _build_seq_meta(
    tids: list[str],
    cds_txs: dict,
    genome,
    chunk_len: int,
    upstream_bp: int,
) -> list[tuple[str, int]]:
    """Build (full_input_seq, true_cds_len) list in tid order."""
    seq_meta: list[tuple[str, int]] = []
    for tid in tids:
        tx = cds_txs[tid]
        cds_seq, cds_len = _extract_cds_seq(tx, genome)
        if not cds_seq:
            seq_meta.append(("", 0))
        else:
            prefix   = _get_upstream_prefix(tx, genome, upstream_bp)
            seq_meta.append(((prefix + cds_seq)[:chunk_len], cds_len))
    return seq_meta


def _score_all(
    seq_meta: list[tuple[str, int]],
    tids: list[str],
    model,
    chunk_len: int,
    upstream_bp: int,
    batch_size: int,
) -> dict[str, dict]:
    """Return {tid: score_dict} given a pre-built seq_meta list."""
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
            all_scores[i] = _score_probs(probs[j], upstream_bp, cds_len)
        batch_seqs.clear()
        batch_idx.clear()

    nan_score = {k: float("nan") for k in [
        "mean_coding_prob", "start_prob", "stop_prob",
        "mean_ir_prob", "frac_argmax_coding", "cds_len_scored",
    ]}
    nan_score["cds_len_scored"] = 0

    for i, (seq, _) in enumerate(seq_meta):
        if not seq:
            all_scores[i] = nan_score.copy()
            continue
        batch_seqs.append(seq)
        batch_idx.append(i)
        if len(batch_seqs) >= batch_size:
            _flush()
        if i > 0 and i % 5000 == 0:
            print(f"  {i}/{len(tids)}", flush=True)

    _flush()
    return {tid: all_scores[i] for i, tid in enumerate(tids)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_filtered_gtf(
    out_gtf: Path,
    passing_tids: set[str],
    tid_lines: dict,
    gene_info: dict,
) -> tuple[int, int]:
    """Write passing transcripts (and their parent gene lines) to out_gtf.

    Gene lines are kept when at least one child transcript passes.
    Line order: gene line → all lines for each passing transcript.

    Returns (n_genes_written, n_transcripts_written).
    """
    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    n_genes = n_tx = 0

    with open(out_gtf, "w") as fh:
        for gid, info in gene_info.items():
            passing_children = [t for t in info["tids"] if t in passing_tids]
            if not passing_children:
                continue
            n_genes += 1
            if info["line"] is not None:
                fh.write(info["line"] + "\n")
            for tid in passing_children:
                for line in tid_lines.get(tid, []):
                    fh.write(line + "\n")
                n_tx += 1

        # Transcripts not linked to any gene line (edge case)
        linked = {t for info in gene_info.values() for t in info["tids"]}
        for tid in sorted(passing_tids - linked):
            for line in tid_lines.get(tid, []):
                fh.write(line + "\n")
            n_tx += 1

    return n_genes, n_tx


def _write_scores_tsv(
    out_tsv: Path,
    tids: list[str],
    cds_txs: dict,
    seq_meta: list[tuple[str, int]],
    scores: dict[str, dict],
    passing_tids: set[str],
) -> None:
    cols = [
        "transcript_id", "gene_id", "contig", "strand", "n_exons",
        "cds_len", "cds_len_scored",
        "mean_coding_prob", "start_prob", "stop_prob",
        "mean_ir_prob", "frac_argmax_coding", "pass_filter",
    ]

    def _fmt(v) -> str:
        if isinstance(v, float):
            return "nan" if v != v else f"{v:.6f}"
        return str(v)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for i, tid in enumerate(tids):
            tx  = cds_txs.get(tid, {})
            sc  = scores.get(tid, {})
            _, cds_len = seq_meta[i] if i < len(seq_meta) else ("", 0)
            row = [
                tid,
                tx.get("gene_id", ""),
                tx.get("contig", ""),
                tx.get("strand", ""),
                str(len(tx.get("cds", []))),
                str(cds_len),
                _fmt(sc.get("cds_len_scored", 0)),
                _fmt(sc.get("mean_coding_prob",   float("nan"))),
                _fmt(sc.get("start_prob",          float("nan"))),
                _fmt(sc.get("stop_prob",           float("nan"))),
                _fmt(sc.get("mean_ir_prob",        float("nan"))),
                _fmt(sc.get("frac_argmax_coding",  float("nan"))),
                "1" if tid in passing_tids else "0",
            ]
            fh.write("\t".join(row) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _enable_gpu_memory_growth() -> None:
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Filter gene predictions by ORFfinder model score."
    )
    ap.add_argument("--gtf",          type=Path, required=True,
                    help="Input gene prediction GTF (must contain CDS features).")
    ap.add_argument("--genome",       type=Path, required=True,
                    help="Genome FASTA matching the GTF contig names.")
    ap.add_argument("--weights",      type=Path, required=True,
                    help="Trained model weights (.h5).")
    ap.add_argument("--config",       type=Path, default=Path("configs/cnn_lstm_run006.yaml"))
    ap.add_argument("--out-gtf",      type=Path, required=True,
                    help="Output filtered GTF path.")
    ap.add_argument("--out-tsv",      type=Path, default=None,
                    help="Optional path for per-transcript score TSV. "
                         "Defaults to <out-gtf>.scores.tsv.")
    ap.add_argument("--upstream-bp",  type=int, default=500,
                    help="Bases of upstream genomic context prepended before "
                         "the CDS (default 500).")
    ap.add_argument("--min-coding",   type=float, default=0.2,
                    help="Drop transcripts with mean_coding_prob below this "
                         "(default 0.2).")
    ap.add_argument("--min-start",    type=float, default=0.2,
                    help="Drop transcripts with start_prob below this "
                         "(default 0.2).")
    ap.add_argument("--batch-size",   type=int, default=200)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args     = _parse_args(argv)
    out_tsv  = args.out_tsv or args.out_gtf.with_suffix(".scores.tsv")
    cfg      = yaml.safe_load(open(args.config))
    chunk_len: int = cfg["data"]["chunk_len"]

    import tensorflow as tf  # noqa: F401
    _enable_gpu_memory_growth()
    from pyfaidx import Fasta
    from tiberius_orf.model.model import build_model_from_config

    model = build_model_from_config(cfg, chunk_len=chunk_len)
    model.load_weights(str(args.weights))
    print(f"[filter] Loaded weights: {args.weights}", flush=True)

    # 1. Parse GTF
    print(f"[filter] Parsing: {args.gtf}", flush=True)
    cds_txs, tid_lines, gene_info = _parse_gtf(args.gtf)
    tids = sorted(cds_txs.keys())
    print(f"[filter]   {len(tids)} transcripts with CDS", flush=True)

    genome = Fasta(str(args.genome), as_raw=True, sequence_always_upper=True)

    # 2. Build input sequences once (reused for scoring and TSV output)
    seq_meta = _build_seq_meta(tids, cds_txs, genome, chunk_len, args.upstream_bp)

    # 3. Score
    print(f"[filter] Scoring (upstream_bp={args.upstream_bp}, "
          f"batch_size={args.batch_size}) …", flush=True)
    scores = _score_all(seq_meta, tids, model, chunk_len,
                        args.upstream_bp, args.batch_size)
    print(f"[filter] Scoring complete.", flush=True)

    # 4. Filter
    passing_tids: set[str] = set()
    n_low_coding = n_low_start = n_nan = 0
    for tid in tids:
        sc = scores[tid]
        mc = sc.get("mean_coding_prob", float("nan"))
        sp = sc.get("start_prob",       float("nan"))
        if mc != mc or sp != sp:   # nan
            n_nan += 1
            continue
        if mc < args.min_coding:
            n_low_coding += 1
            continue
        if sp < args.min_start:
            n_low_start += 1
            continue
        passing_tids.add(tid)

    n_dropped = len(tids) - len(passing_tids)
    print(
        f"[filter] Filter (min_coding={args.min_coding}, "
        f"min_start={args.min_start}):",
        flush=True,
    )
    print(f"[filter]   passed  : {len(passing_tids):>8,}", flush=True)
    print(f"[filter]   dropped : {n_dropped:>8,}  "
          f"(low_coding={n_low_coding}, low_start={n_low_start}, "
          f"no_score={n_nan})", flush=True)

    # 5. Write output GTF
    n_genes, n_tx = _write_filtered_gtf(
        args.out_gtf, passing_tids, tid_lines, gene_info,
    )
    print(f"[filter] Wrote {n_tx:,} transcripts in {n_genes:,} genes "
          f"-> {args.out_gtf}", flush=True)

    # 6. Write score TSV
    _write_scores_tsv(out_tsv, tids, cds_txs, seq_meta, scores, passing_tids)
    print(f"[filter] Score TSV -> {out_tsv}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
