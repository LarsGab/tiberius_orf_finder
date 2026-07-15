"""Tests for the within-gene pruning logic in filter_orf_by_diamond_within_gene."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
_MOD_PATH = REPO_ROOT / "scripts" / "filter_orf_by_diamond_within_gene.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "filter_orf_soft", _MOD_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["filter_orf_soft"] = mod
    spec.loader.exec_module(mod)
    return mod


fs = _load()
prune_within_gene = fs.prune_within_gene
_fallback_gene_id = fs._fallback_gene_id
read_gtf_gene_map = fs.read_gtf_gene_map


def _make(genes):
    tid_to_gid = {}
    gid_to_tids = {}
    for gid, tids in genes.items():
        gid_to_tids[gid] = list(tids)
        for t in tids:
            tid_to_gid[t] = gid
    return tid_to_gid, gid_to_tids


def test_gene_all_hits_keeps_all():
    tid, gid = _make({"STRG.1": ["STRG.1.1", "STRG.1.2"]})
    kept, dropped = prune_within_gene(tid, gid,
                                      hit_tids={"STRG.1.1", "STRG.1.2"})
    assert kept == {"STRG.1.1", "STRG.1.2"}
    assert dropped == {}


def test_gene_partial_hits_drops_unhit_siblings():
    tid, gid = _make({"STRG.1": ["STRG.1.1", "STRG.1.2", "STRG.1.3"]})
    kept, dropped = prune_within_gene(tid, gid, hit_tids={"STRG.1.2"})
    assert kept == {"STRG.1.2"}
    assert dropped == {"STRG.1.1": "STRG.1.2", "STRG.1.3": "STRG.1.2"}


def test_gene_no_hits_keeps_all():
    tid, gid = _make({"STRG.1": ["STRG.1.1", "STRG.1.2"]})
    kept, dropped = prune_within_gene(tid, gid, hit_tids=set())
    assert kept == {"STRG.1.1", "STRG.1.2"}
    assert dropped == {}


def test_multiple_genes_independent():
    tid, gid = _make({
        "STRG.1": ["STRG.1.1", "STRG.1.2"],   # STRG.1.1 has hit
        "STRG.2": ["STRG.2.1"],                # no hits
        "STRG.3": ["STRG.3.1", "STRG.3.2"],   # both hit
    })
    kept, dropped = prune_within_gene(
        tid, gid, hit_tids={"STRG.1.1", "STRG.3.1", "STRG.3.2"},
    )
    assert kept == {"STRG.1.1", "STRG.2.1", "STRG.3.1", "STRG.3.2"}
    assert dropped == {"STRG.1.2": "STRG.1.1"}


def test_fallback_gene_id_from_transcript():
    assert _fallback_gene_id("STRG.42.7") == "STRG.42"
    assert _fallback_gene_id("noiso") == "noiso"


def test_read_gtf_falls_back_when_gene_id_equals_transcript_id(tmp_path):
    """ORF-pipeline output writes gene_id == transcript_id; the reader must
    fall back to the STRG.G.I → STRG.G derivation so siblings group."""
    gtf = tmp_path / "orfs.gtf"
    gtf.write_text(
        'chr1\ttiberius_orf\tCDS\t100\t200\t.\t+\t0\t'
        'transcript_id "STRG.7.1"; gene_id "STRG.7.1";\n'
        'chr1\ttiberius_orf\tCDS\t300\t400\t.\t+\t0\t'
        'transcript_id "STRG.7.2"; gene_id "STRG.7.2";\n'
        'chr1\ttiberius_orf\tCDS\t500\t600\t.\t+\t0\t'
        'transcript_id "STRG.9.1"; gene_id "STRG.9.1";\n'
    )
    tid_to_gid, gid_to_tids = read_gtf_gene_map(gtf)
    assert tid_to_gid == {
        "STRG.7.1": "STRG.7", "STRG.7.2": "STRG.7", "STRG.9.1": "STRG.9",
    }
    assert set(gid_to_tids["STRG.7"]) == {"STRG.7.1", "STRG.7.2"}
    assert gid_to_tids["STRG.9"] == ["STRG.9.1"]


def test_read_gtf_uses_proper_gene_id_when_distinct(tmp_path):
    gtf = tmp_path / "stringtie.gtf"
    gtf.write_text(
        'chr1\tStringTie\texon\t100\t200\t.\t+\t.\t'
        'gene_id "STRG.7"; transcript_id "STRG.7.1";\n'
        'chr1\tStringTie\texon\t300\t400\t.\t+\t.\t'
        'gene_id "STRG.7"; transcript_id "STRG.7.2";\n'
    )
    tid_to_gid, _ = read_gtf_gene_map(gtf)
    assert tid_to_gid == {"STRG.7.1": "STRG.7", "STRG.7.2": "STRG.7"}


def test_hits_referring_to_unknown_tid_are_ignored():
    tid, gid = _make({"STRG.1": ["STRG.1.1", "STRG.1.2"]})
    kept, dropped = prune_within_gene(
        tid, gid, hit_tids={"STRG.99.9"},  # not in this gene set
    )
    # No isoform in STRG.1 has a hit → keep everything.
    assert kept == {"STRG.1.1", "STRG.1.2"}
    assert dropped == {}
