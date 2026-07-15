"""Tests for the joint-locus protein-evidence rescoring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
_MOD_PATH = REPO_ROOT / "scripts" / "rescore_predictions_by_protein_joint_locus.py"


def _load():
    spec = importlib.util.spec_from_file_location("rescore_jl", _MOD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rescore_jl"] = mod
    spec.loader.exec_module(mod)
    return mod


rj = _load()
rescore = rj.rescore
cluster_by_span = rj.cluster_by_span


def _rec(contig, strand, span):
    return {"contig": contig, "strand": strand, "span": span,
            "cds_spans": [span]}


def test_cluster_by_span_two_disjoint_loci():
    spans = {("chr1", "+"): [(100, 200, "a"), (150, 220, "b"),
                             (500, 600, "c")]}
    lo = cluster_by_span(spans)
    # a and b overlap, c is separate
    assert lo["a"] == lo["b"]
    assert lo["c"] != lo["a"]


def test_rescore_keeps_top_and_close_drops_far():
    tib = {"g.t1": _rec("c1", "+", (100, 500))}
    orf = {"o.1":  _rec("c1", "+", (100, 500)),
           "o.2":  _rec("c1", "+", (100, 500))}
    # top bitscore 200, tol 0.7 → threshold 140
    tib_bs = {"g.t1": 200.0}
    orf_bs = {"o.1": 180.0, "o.2": 100.0}
    kept_tib, kept_orf, report = rescore(tib, orf, tib_bs, orf_bs, tol=0.7)
    assert kept_tib == {"g.t1"}
    assert kept_orf == {"o.1"}  # 180 >= 140
    assert {r["tid"] for r in report} == {"o.2"}


def test_rescore_no_hits_keeps_all():
    tib = {"g.t1": _rec("c1", "+", (100, 500))}
    orf = {"o.1":  _rec("c1", "+", (100, 500))}
    kept_tib, kept_orf, report = rescore(tib, orf, {}, {}, tol=0.7)
    assert kept_tib == {"g.t1"}
    assert kept_orf == {"o.1"}
    assert report == []


def test_rescore_disjoint_loci_do_not_influence():
    tib = {"g.t1": _rec("c1", "+", (100, 200))}
    orf = {"o.1":  _rec("c1", "+", (500, 600))}
    # separate loci; each has its own local max
    tib_bs = {"g.t1": 300.0}
    orf_bs = {"o.1": 10.0}
    kept_tib, kept_orf, report = rescore(tib, orf, tib_bs, orf_bs, tol=0.7)
    assert kept_tib == {"g.t1"}
    assert kept_orf == {"o.1"}  # 10 is the max in its own locus
    assert report == []


def test_rescore_strand_scoping():
    tib = {"g.t1": _rec("c1", "+", (100, 500))}
    orf = {"o.1":  _rec("c1", "-", (100, 500))}
    # opposite strand → separate loci, each keeps its own
    tib_bs = {"g.t1": 300.0}
    orf_bs = {"o.1": 10.0}
    kept_tib, kept_orf, report = rescore(tib, orf, tib_bs, orf_bs, tol=0.7)
    assert kept_tib == {"g.t1"}
    assert kept_orf == {"o.1"}
    assert report == []


def test_rescore_hit_dominates_no_hit_sibling():
    tib = {"g.t1": _rec("c1", "+", (100, 500))}
    orf = {"o.1":  _rec("c1", "+", (100, 500))}
    # tib has a good hit; orf has no hit → orf gets dropped
    tib_bs = {"g.t1": 150.0}
    orf_bs = {}
    kept_tib, kept_orf, report = rescore(tib, orf, tib_bs, orf_bs, tol=0.7)
    assert kept_tib == {"g.t1"}
    assert kept_orf == set()
    assert {r["tid"] for r in report} == {"o.1"}


def test_rescore_tol_1_keeps_only_ties():
    tib = {"g.t1": _rec("c1", "+", (100, 500))}
    orf = {"o.1":  _rec("c1", "+", (100, 500))}
    tib_bs = {"g.t1": 200.0}
    orf_bs = {"o.1": 200.0}  # tied at the top
    kept_tib, kept_orf, report = rescore(tib, orf, tib_bs, orf_bs, tol=1.0)
    assert kept_tib == {"g.t1"}
    assert kept_orf == {"o.1"}
    assert report == []
