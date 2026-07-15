"""Tests for scripts/filter_orf_against_tiberius.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
_MOD_PATH = REPO_ROOT / "scripts" / "filter_orf_against_tiberius.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "filter_orf_vs_tib", _MOD_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["filter_orf_vs_tib"] = mod
    spec.loader.exec_module(mod)
    return mod


fov = _load()
find_dropable_vs_tib = fov.find_dropable_vs_tib


def _rec(contig, strand, exons):
    return {
        "contig": contig, "strand": strand,
        "exons": sorted(exons), "lines": ["l"],
    }


def test_orf_dropped_when_strict_subseq_of_tib():
    orf = {"orf.1": _rec("chr1", "+", [(300, 400), (500, 600)])}
    tib = {"T.1": _rec("chr1", "+", [(100, 200), (300, 400),
                                     (500, 600), (700, 800)])}
    assert find_dropable_vs_tib(orf, tib) == {"orf.1": "T.1"}


def test_orf_kept_when_no_tib_contains_it():
    orf = {"orf.1": _rec("chr1", "+", [(300, 400), (500, 600)])}
    tib = {"T.1": _rec("chr1", "+", [(100, 250)])}  # doesn't overlap fully
    assert find_dropable_vs_tib(orf, tib) == {}


def test_strand_scoping():
    # ORF is + strand; Tiberius same coords but − strand → no drop.
    orf = {"orf.1": _rec("chr1", "+", [(300, 400), (500, 600)])}
    tib = {"T.1": _rec("chr1", "-", [(100, 200), (300, 400),
                                     (500, 600), (700, 800)])}
    assert find_dropable_vs_tib(orf, tib) == {}


def test_contig_scoping():
    orf = {"orf.1": _rec("chr1", "+", [(300, 400), (500, 600)])}
    tib = {"T.1": _rec("chr2", "+", [(100, 200), (300, 400),
                                     (500, 600), (700, 800)])}
    assert find_dropable_vs_tib(orf, tib) == {}


def test_orf_kept_when_it_extends_past_tib():
    # ORF has an extra CDS block Tiberius doesn't have → not a subseq.
    orf = {"orf.1": _rec("chr1", "+", [(300, 400), (500, 600), (900, 1000)])}
    tib = {"T.1": _rec("chr1", "+", [(100, 200), (300, 400),
                                     (500, 600), (700, 800)])}
    assert find_dropable_vs_tib(orf, tib) == {}


def test_multiple_tib_only_one_needs_to_contain():
    # T.1 doesn't contain ORF; T.2 does → dropped, keeper = T.2.
    orf = {"orf.1": _rec("chr1", "+", [(300, 400), (500, 600)])}
    tib = {
        "T.1": _rec("chr1", "+", [(100, 150), (170, 200)]),
        "T.2": _rec("chr1", "+", [(100, 200), (300, 400),
                                  (500, 600), (700, 800)]),
    }
    result = find_dropable_vs_tib(orf, tib)
    assert result == {"orf.1": "T.2"}


def test_tolerance_flags_relax_matching():
    # Terminal overhang: ORF's first block extends 30 nt past tib's.
    orf = {"orf.1": _rec("chr1", "+", [(270, 400), (500, 600)])}
    tib = {"T.1": _rec("chr1", "+", [(100, 200), (300, 400),
                                     (500, 600), (700, 800)])}
    assert find_dropable_vs_tib(orf, tib) == {}
    assert find_dropable_vs_tib(orf, tib,
                                terminal_overhang_nt=30) == {"orf.1": "T.1"}


def test_no_overlap_bucket_pruned_by_span_sort():
    # Verify the span_start early-break doesn't miss a valid match hidden
    # after a non-overlapping later Tiberius entry.
    orf = {"orf.1": _rec("chr1", "+", [(300, 400), (500, 600)])}
    tib = {
        "T_late": _rec("chr1", "+", [(2000, 3000)]),     # non-overlap, span_start > orf end
        "T_hit":  _rec("chr1", "+", [(100, 200), (300, 400),
                                     (500, 600), (700, 800)]),
    }
    assert find_dropable_vs_tib(orf, tib) == {"orf.1": "T_hit"}
