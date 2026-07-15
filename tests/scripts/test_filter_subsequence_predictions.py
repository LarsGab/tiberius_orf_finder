"""Tests for scripts/filter_subsequence_predictions.py.

Covers strict-mode behavior plus the three near-subsequence tolerance
modes exercised by real STRG cases from the Bos_taurus test set.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
_MOD_PATH = REPO_ROOT / "scripts" / "filter_subsequence_predictions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "filter_subseq_pred", _MOD_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["filter_subseq_pred"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


fsp = _load_module()
is_subsequence = fsp.is_subsequence
find_dropable = fsp.find_dropable


def _tid(name, contig, strand, exons):
    return name, {
        "contig": contig, "strand": strand,
        "exons": sorted(exons), "lines": [f"line({name})"],
    }


# --- strict mode (defaults) -------------------------------------------------

def test_strict_single_exon_inside_multi_exon():
    a = [(100, 200)]
    b = [(50, 250), (400, 500)]
    assert is_subsequence(a, b)


def test_strict_exact_terminal_trim():
    # A trimmed inward at both ends, interior identical.
    b = [(100, 200), (300, 400), (500, 700)]
    a = [(150, 200), (300, 400), (500, 650)]
    assert is_subsequence(a, b)


def test_strict_rejects_overhang():
    # A extends past B on the left.
    b = [(100, 200), (300, 400)]
    a = [(80, 200), (300, 400)]
    assert not is_subsequence(a, b)


def test_strict_rejects_interior_shift():
    b = [(100, 200), (300, 400), (500, 600)]
    a = [(100, 200), (303, 400), (500, 600)]  # interior differs by 3 nt
    assert not is_subsequence(a, b)


def test_strict_rejects_exon_skip():
    # A skips B's middle exon.
    b = [(100, 200), (300, 400), (500, 600)]
    a = [(100, 200), (500, 600)]
    assert not is_subsequence(a, b)


# --- terminal-overhang tolerance -------------------------------------------

def test_overhang_left_extends_by_63nt():
    """STRG.638.3-like: A's leftmost extends 63 nt past B's."""
    b = [(82384546, 82384639), (82390046, 82390176),
         (82396036, 82396104), (82396904, 82397056)]
    a = [(82384483, 82384639), (82390046, 82390176),
         (82396036, 82396104), (82396904, 82397056)]
    assert not is_subsequence(a, b)                          # strict
    assert not is_subsequence(a, b, terminal_overhang_nt=60)  # too tight
    assert is_subsequence(a, b, terminal_overhang_nt=63)      # exact
    assert is_subsequence(a, b, terminal_overhang_nt=90)      # roomy


def test_overhang_right_extends_by_24nt():
    """STRG.703.1-like: A's rightmost ends 24 nt past B's."""
    b = [(83125893, 83126021), (83143332, 83143489),
         (83145857, 83146012), (83146645, 83146792)]
    a = [(83125893, 83126021), (83143332, 83143489),
         (83145857, 83146012), (83146645, 83146816)]
    assert not is_subsequence(a, b)
    assert is_subsequence(a, b, terminal_overhang_nt=24)


def test_overhang_does_not_relax_interior():
    # An interior shift should NOT be forgiven by terminal-overhang alone.
    b = [(100, 200), (300, 400), (500, 600)]
    a = [(100, 200), (303, 400), (500, 600)]
    assert not is_subsequence(a, b, terminal_overhang_nt=100)


# --- splice-shift tolerance -------------------------------------------------

def test_interior_shift_3nt_inframe():
    """STRG.624.4-like: one interior donor shifted by 3 nt."""
    b = [(81288416, 81288451), (81299736, 81299869),
         (81300924, 81301086), (81302409, 81302597),
         (81304467, 81304582), (81304889, 81304972),
         (81306181, 81306240), (81307010, 81307083),
         (81307561, 81307571)]
    a = [(81301054, 81301086), (81302409, 81302597),
         (81304467, 81304582), (81304889, 81304972),
         (81306181, 81306240), (81307013, 81307083),  # start +3 nt
         (81307561, 81307571)]
    assert not is_subsequence(a, b)
    assert not is_subsequence(a, b, splice_shift_nt=2)  # 3 nt exceeds 2
    assert is_subsequence(a, b, splice_shift_nt=3)
    assert is_subsequence(a, b, splice_shift_nt=6)


def test_interior_shift_out_of_frame_rejected():
    b = [(100, 200), (300, 400), (500, 600)]
    a = [(100, 200), (302, 400), (500, 600)]  # 2 nt shift, not in-frame
    assert not is_subsequence(a, b, splice_shift_nt=6)


# --- exon-skip tolerance ----------------------------------------------------

def test_skip_middle_exon():
    """STRG.632.2-like: A skips one interior B exon."""
    b = [(81877905, 81877978), (81885715, 81885818),
         (81893940, 81894112), (81898498, 81898609),
         (81907711, 81907815), (81919801, 81920142),
         (81929417, 81930678)]
    a = [(81885801, 81885818), (81898498, 81898609),   # skips B[2]
         (81907711, 81907815), (81919801, 81920142),
         (81929417, 81930678)]
    assert not is_subsequence(a, b)
    assert is_subsequence(a, b, allow_exon_skip=True)


def test_skip_does_not_admit_extension_past_b():
    # A extends past B on the right → still rejected under skip mode.
    b = [(100, 200), (300, 400), (500, 600)]
    a = [(100, 200), (500, 700)]  # end 700 > 600
    assert not is_subsequence(a, b, allow_exon_skip=True)


def test_skip_combined_with_overhang_and_shift():
    # Realistic combined case: A skips one B exon AND has 3 nt interior
    # shift AND terminal overhang.
    b = [(100, 200), (300, 400), (500, 600), (700, 800), (900, 1000)]
    a = [(150, 200), (303, 400), (700, 800), (900, 1020)]  # skips B[2], shift+3, +20 overhang
    assert not is_subsequence(a, b)
    assert is_subsequence(
        a, b,
        terminal_overhang_nt=20,
        splice_shift_nt=3,
        allow_exon_skip=True,
    )


# --- find_dropable integration ---------------------------------------------

def test_find_dropable_default_strict_matches_prior_behavior():
    by_tid = dict([
        _tid("g.1", "chr1", "+", [(100, 200)]),                     # inside g.2
        _tid("g.2", "chr1", "+", [(50, 300), (400, 500)]),          # keeper
        _tid("g.3", "chr1", "+", [(1000, 2000)]),                    # elsewhere
    ])
    drop = find_dropable(by_tid)
    assert drop == {"g.1": "g.2"}


def test_find_dropable_locus_scoping_by_gene_prefix():
    # Different locus prefix → do not compare across.
    by_tid = dict([
        _tid("gA.1", "chr1", "+", [(100, 200)]),
        _tid("gB.1", "chr1", "+", [(50, 300)]),
    ])
    assert find_dropable(by_tid) == {}


def test_find_dropable_strand_scoping():
    by_tid = dict([
        _tid("g.1", "chr1", "+", [(100, 200)]),
        _tid("g.2", "chr1", "-", [(50, 300)]),
    ])
    assert find_dropable(by_tid) == {}


def test_find_dropable_with_all_tolerances_drops_more():
    """Same input evaluated under strict vs. lenient tolerances."""
    by_tid = dict([
        _tid("g.1", "chr1", "+", [(300, 400), (500, 600)]),
        _tid("g.2", "chr1", "+", [(100, 200), (300, 400),
                                    (500, 600), (700, 800)]),
    ])
    # g.1 is a strict subseq of g.2 (interior/terminal exactly).
    strict = find_dropable(by_tid)
    assert "g.1" in strict and strict["g.1"] == "g.2"

    # Perturb g.1 to require tolerances.
    by_tid_p = dict(by_tid)
    by_tid_p["g.1"] = {
        "contig": "chr1", "strand": "+",
        "exons": [(303, 400), (500, 620)],  # 3nt interior shift + 20nt overhang
        "lines": ["line(g.1)"],
    }
    assert find_dropable(by_tid_p) == {}
    assert find_dropable(
        by_tid_p,
        terminal_overhang_nt=20,
        splice_shift_nt=3,
    ) == {"g.1": "g.2"}
