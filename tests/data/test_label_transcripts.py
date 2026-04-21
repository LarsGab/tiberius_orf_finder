"""Unit tests for label projection.

Conventions in fixtures:
* GTF is 1-based inclusive.
* Sequence labels: IR=0, START=1, E1=2, E2=3, E0=4, STOP=5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tiberius_orf.data.label_transcripts import (
    E0,
    E1,
    E2,
    IR,
    START,
    STOP,
    build_labels,
    parse_reference_cds,
    parse_stringtie_gtf,
    project_labels,
)


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- build_labels ----------

def test_build_labels_example_from_spec():
    # A 12 nt "transcript" with a 9 nt ORF starting at position 2:
    #   positions:   0 1 2     3 4 5 6 7 8 9 10   11
    #   labels   :   IR IR START E1 E2 E0 E1 E2 E0 E1 STOP IR
    labels = build_labels(12, orf=(2, 10))
    expected = np.array([
        IR, IR, START, E1, E2, E0, E1, E2, E0, E1, STOP, IR,
    ], dtype=np.int8)
    assert np.array_equal(labels, expected)


def test_build_labels_all_ir():
    labels = build_labels(5)
    assert np.array_equal(labels, np.zeros(5, dtype=np.int8))


# ---------- parsing ----------

def test_parse_stringtie_gtf_collects_exons(tmp_path):
    gtf = tmp_path / "st.gtf"
    _write(gtf, [
        'chr1\tStringTie\ttranscript\t10\t200\t.\t+\t.\ttranscript_id "ST.1";',
        'chr1\tStringTie\texon\t10\t50\t.\t+\t.\ttranscript_id "ST.1";',
        'chr1\tStringTie\texon\t150\t200\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    st = parse_stringtie_gtf(gtf)
    assert list(st) == ["ST.1"]
    t = st["ST.1"]
    assert t.contig == "chr1"
    assert t.strand == "+"
    assert t.exons == [(9, 50), (149, 200)]
    assert t.length == (50 - 9) + (200 - 149)


def test_parse_reference_cds_merges_stop_codon(tmp_path):
    gff = tmp_path / "ref.gtf"
    _write(gff, [
        'chr1\tRef\tCDS\t100\t150\t.\t+\t0\ttranscript_id "R.1";',
        'chr1\tRef\tstop_codon\t151\t153\t.\t+\t0\ttranscript_id "R.1";',
    ])
    refs = parse_reference_cds(gff)
    r = refs["R.1"]
    assert r.cds_parts == [(99, 153)]
    assert r.has_stop_codon is True
    assert r.total_cds_len == 54
    assert r.complete  # 54 % 3 == 0, length >= 6


# ---------- end-to-end projection ----------

@pytest.fixture
def simple_plus_fixture(tmp_path):
    """A 100 nt single-exon StringTie transcript on chr1 '+'; a contained
    reference CDS of length 9 (1 ATG + 1 codon + 1 stop) at positions 50..58
    in genomic 1-based inclusive, which is 49..57 in 0-based inclusive, which
    is transcript coords 39..47 (exon starts at 10).
    """
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t109\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        # 9 nt CDS including stop: ATG + 1 codon + stop => 9 bases total
        'chr1\tRef\tCDS\t50\t58\t.\t+\t0\ttranscript_id "R.1";',
    ])
    return st, ref


def test_simple_plus_strand_contained(simple_plus_fixture):
    st, ref = simple_plus_fixture
    res = project_labels(st, ref)

    assert res.stats == {"kept_single": 1}
    labels = res.labels["ST.1"]
    assert labels.shape == (100,)

    # Exon starts at g=9 (0-based). CDS at g=49..57 -> t=40..48.
    orf_s, orf_e = 40, 48
    assert labels[orf_s] == START
    assert labels[orf_e] == STOP
    # Frame cycle between them: E1 E2 E0 E1 E2 E0 E1
    assert list(labels[orf_s + 1: orf_e]) == [E1, E2, E0, E1, E2, E0, E1]
    # Everything else is IR.
    assert (labels[:orf_s] == IR).all()
    assert (labels[orf_e + 1:] == IR).all()
    assert res.chosen_ref["ST.1"] == "R.1"


def test_no_overlap_is_ir_only(tmp_path):
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t50\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t1000\t1008\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"ir_only": 1}
    assert (res.labels["ST.1"] == IR).all()


def test_antisense_only_is_dropped(tmp_path):
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t100\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t20\t28\t.\t-\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"dropped_antisense_only": 1}
    assert "ST.1" not in res.labels


def test_partial_overlap_is_dropped(tmp_path):
    # StringTie exon 10..50. Reference CDS 40..60 (extends past the StringTie
    # exon's 3' end) -> not fully contained, should be dropped.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t50\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t40\t60\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"dropped_not_contained": 1}


def test_ref_partial_cds_length_is_dropped(tmp_path):
    # CDS length 10 is not divisible by 3 -> ref.complete == False
    # -> classified as dropped_ref_partial.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t100\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t20\t29\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"dropped_ref_partial": 1}


def test_multi_hit_longest_chosen(tmp_path):
    # Two contained ref CDSes: R.short (9 nt) and R.long (15 nt).
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t1\t200\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t10\t18\t.\t+\t0\ttranscript_id "R.short";',
        'chr1\tRef\tCDS\t100\t114\t.\t+\t0\ttranscript_id "R.long";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_multi": 1}
    assert res.chosen_ref["ST.1"] == "R.long"


def test_minus_strand_contained(tmp_path):
    # Single-exon minus-strand StringTie transcript on chr1, genomic 1..30.
    # 0-based: exon [0, 30), transcript length 30.  strand '-' => transcript
    # coordinate 0 corresponds to genomic position 29.
    # Reference CDS at genomic 10..18 (1-based incl), i.e. 0-based [9, 18).
    # On the transcript, the CDS maps to t-positions 29-9=20 down to 29-17=12,
    # sorted ascending that's 12..20 inclusive -> orf_start=12, orf_end=20.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t1\t30\t.\t-\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t10\t18\t.\t-\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_single": 1}
    labels = res.labels["ST.1"]
    assert labels[12] == START
    assert labels[20] == STOP
    assert list(labels[13:20]) == [E1, E2, E0, E1, E2, E0, E1]


def test_plus_strand_cds_spanning_matching_intron(tmp_path):
    # StringTie has two exons: [0, 20) and [30, 50) (genomic 0-based).
    # Reference CDS has two parts: [10, 20) and [30, 40).  Their introns
    # match, so in transcript coords the two CDS parts are contiguous
    # (t: 10..19 and 20..29 -> ORF from t=10 to t=29, length 20... but
    # 20 is not divisible by 3, so for this test we shrink to 18).
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t1\t20\t.\t+\t.\ttranscript_id "ST.1";',
        'chr1\tStringTie\texon\t31\t50\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        # CDS parts: g [10, 20) (len 10) and g [30, 38) (len 8) -> total 18.
        # Ref intron [20, 30) == StringTie intron [20, 30), so contiguous in t.
        'chr1\tRef\tCDS\t11\t20\t.\t+\t0\ttranscript_id "R.1";',
        'chr1\tRef\tCDS\t31\t38\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_single": 1}
    labels = res.labels["ST.1"]
    # Transcript length = 20 + 20 = 40.  First-exon CDS starts at t=10
    # (g=10 lies at exon-1 offset 10).  ORF length 18 -> ends at t=27.
    assert labels[10] == START
    assert labels[27] == STOP
    # Check contiguity: all positions between must be coding, not IR.
    assert (labels[10:28] != IR).all()


def test_plus_strand_cds_spanning_nonmatching_intron_is_dropped(tmp_path):
    # StringTie single exon covers everything [0, 50).  Reference CDS has
    # two parts with a genomic intron (20..29) in between that StringTie
    # treats as part of the exon -> CDS would be non-contiguous in
    # transcript coords -> dropped.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t1\t50\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        # Parts g [10, 19) + g [29, 38); total 18 (%3==0, complete).
        'chr1\tRef\tCDS\t11\t19\t.\t+\t0\ttranscript_id "R.1";',
        'chr1\tRef\tCDS\t30\t38\t.\t+\t0\ttranscript_id "R.1";',
    ])
    # Total 18, complete=True; but projection will still be contiguous here
    # because StringTie treats the intron as part of the exon, so in
    # transcript coords t=10..18 and t=29..36 are NOT contiguous (gap at
    # t=19..28 where ref had an intron but StringTie has exon).
    res = project_labels(st, ref)
    assert res.stats == {"dropped_not_contained": 1}
