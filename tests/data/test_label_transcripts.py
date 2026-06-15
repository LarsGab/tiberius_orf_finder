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


def test_antisense_only_is_ir(tmp_path):
    # Only an opposite-strand ref overlaps -> emit all-IR labels (antisense
    # rescue). The transcript is kept in training, not dropped.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t100\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t20\t28\t.\t-\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"antisense_ir": 1}
    assert "ST.1" in res.labels
    assert (res.labels["ST.1"] == IR).all()
    # No ref was chosen and the transcript is not flagged as partial.
    assert "ST.1" not in res.chosen_ref
    assert "ST.1" not in res.partial


def test_partial_overlap_3prime_truncation_is_kept(tmp_path):
    # StringTie exon g 10..50 -> 0-based [9, 50), length 41, t=0..40.
    # Ref CDS g 40..60 on '+' -> 0-based [39, 60), 21 nt long (complete).
    # Visible portion on transcript: g 39..49 -> t = 30..40 (length 11,
    # ref positions 0..10). 3' end of ref CDS is NOT visible.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t50\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t40\t60\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_partial": 1}

    labels = res.labels["ST.1"]
    assert labels.shape == (41,)
    # Visible region starts at ref pos 0 -> START at t=30, then frame cycle.
    assert labels[30] == START
    assert list(labels[31:41]) == [E1, E2, E0, E1, E2, E0, E1, E2, E0, E1]
    # No STOP because the 3' end of the ref CDS is not visible.
    assert STOP not in labels
    # Flanking region is IR.
    assert (labels[:30] == IR).all()

    p = res.partial["ST.1"]
    assert p["has_5p"] is True
    assert p["has_3p"] is False
    assert (p["t_start"], p["t_end"]) == (30, 40)
    assert (p["ref_start_pos"], p["ref_end_pos"]) == (0, 10)
    assert p["ref_total_len"] == 21
    assert res.chosen_ref["ST.1"] == "R.1"


def test_partial_overlap_5prime_truncation_is_kept(tmp_path):
    # StringTie exon g 30..100 -> 0-based [29, 100), length 71, t=0..70.
    # Ref CDS g 10..39 on '+' -> 0-based [9, 39), 30 nt long (complete).
    # Visible portion: g 29..38 -> ref positions 20..29 (5' is cut off),
    # t = 0..9. 3' end of ref CDS IS visible -> STOP at t=9.
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t30\t100\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t10\t39\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_partial": 1}

    labels = res.labels["ST.1"]
    # ref pos 29 (the last base) -> STOP. ref pos 20..28 use the frame cycle.
    assert labels[9] == STOP
    # frame at t=k corresponds to ref_pos = 20 + k for k in 0..8.
    # _FRAME_CYCLE[(rp-1) % 3]: rp=20 -> FC[1]=E2, rp=21 -> FC[2]=E0, rp=22 -> FC[0]=E1, ...
    assert list(labels[0:9]) == [E2, E0, E1, E2, E0, E1, E2, E0, E1]
    assert (labels[10:] == IR).all()
    # No START because ref pos 0 is not visible.
    assert START not in labels

    p = res.partial["ST.1"]
    assert p["has_5p"] is False
    assert p["has_3p"] is True
    assert (p["ref_start_pos"], p["ref_end_pos"]) == (20, 29)
    assert p["ref_total_len"] == 30


def test_only_partial_refs_is_ir(tmp_path):
    # CDS length 10 (not divisible by 3) -> ref.complete == False.
    # No complete same-strand ref overlaps -> emit all-IR labels (the reading
    # frame is not derivable from a partial ref, so we don't fabricate one).
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t10\t100\t.\t+\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t20\t29\t.\t+\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"ref_partial_ir": 1}
    assert (res.labels["ST.1"] == IR).all()
    assert "ST.1" not in res.partial


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


def test_plus_strand_cds_spanning_nonmatching_intron_is_kept_partial(tmp_path):
    # StringTie single exon covers everything [0, 50).  Reference CDS has
    # two parts with a genomic intron (20..29) in between that StringTie
    # treats as part of the exon -> CDS is non-contiguous in transcript
    # coords. The partial rescue labels the longest contiguous run.
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
    res = project_labels(st, ref)
    assert res.stats == {"kept_partial": 1}

    labels = res.labels["ST.1"]
    # Two equal-length runs (t=10..18 ref pos 0..8; t=29..37 ref pos 9..17).
    # Tie-broken to the first (5'-anchored) run -> START at t=10.
    assert labels[10] == START
    assert list(labels[11:18]) == [E1, E2, E0, E1, E2, E0, E1]
    # ref pos 8 != ref_total-1 -> not STOP, just frame.
    assert labels[18] != STOP
    assert labels[18] != IR
    # The second half (real CDS in genomic terms) gets IR in transcript coords
    # because StringTie's missing splice breaks contiguity.
    assert (labels[19:] == IR).all()

    p = res.partial["ST.1"]
    assert p["has_5p"] is True
    assert p["has_3p"] is False


def test_minus_strand_partial_overlap_is_kept(tmp_path):
    # Minus-strand single exon g 1..30 -> 0-based [0, 30), t length 30.
    # Ref CDS on '-' strand at g 20..40 -> 0-based [19, 40), 21 nt (complete).
    # Visible portion: g 19..29 -> ref positions 10..20 (5' of ref cut off),
    # corresponding to t positions 0..10 (since t=29-g for single exon).
    st = tmp_path / "st.gtf"
    _write(st, [
        'chr1\tStringTie\texon\t1\t30\t.\t-\t.\ttranscript_id "ST.1";',
    ])
    ref = tmp_path / "ref.gtf"
    _write(ref, [
        'chr1\tRef\tCDS\t20\t40\t.\t-\t0\ttranscript_id "R.1";',
    ])
    res = project_labels(st, ref)
    assert res.stats == {"kept_partial": 1}

    p = res.partial["ST.1"]
    # 3' end of ref (last base) is at lowest genomic position; for a -strand
    # ref at g [19, 40) the last base is g=19 -> visible on this transcript
    # -> has_3p should be True; the 5' end (g=39) is not visible -> has_5p
    # should be False.
    assert p["has_5p"] is False
    assert p["has_3p"] is True
    assert p["ref_total_len"] == 21
    # The visible region must contain a STOP label at the 3' end.
    labels = res.labels["ST.1"]
    assert (labels == STOP).any()
    assert START not in labels
