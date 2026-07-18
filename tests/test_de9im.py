"""Tests for the DE-9IM completion (deixis.verify.de9im).

We build exact-rational axis-aligned boxes for the canonical contact cases
(face / edge / corner / disjoint / containment / equal), check the 3x3
intersection-dimension matrix and its pattern string, and verify that
``rcc8_from_de9im`` agrees box-for-box with the independent RCC-8 extraction in
``deixis.verify.reverse`` (including its multi-pair ``extract_relations``).
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from deixis.core.types import Box, Realization
from deixis.core import rcc8
from deixis.verify.de9im import (
    DE9IM_LABELS,
    EMPTY_DIM,
    de9im_matrix,
    de9im_string,
    rcc8_from_de9im,
)
from deixis.verify.reverse import extract_relations, relation_between_boxes


# ------------------------------------------------------------------ builders

def _box(rid, lo, hi):
    return Box(region_id=rid,
               lo=tuple(Fraction(v) for v in lo),
               hi=tuple(Fraction(v) for v in hi))


def _rc(matrix):
    return rcc8.names(rcc8_from_de9im(matrix))


# ------------------------------------------------------------------ canonical 2D contacts

def test_equal_2d():
    A = _box("A", (0, 0), (2, 2))
    B = _box("B", (0, 0), (2, 2))
    m = de9im_matrix(A, B, 2)
    assert de9im_string(m) == "2FFF1FFF2"
    # interior/interior = area, boundary/boundary = line, everything cross = F.
    assert m == (2, -1, -1, -1, 1, -1, -1, -1, 2)
    assert _rc(m) == ["EQ"]


def test_face_contact_2d_is_edge_share():
    # two unit squares sharing the x=1 edge: EC, boundaries meet along a 1D line.
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (1, 0), (2, 1))
    m = de9im_matrix(A, B, 2)
    II, IB, IE, BI, BB, BE, EI, EB, EE = m
    assert II == EMPTY_DIM            # interiors disjoint
    assert BB == 1                    # shared boundary is a 1D edge
    assert IE == 2 and EI == 2 and EE == 2
    assert _rc(m) == ["EC"]


def test_edge_contact_2d_corner_point():
    # squares meeting only at the corner (1,1): EC, boundaries meet at a 0D point.
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (1, 1), (2, 2))
    m = de9im_matrix(A, B, 2)
    assert m[0] == EMPTY_DIM          # II empty
    assert m[4] == 0                  # BB is a single point
    assert _rc(m) == ["EC"]


def test_disjoint_2d():
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (5, 5), (6, 6))
    m = de9im_matrix(A, B, 2)
    # interiors AND boundaries all disjoint -> DC
    II, IB, IE, BI, BB, BE, EI, EB, EE = m
    assert II == IB == BI == BB == EMPTY_DIM
    assert EE == 2
    assert _rc(m) == ["DC"]
    assert de9im_string(m) == "FF2FF1212"


def test_partial_overlap_2d_matches_reference_string():
    # the textbook overlapping-rectangles pattern.
    A = _box("A", (0, 0), (2, 2))
    B = _box("B", (1, 1), (3, 3))
    m = de9im_matrix(A, B, 2)
    assert de9im_string(m) == "212101212"
    assert _rc(m) == ["PO"]


def test_containment_ntpp_2d():
    # A strictly inside B: non-tangential proper part (boundaries do not meet).
    A = _box("A", (1, 1), (2, 2))
    B = _box("B", (0, 0), (3, 3))
    m = de9im_matrix(A, B, 2)
    II, IB, IE, BI, BB, BE, EI, EB, EE = m
    assert II == 2 and IE == EMPTY_DIM and EI == 2
    assert BB == EMPTY_DIM             # no shared boundary -> non-tangential
    assert BI == 1                     # A boundary lies in B interior
    assert _rc(m) == ["NTPP"]
    # converse is NTPPi
    assert rcc8.names(rcc8_from_de9im(de9im_matrix(B, A, 2))) == ["NTPPi"]


def test_containment_tpp_2d():
    # A inside B but sharing the full x-extent: tangential proper part.
    A = _box("A", (0, 1), (4, 3))
    B = _box("B", (0, 0), (4, 4))
    m = de9im_matrix(A, B, 2)
    II, IB, IE, BI, BB, BE, EI, EB, EE = m
    assert II == 2 and IE == EMPTY_DIM and EI == 2
    assert BB != EMPTY_DIM              # shared boundary -> tangential
    assert _rc(m) == ["TPP"]
    assert rcc8.names(rcc8_from_de9im(de9im_matrix(B, A, 2))) == ["TPPi"]


# ------------------------------------------------------------------ 3D dimensional extension

def test_face_share_3d():
    # unit cubes sharing the z=1 face: EC with a 2D shared boundary; exterior∩exterior=3.
    A = _box("A", (0, 0, 0), (1, 1, 1))
    B = _box("B", (0, 0, 1), (1, 1, 2))
    m = de9im_matrix(A, B, 3)
    II, IB, IE, BI, BB, BE, EI, EB, EE = m
    assert II == EMPTY_DIM
    assert BB == 2                     # shared face is 2D
    assert IE == 3 and EI == 3 and EE == 3
    assert de9im_string(m) == "FF3F22323"
    assert _rc(m) == ["EC"]


def test_edge_share_3d():
    # cubes sharing only the z-axis edge at x=1,y=1: EC, boundaries meet on a 1D edge.
    A = _box("A", (0, 0, 0), (1, 1, 1))
    B = _box("B", (1, 1, 0), (2, 2, 1))
    m = de9im_matrix(A, B, 3)
    assert m[0] == EMPTY_DIM
    assert m[4] == 1                   # BB is a 1D edge
    assert _rc(m) == ["EC"]


def test_corner_share_3d():
    # cubes meeting only at a corner: EC, boundaries meet at a 0D point.
    A = _box("A", (0, 0, 0), (1, 1, 1))
    B = _box("B", (1, 1, 1), (2, 2, 2))
    m = de9im_matrix(A, B, 3)
    assert m[0] == EMPTY_DIM
    assert m[4] == 0
    assert _rc(m) == ["EC"]


def test_equal_3d_volume():
    A = _box("A", (0, 0, 0), (1, 1, 1))
    B = _box("B", (0, 0, 0), (1, 1, 1))
    m = de9im_matrix(A, B, 3)
    assert m[0] == 3                   # interior/interior is a 3D volume
    assert m[4] == 2                   # boundary/boundary is the 2D shell
    assert m[8] == 3
    assert _rc(m) == ["EQ"]


# ------------------------------------------------------------------ string / labels

def test_string_symbols_and_labels():
    m = (2, -1, 0, 1, -1, 3, -1, 2, -1)
    # -1 -> 'F', digits verbatim (incl. the 3 from a 3D dimensional extension).
    assert de9im_string(m) == "2F01F3F2F"
    assert DE9IM_LABELS == ("II", "IB", "IE", "BI", "BB", "BE", "EI", "EB", "EE")


def test_string_rejects_bad_length():
    with pytest.raises(ValueError):
        de9im_string((1, 2, 3))


# ------------------------------------------------------------------ consistency with reverse

def _interval_cases():
    """Exhaustive integer-endpoint 1-axis configs of A=[0,4] vs B, spanning every
    qualitative interval relation (same grid style as test_reverse)."""
    A0, A1 = Fraction(0), Fraction(4)
    out = []
    for b0 in range(-2, 7):
        for b1 in range(b0 + 1, 8):
            out.append(((A0, A1), (Fraction(b0), Fraction(b1))))
    return out


def test_rcc8_from_de9im_matches_relation_between_boxes_2d():
    """Over an exhaustive 2D grid, rcc8_from_de9im(de9im_matrix(A,B)) equals the
    independent relation_between_boxes(A,B) — the 9-IM completion is consistent
    with the RCC-8 extraction, and its converse holds."""
    intervals = _interval_cases()
    seen = set()
    for (ax, bx) in intervals:
        for (ay, by) in intervals:
            A = Box("A", (ax[0], ay[0]), (ax[1], ay[1]))
            B = Box("B", (bx[0], by[0]), (bx[1], by[1]))
            mask_ref = relation_between_boxes(A, B)
            mask_9im = rcc8_from_de9im(de9im_matrix(A, B, 2))
            assert rcc8.is_base(mask_9im)
            assert mask_9im == mask_ref, (
                rcc8.names(mask_ref), rcc8.names(mask_9im),
                de9im_string(de9im_matrix(A, B, 2)))
            # converse consistency through the 9-IM as well
            mask_ba = rcc8_from_de9im(de9im_matrix(B, A, 2))
            assert rcc8.converse(mask_9im) == mask_ba
            seen.add(rcc8.names(mask_9im)[0])
    assert seen == set(rcc8.BASE)      # grid exhibits all 8 base relations


def test_rcc8_from_de9im_matches_relation_between_boxes_3d():
    """A smaller exhaustive 3D grid: still agrees with relation_between_boxes."""
    vals = [(-2, 0), (0, 4), (0, 2), (2, 6), (4, 6), (1, 3), (0, 6)]
    ivs = [(Fraction(a), Fraction(b)) for (a, b) in vals]
    for ax in ivs:
        for ay in ivs:
            for az in ivs:
                A = Box("A", (Fraction(0), Fraction(0), Fraction(0)),
                        (Fraction(4), Fraction(4), Fraction(4)))
                B = Box("B", (ax[0], ay[0], az[0]), (ax[1], ay[1], az[1]))
                mask_ref = relation_between_boxes(A, B)
                mask_9im = rcc8_from_de9im(de9im_matrix(A, B, 3))
                assert mask_9im == mask_ref, (
                    rcc8.names(mask_ref), rcc8.names(mask_9im),
                    de9im_string(de9im_matrix(A, B, 3)))


def test_agrees_with_extract_relations_multipair():
    """rcc8_from_de9im matches reverse.extract_relations for every ordered pair
    of a multi-region realization."""
    A = _box("A", (0, 0), (2, 2))
    B = _box("B", (1, 1), (3, 3))     # PO with A
    C = _box("C", (10, 10), (11, 11))  # DC from both
    real = Realization(id="r", boxes=(A, B, C))
    rels = extract_relations(real)
    bm = {b.region_id: b for b in real.boxes}
    for (i, j), mask_ref in rels.items():
        m = de9im_matrix(bm[i], bm[j], 2)
        assert rcc8_from_de9im(m) == mask_ref, (i, j, rcc8.names(mask_ref))


def test_fractional_coordinates_exact():
    # thirds are not float-representable; the matrix must stay exact.
    A = _box("A", (Fraction(0), Fraction(0)), (Fraction(1, 3), Fraction(1)))
    B = _box("B", (Fraction(1, 3), Fraction(0)), (Fraction(2, 3), Fraction(1)))
    m = de9im_matrix(A, B, 2)
    assert m[0] == EMPTY_DIM and m[4] == 1   # EC via shared 1D edge
    assert _rc(m) == ["EC"]


# ------------------------------------------------------------------ validation errors

def test_matrix_rejects_wrong_arity():
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (0, 0), (1, 1))
    with pytest.raises(ValueError):
        de9im_matrix(A, B, 3)          # boxes are 2D


def test_matrix_rejects_degenerate_box():
    A = _box("A", (0, 0), (0, 1))       # zero width on axis 0
    B = _box("B", (1, 0), (2, 1))
    with pytest.raises(ValueError):
        de9im_matrix(A, B, 2)


def test_matrix_rejects_non_fraction_bounds():
    A = Box("A", (0.0, 0.0), (1.0, 1.0))  # floats, not Fraction
    B = _box("B", (1, 0), (2, 1))
    with pytest.raises(TypeError):
        de9im_matrix(A, B, 2)
