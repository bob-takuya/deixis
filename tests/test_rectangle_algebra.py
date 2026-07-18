"""Tests for Rectangle Algebra (2-D product of Allen's interval algebra).

Coverage:
  * rect_from_boxes extracts exact per-axis Allen relations, including face-area
    overlap, edge (line) contact, and corner (point) contact;
  * converse is component-wise Allen converse and matches box-order reversal;
  * compose is component-wise Allen composition and is a sound superset for
    three concrete boxes;
  * rcc8_from_rect reproduces reverse.relation_between_boxes / extract_relations
    base-for-base over an exhaustive grid of box pairs.
"""
from __future__ import annotations

from fractions import Fraction as Fr
from itertools import product

import pytest

from deixis.core import rcc8
from deixis.core.types import Box, Realization
from deixis.solver import allen
from deixis.solver import rectangle_algebra as ra
from deixis.verify import reverse


def box(rid: str, x0, y0, x1, y1) -> Box:
    return Box(region_id=rid,
               lo=(Fr(x0), Fr(y0)),
               hi=(Fr(x1), Fr(y1)))


# ---------------------------------------------------------------- rect_from_boxes

def test_rect_from_boxes_area_overlap_is_two_axis_overlap():
    # A starts before/ends inside B on both axes -> (o, o), which is PO.
    a = box("A", 0, 0, 2, 2)
    b = box("B", 1, 1, 3, 3)
    rel = ra.rect_from_boxes(a, b)
    assert rel == (allen.bit("o"), allen.bit("o"))
    assert rel.allen_x_mask == allen.bit("o")
    assert rel.allen_y_mask == allen.bit("o")
    assert ra.rcc8_from_rect(rel) == rcc8.bit("PO")


def test_rect_from_boxes_edge_contact_line():
    # Shared vertical edge at x=2: x-axis meets, y-axis overlaps -> EC (line).
    a = box("A", 0, 0, 2, 2)
    b = box("B", 2, 1, 4, 3)
    rel = ra.rect_from_boxes(a, b)
    assert rel.allen_x_mask == allen.bit("m")
    assert rel.allen_y_mask == allen.bit("o")
    assert ra.rcc8_from_rect(rel) == rcc8.bit("EC")


def test_rect_from_boxes_corner_contact_point():
    # Touch at the single corner (2,2): both axes meet -> EC (point).
    a = box("A", 0, 0, 2, 2)
    b = box("B", 2, 2, 4, 4)
    rel = ra.rect_from_boxes(a, b)
    assert rel.allen_x_mask == allen.bit("m")
    assert rel.allen_y_mask == allen.bit("m")
    assert ra.rcc8_from_rect(rel) == rcc8.bit("EC")


def test_rect_from_boxes_separated_is_dc():
    a = box("A", 0, 0, 1, 1)
    b = box("B", 5, 5, 6, 6)
    rel = ra.rect_from_boxes(a, b)
    assert rel.allen_x_mask == allen.bit("b")
    assert rel.allen_y_mask == allen.bit("b")
    assert ra.rcc8_from_rect(rel) == rcc8.bit("DC")


def test_rect_from_boxes_equal():
    a = box("A", 0, 0, 2, 2)
    b = box("B", 0, 0, 2, 2)
    rel = ra.rect_from_boxes(a, b)
    assert rel == (allen.bit("eq"), allen.bit("eq"))
    assert ra.rcc8_from_rect(rel) == rcc8.bit("EQ")


def test_rect_from_boxes_ntpp_strict_inside():
    # A strictly inside B on both axes -> (d, d) -> NTPP.
    a = box("A", 1, 1, 2, 2)
    b = box("B", 0, 0, 5, 5)
    rel = ra.rect_from_boxes(a, b)
    assert rel == (allen.bit("d"), allen.bit("d"))
    assert ra.rcc8_from_rect(rel) == rcc8.bit("NTPP")


def test_rect_from_boxes_tpp_shared_boundary():
    # A inside B but sharing the left edge -> x-axis starts (s), y-axis during (d) -> TPP.
    a = box("A", 0, 1, 2, 2)
    b = box("B", 0, 0, 5, 5)
    rel = ra.rect_from_boxes(a, b)
    assert rel == (allen.bit("s"), allen.bit("d"))
    assert ra.rcc8_from_rect(rel) == rcc8.bit("TPP")


def test_rect_from_boxes_rejects_non_2d():
    a = Box("A", (Fr(0), Fr(0), Fr(0)), (Fr(1), Fr(1), Fr(1)))
    b = box("B", 0, 0, 1, 1)
    with pytest.raises(ValueError):
        ra.rect_from_boxes(a, b)


def test_rect_from_boxes_rejects_float():
    a = Box("A", (0.0, 0.0), (1.0, 1.0))  # float bounds, not Fraction
    b = box("B", 0, 0, 1, 1)
    with pytest.raises(TypeError):
        ra.rect_from_boxes(a, b)


# ---------------------------------------------------------------- converse

def test_converse_is_componentwise_allen():
    a = box("A", 0, 0, 2, 2)
    b = box("B", 1, 1, 3, 3)
    rel = ra.rect_from_boxes(a, b)
    conv = ra.converse(rel)
    assert conv.allen_x_mask == allen.converse(rel.allen_x_mask)
    assert conv.allen_y_mask == allen.converse(rel.allen_y_mask)


def test_converse_matches_box_order_reversal():
    a = box("A", 0, 0, 2, 3)
    b = box("B", 1, 1, 5, 2)
    fwd = ra.rect_from_boxes(a, b)
    rev = ra.rect_from_boxes(b, a)
    assert ra.converse(fwd) == rev


def test_converse_projects_to_rcc8_converse():
    # rcc8_from_rect(converse(r)) == rcc8.converse(rcc8_from_rect(r)) for base rects.
    a = box("A", 0, 1, 2, 2)
    b = box("B", 0, 0, 5, 5)
    rel = ra.rect_from_boxes(a, b)  # TPP
    assert ra.rcc8_from_rect(ra.converse(rel)) == rcc8.converse(ra.rcc8_from_rect(rel))


# ---------------------------------------------------------------- compose

def test_compose_is_componentwise_allen():
    r1 = ra.make_rect(allen.bit("b"), allen.bit("o"))
    r2 = ra.make_rect(allen.bit("d"), allen.bit("m"))
    comp = ra.compose(r1, r2)
    assert comp.allen_x_mask == allen.compose(allen.bit("b"), allen.bit("d"))
    assert comp.allen_y_mask == allen.compose(allen.bit("o"), allen.bit("m"))


def test_compose_is_sound_superset_for_three_boxes():
    # For concrete A,B,C the realized A-rel-C must lie inside the composed mask,
    # per axis (weak composition is a sound over-approximation).
    a = box("A", 0, 0, 2, 2)
    b = box("B", 1, 1, 4, 5)
    c = box("C", 3, 2, 6, 3)
    ab = ra.rect_from_boxes(a, b)
    bc = ra.rect_from_boxes(b, c)
    ac = ra.rect_from_boxes(a, c)
    comp = ra.compose(ab, bc)
    assert ac.allen_x_mask & comp.allen_x_mask == ac.allen_x_mask
    assert ac.allen_y_mask & comp.allen_y_mask == ac.allen_y_mask


# ---------------------------------------------------------------- rcc8 agreement

def _coords():
    """A small set of 1-D intervals hitting before/meet/overlap/during/eq/etc."""
    return [(Fr(a), Fr(b)) for a, b in [
        (0, 2), (1, 3), (2, 4), (0, 4), (1, 2), (0, 1), (2, 3), (3, 5),
    ]]


def test_rcc8_from_rect_matches_reverse_relation_between_boxes():
    ivs = _coords()
    checked = 0
    seen_rels = set()
    for (ax0, ax1), (ay0, ay1), (bx0, bx1), (by0, by1) in product(ivs, repeat=4):
        a = Box("A", (ax0, ay0), (ax1, ay1))
        b = Box("B", (bx0, by0), (bx1, by1))
        expected = reverse.relation_between_boxes(a, b)
        got = ra.rcc8_from_rect(ra.rect_from_boxes(a, b))
        assert got == expected, (a.lo, a.hi, b.lo, b.hi,
                                 rcc8.names(got), rcc8.names(expected))
        seen_rels.add(expected)
        checked += 1
    assert checked > 0
    # the grid is rich enough to exercise several distinct RCC-8 base relations.
    assert len(seen_rels) >= 5


def test_rcc8_from_rect_matches_extract_relations():
    a = box("A", 0, 0, 2, 2)
    b = box("B", 1, 1, 3, 3)
    c = box("C", 5, 5, 6, 6)
    realization = Realization(id="r", boxes=(a, b, c))
    extracted = reverse.extract_relations(realization)
    for (ra_id, rb_id), expected in extracted.items():
        boxes = {x.region_id: x for x in (a, b, c)}
        got = ra.rcc8_from_rect(ra.rect_from_boxes(boxes[ra_id], boxes[rb_id]))
        assert got == expected


def test_rcc8_from_rect_disjunction_unions_bases():
    # A disjunction on one axis unions the RCC-8 projections. (b|bi, eq): either
    # axis-separation direction -> both give DC.
    rel = ra.make_rect(allen.mask("b", "bi"), allen.bit("eq"))
    assert ra.rcc8_from_rect(rel) == rcc8.bit("DC")


def test_rcc8_from_rect_empty_axis_is_empty():
    rel = ra.make_rect(allen.EMPTY, allen.bit("eq"))
    assert ra.rcc8_from_rect(rel) == rcc8.EMPTY


# ---------------------------------------------------------------- validation

def test_make_rect_rejects_out_of_range():
    with pytest.raises(ValueError):
        ra.make_rect(allen.UNIVERSAL + 1, allen.bit("eq"))
    with pytest.raises(ValueError):
        ra.make_rect(allen.bit("eq"), -1)


def test_make_rect_rejects_bool():
    with pytest.raises(TypeError):
        ra.make_rect(True, allen.bit("eq"))
