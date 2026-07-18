"""Tests for reverse-verification (deixis.verify.reverse).

We drive the forward solver (solve_boxes) to produce exact-rational realizations,
reverse-extract their RCC-8 relations / contact dimensions, and check round-trip
agreement, plus the central Witnessed-IR claim: an EC realization whose contact is
an *edge* while the witness demanded a *face* is caught as a witness violation.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from deixis.core.types import (
    Box,
    ContactDim,
    IncidenceWitness,
    Realizability,
    Realization,
    Region,
    RelationConstraint,
    RelSpec,
)
from deixis.core import rcc8
from deixis.solver.geom_solver import solve_boxes
from deixis.verify.reverse import (
    extract_contact_dimension,
    extract_relations,
    relation_between_boxes,
    verify,
    WitnessViolation,
)


# ------------------------------------------------------------------ builders

def _spec(regions, constraints, witnesses=()):
    return RelSpec(
        regions=tuple(Region(id=r) for r in regions),
        constraints=tuple(constraints),
        witnesses=tuple(witnesses),
    )


def _c(src, dst, *names, cid=None):
    return RelationConstraint(
        src=src, dst=dst, rcc8_mask=rcc8.mask(*names), id=cid or f"{src}-{dst}",
    )


def _box(rid, lo, hi):
    return Box(region_id=rid,
               lo=tuple(Fraction(v) for v in lo),
               hi=tuple(Fraction(v) for v in hi))


# ------------------------------------------------------------------ round-trip: every base relation

@pytest.mark.parametrize("rel", ["DC", "EC", "PO", "TPP", "NTPP", "TPPi", "NTPPi", "EQ"])
def test_roundtrip_base_relation_2d(rel):
    """solve_boxes(rel) -> extract_relations recovers exactly that base relation."""
    spec = _spec(["A", "B"], [_c("A", "B", rel)])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 20))
    assert real.status == Realizability.REALIZED_IN_D
    rels = extract_relations(real)
    got = rels[("A", "B")]
    assert rcc8.is_base(got)
    assert rcc8.names(got) == [rel], (rel, rcc8.names(got))
    # forward-verify agrees: the constraint is satisfied by its own realization.
    rep = verify(spec, real)
    assert rep.satisfied == ("A-B",)
    assert rep.violated == () and rep.undecided == ()


def test_converse_is_consistent():
    """relation_between_boxes(A,B) is the converse of relation_between_boxes(B,A)."""
    spec = _spec(["A", "B"], [_c("A", "B", "TPP")])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 20))
    bm = {b.region_id: b for b in real.boxes}
    ab = relation_between_boxes(bm["A"], bm["B"])
    ba = relation_between_boxes(bm["B"], bm["A"])
    assert rcc8.converse(ab) == ba
    assert rcc8.names(ab) == ["TPP"] and rcc8.names(ba) == ["TPPi"]


# ------------------------------------------------------------------ hand-built exact boxes

def test_manual_ec_edge_2d():
    # unit squares sharing the x=1 edge: EC, edge (LINE) contact.
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (1, 0), (2, 1))
    assert rcc8.names(relation_between_boxes(A, B)) == ["EC"]
    real = Realization(id="r", boxes=(A, B))
    assert extract_contact_dimension(real, "A", "B") == ContactDim.LINE


def test_manual_ec_corner_2d():
    # squares meeting only at the corner (1,1): EC, POINT contact.
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (1, 1), (2, 2))
    assert rcc8.names(relation_between_boxes(A, B)) == ["EC"]
    real = Realization(id="r", boxes=(A, B))
    assert extract_contact_dimension(real, "A", "B") == ContactDim.POINT


def test_manual_face_share_3d():
    # unit cubes sharing the z=1 face: EC, FACE contact.
    A = _box("A", (0, 0, 0), (1, 1, 1))
    B = _box("B", (0, 0, 1), (1, 1, 2))
    assert rcc8.names(relation_between_boxes(A, B)) == ["EC"]
    real = Realization(id="r", boxes=(A, B))
    assert extract_contact_dimension(real, "A", "B") == ContactDim.FACE


def test_manual_edge_share_3d():
    # cubes sharing only the z-axis edge at x=1,y=1: EC, LINE contact.
    A = _box("A", (0, 0, 0), (1, 1, 1))
    B = _box("B", (1, 1, 0), (2, 2, 1))
    assert rcc8.names(relation_between_boxes(A, B)) == ["EC"]
    real = Realization(id="r", boxes=(A, B))
    assert extract_contact_dimension(real, "A", "B") == ContactDim.LINE


def test_fractional_coordinates_exact():
    # thirds: no float can represent 1/3 exactly; extraction must stay exact.
    A = _box("A", (Fraction(0), Fraction(0)), (Fraction(1, 3), Fraction(1)))
    B = _box("B", (Fraction(1, 3), Fraction(0)), (Fraction(2, 3), Fraction(1)))
    assert rcc8.names(relation_between_boxes(A, B)) == ["EC"]
    real = Realization(id="r", boxes=(A, B))
    assert extract_contact_dimension(real, "A", "B") == ContactDim.LINE


# ------------------------------------------------------------------ contact-dim errors

def test_contact_dimension_disjoint_raises():
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (5, 5), (6, 6))
    real = Realization(id="r", boxes=(A, B))
    with pytest.raises(ValueError):
        extract_contact_dimension(real, "A", "B")


def test_contact_dimension_full_overlap_raises():
    A = _box("A", (0, 0), (2, 2))
    B = _box("B", (1, 1), (3, 3))  # PO: interiors overlap on both axes
    real = Realization(id="r", boxes=(A, B))
    with pytest.raises(ValueError):
        extract_contact_dimension(real, "A", "B")


def test_contact_dimension_missing_region_raises():
    A = _box("A", (0, 0), (1, 1))
    real = Realization(id="r", boxes=(A,))
    with pytest.raises(ValueError):
        extract_contact_dimension(real, "A", "Z")


# ------------------------------------------------------------------ THE hero: face demanded, edge realized

def test_face_required_edge_realized_is_witness_violation_3d():
    """RCC-8 EC is satisfied, but the intended FACE contact is realized as an edge.

    The reverse verifier must accept the EC constraint yet flag the witness."""
    # realize an EC pair whose contact is a 1D edge (LINE) in 3D...
    w_edge = IncidenceWitness(
        id="w_edge", cell_ids=("e0",), incident_regions=("A", "B"),
        intended_contact_dimension=ContactDim.LINE,
    )
    build = _spec(["A", "B"], [_c("A", "B", "EC")], witnesses=[w_edge])
    real = solve_boxes(build, domain="aabb_3d", bounds=(0, 10))
    assert real.status == Realizability.REALIZED_IN_D
    assert extract_contact_dimension(real, "A", "B") == ContactDim.LINE

    # ...but the *spec we verify against* demanded a FACE contact.
    w_face = IncidenceWitness(
        id="w_face", cell_ids=("f0",), incident_regions=("A", "B"),
        intended_contact_dimension=ContactDim.FACE,
    )
    spec = _spec(["A", "B"], [_c("A", "B", "EC")], witnesses=[w_face])

    rep = verify(spec, real)
    # the RCC-8 relation constraint itself IS satisfied (EC holds).
    assert rep.satisfied == ("A-B",)
    assert rep.violated == ()
    # but the witness is violated: face demanded, edge realized.
    assert len(rep.witness_violations) == 1
    v = rep.witness_violations[0]
    assert isinstance(v, WitnessViolation)
    assert v.code == "contact_dim_mismatch"
    assert v.witness_id == "w_face"
    assert v.intended == int(ContactDim.FACE)
    assert v.actual == int(ContactDim.LINE)


def test_face_required_edge_realized_2d():
    """Same claim in 2D: witness wants FACE(2) but 2D EC can only be an edge."""
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (1, 0), (2, 1))  # shares an edge -> EC, LINE contact
    real = Realization(id="r", boxes=(A, B))
    w_face = IncidenceWitness(
        id="w_face", cell_ids=(), incident_regions=("A", "B"),
        intended_contact_dimension=ContactDim.FACE,
    )
    spec = _spec(["A", "B"], [_c("A", "B", "EC")], witnesses=[w_face])
    rep = verify(spec, real)
    assert rep.satisfied == ("A-B",)
    assert rep.witness_violations[0].code == "contact_dim_mismatch"
    assert rep.witness_violations[0].actual == int(ContactDim.LINE)


def test_matching_contact_dimension_no_violation():
    """When the geometry realizes the intended contact dimension, no violation."""
    w_edge = IncidenceWitness(
        id="w_edge", cell_ids=(), incident_regions=("A", "B"),
        intended_contact_dimension=ContactDim.LINE,
    )
    spec = _spec(["A", "B"], [_c("A", "B", "EC")], witnesses=[w_edge])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))
    rep = verify(spec, real)
    assert rep.satisfied == ("A-B",)
    assert rep.witness_violations == ()


# ------------------------------------------------------------------ witness on a non-EC realization

def test_witness_on_disconnected_realization_flags_not_ec():
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (5, 0), (6, 1))  # DC
    real = Realization(id="r", boxes=(A, B))
    w = IncidenceWitness(id="w", cell_ids=(), incident_regions=("A", "B"),
                         intended_contact_dimension=ContactDim.LINE)
    spec = _spec(["A", "B"], [], witnesses=[w])
    rep = verify(spec, real)
    assert len(rep.witness_violations) == 1
    assert rep.witness_violations[0].code == "not_externally_connected"


def test_witness_missing_region():
    A = _box("A", (0, 0), (1, 1))
    real = Realization(id="r", boxes=(A,))
    w = IncidenceWitness(id="w", cell_ids=(), incident_regions=("A", "B"),
                         intended_contact_dimension=ContactDim.LINE)
    spec = _spec(["A", "B"], [], witnesses=[w])
    rep = verify(spec, real)
    assert rep.witness_violations[0].code == "missing_region"


# ------------------------------------------------------------------ relation constraint violations

def test_constraint_violated_when_geometry_disagrees():
    # geometry realizes EC, but the spec demands DC -> violated.
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (1, 0), (2, 1))  # EC
    real = Realization(id="r", boxes=(A, B))
    spec = _spec(["A", "B"], [_c("A", "B", "DC", cid="c")])
    rep = verify(spec, real)
    assert rep.violated == ("c",)
    assert rep.satisfied == ()


def test_disjunction_constraint_satisfied():
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (1, 0), (2, 1))  # EC, in {DC,EC}
    real = Realization(id="r", boxes=(A, B))
    spec = _spec(["A", "B"], [_c("A", "B", "DC", "EC", cid="c")])
    rep = verify(spec, real)
    assert rep.satisfied == ("c",)


def test_constraint_undecided_when_region_missing():
    A = _box("A", (0, 0), (1, 1))
    real = Realization(id="r", boxes=(A,))
    spec = _spec(["A", "B"], [_c("A", "B", "EC", cid="c")])
    rep = verify(spec, real)
    assert rep.undecided == ("c",)


def test_empty_mask_always_violated():
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (1, 0), (2, 1))
    real = Realization(id="r", boxes=(A, B))
    spec = RelSpec(
        regions=(Region(id="A"), Region(id="B")),
        constraints=(RelationConstraint(src="A", dst="B", rcc8_mask=rcc8.EMPTY, id="c"),),
    )
    rep = verify(spec, real)
    assert rep.violated == ("c",)


# ------------------------------------------------------------------ multi-region extraction

def test_extract_all_pairs_nested_chain():
    # A NTPP B NTPP C : extract_relations reports all three ordered pairs.
    spec = _spec(["A", "B", "C"], [_c("A", "B", "NTPP"), _c("B", "C", "NTPP")])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 30))
    rels = extract_relations(real)
    assert rcc8.names(rels[("A", "B")]) == ["NTPP"]
    assert rcc8.names(rels[("B", "C")]) == ["NTPP"]
    assert rcc8.names(rels[("A", "C")]) == ["NTPP"]  # transitively nested


def test_extract_relations_rejects_duplicate_ids():
    A = _box("A", (0, 0), (1, 1))
    A2 = _box("A", (2, 2), (3, 3))
    real = Realization(id="r", boxes=(A, A2))
    with pytest.raises(ValueError):
        extract_relations(real)


def test_degenerate_box_rejected():
    A = _box("A", (0, 0), (0, 1))   # zero width on axis 0
    B = _box("B", (1, 0), (2, 1))
    with pytest.raises(ValueError):
        relation_between_boxes(A, B)


def test_inverted_box_rejected():
    A = _box("A", (2, 0), (1, 1))   # lo > hi on axis 0
    B = _box("B", (3, 0), (4, 1))
    with pytest.raises(ValueError):
        relation_between_boxes(A, B)


def test_mixed_dimensionality_rejected():
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError):
        relation_between_boxes(A, B)


# ------------------------------------------------------------------ JEPD exhaustive enumeration

def _interval_relations():
    """All qualitatively-distinct 1-axis interval configs of [a0,a1] vs [b0,b1],
    with a fixed non-degenerate A=[0,4] and integer B endpoints spanning every case."""
    A0, A1 = Fraction(0), Fraction(4)
    cases = []
    for b0 in range(-2, 7):
        for b1 in range(b0 + 1, 8):  # b1 > b0 : non-degenerate
            cases.append(((A0, A1), (Fraction(b0), Fraction(b1))))
    return cases


def test_jepd_exhaustive_2d():
    """Every 2D box pair from an exhaustive integer-endpoint grid classifies to
    exactly one JEPD base relation, and the converse is consistent."""
    intervals = _interval_relations()
    seen = set()
    for (ax, bx) in intervals:
        for (ay, by) in intervals:
            A = Box("A", (ax[0], ay[0]), (ax[1], ay[1]))
            B = Box("B", (bx[0], by[0]), (bx[1], by[1]))
            m = relation_between_boxes(A, B)
            # exactly one base relation
            assert rcc8.is_base(m), (A, B, rcc8.names(m))
            # converse consistency
            assert rcc8.converse(m) == relation_between_boxes(B, A)
            seen.add(rcc8.names(m)[0])
    # the grid is rich enough to exhibit all 8 base relations
    assert seen == set(rcc8.BASE)


def test_po_with_coincident_endpoint():
    # interiors overlap on both axes, a single endpoint coincides, neither contains
    # the other -> must still be PO (not TPP/EC).
    A = _box("A", (0, 0), (2, 2))
    B = _box("B", (0, 1), (3, 3))   # b.lo[0]==a.lo[0]==0 but B extends past A -> PO
    assert rcc8.names(relation_between_boxes(A, B)) == ["PO"]


def test_tpp_one_axis_equal_other_strict():
    # A inside B, sharing the full x-extent (both x boundaries coincide), strictly
    # inside on y -> tangential proper part.
    A = _box("A", (0, 1), (4, 3))
    B = _box("B", (0, 0), (4, 4))
    assert rcc8.names(relation_between_boxes(A, B)) == ["TPP"]
    assert rcc8.names(relation_between_boxes(B, A)) == ["TPPi"]


def test_verifyreport_dict_access():
    A = _box("A", (0, 0), (1, 1))
    B = _box("B", (1, 0), (2, 1))
    real = Realization(id="r", boxes=(A, B))
    spec = _spec(["A", "B"], [_c("A", "B", "EC", cid="c")])
    rep = verify(spec, real)
    assert rep["satisfied"] == ("c",)
    assert set(rep.keys()) == {"satisfied", "violated", "undecided", "witness_violations"}
