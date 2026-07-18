"""Tests for the bounded exact-rational RCC-8 -> AABB solver (geom_solver)."""
from __future__ import annotations

from fractions import Fraction

import pytest

from deixis.core.types import (
    ContactDim,
    IncidenceWitness,
    Realizability,
    Region,
    RelationConstraint,
    RelSpec,
)
from deixis.core import rcc8
from deixis.solver.geom_solver import solve_boxes


# ------------------------------------------------------------------ geometry helpers
# All operate on exact Fraction box bounds pulled from a Realization.

def _boxmap(real):
    return {b.region_id: b for b in real.boxes}


def _assert_exact(real):
    for b in real.boxes:
        for v in (*b.lo, *b.hi):
            assert isinstance(v, Fraction), f"coordinate {v!r} is not an exact Fraction"


def _iover(A, B, d):
    return A.lo[d] < B.hi[d] and B.lo[d] < A.hi[d]


def _touch(A, B, d):
    return A.hi[d] == B.lo[d] or B.hi[d] == A.lo[d]


def _sep(A, B, d):
    return A.hi[d] < B.lo[d] or B.hi[d] < A.lo[d]


def _closed_intersect(A, B, d):
    return A.lo[d] <= B.hi[d] and B.lo[d] <= A.hi[d]


def _nondegenerate(b):
    return all(lo < hi for lo, hi in zip(b.lo, b.hi))


# ------------------------------------------------------------------ spec builders

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


# ------------------------------------------------------------------ DC

def test_dc_separates():
    spec = _spec(["A", "B"], [_c("A", "B", "DC")])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))
    assert real.status == Realizability.REALIZED_IN_D
    _assert_exact(real)
    bm = _boxmap(real)
    A, B = bm["A"], bm["B"]
    assert _nondegenerate(A) and _nondegenerate(B)
    # disconnected => a gap on at least one axis
    assert _sep(A, B, 0) or _sep(A, B, 1)
    assert "A-B" in real.satisfied


# ------------------------------------------------------------------ EC by face (edge share)

def test_ec_by_face_shares_an_edge():
    w = IncidenceWitness(
        id="w0",
        cell_ids=("e0",),
        incident_regions=("A", "B"),
        intended_contact_dimension=ContactDim.LINE,  # edge share in 2D
    )
    spec = _spec(["A", "B"], [_c("A", "B", "EC")], witnesses=[w])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))
    assert real.status == Realizability.REALIZED_IN_D
    _assert_exact(real)
    A, B = _boxmap(real)["A"], _boxmap(real)["B"]
    assert _nondegenerate(A) and _nondegenerate(B)
    # exactly one axis touches at a coincident boundary; the other overlaps (segment).
    touch_axes = [d for d in range(2) if _touch(A, B, d)]
    over_axes = [d for d in range(2) if _iover(A, B, d)]
    assert len(touch_axes) == 1, (A, B)
    assert len(over_axes) == 1
    # touch axis has coincident boundary => zero interior overlap there
    td = touch_axes[0]
    assert not _iover(A, B, td)
    # connected on both axes (they actually meet)
    assert _closed_intersect(A, B, 0) and _closed_intersect(A, B, 1)


def test_ec_by_point_touches_at_corner():
    w = IncidenceWitness(
        id="w0",
        cell_ids=("v0",),
        incident_regions=("A", "B"),
        intended_contact_dimension=ContactDim.POINT,  # corner-only touch
    )
    spec = _spec(["A", "B"], [_c("A", "B", "EC")], witnesses=[w])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))
    assert real.status == Realizability.REALIZED_IN_D
    A, B = _boxmap(real)["A"], _boxmap(real)["B"]
    # both axes touch at a boundary, no interior overlap anywhere => single point
    assert _touch(A, B, 0) and _touch(A, B, 1)
    assert not _iover(A, B, 0) and not _iover(A, B, 1)


# ------------------------------------------------------------------ NTPP (true containment)

def test_ntpp_strict_containment():
    spec = _spec(["A", "B"], [_c("A", "B", "NTPP")])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))
    assert real.status == Realizability.REALIZED_IN_D
    _assert_exact(real)
    A, B = _boxmap(real)["A"], _boxmap(real)["B"]
    # A strictly interior to B on every axis
    for d in range(2):
        assert B.lo[d] < A.lo[d] and A.hi[d] < B.hi[d]


def test_tpp_tangential_containment():
    spec = _spec(["A", "B"], [_c("A", "B", "TPP")])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))
    assert real.status == Realizability.REALIZED_IN_D
    A, B = _boxmap(real)["A"], _boxmap(real)["B"]
    # A inside B on every axis
    for d in range(2):
        assert A.lo[d] >= B.lo[d] and A.hi[d] <= B.hi[d]
    # at least one boundary coincides (tangential) and A != B (proper)
    coincides = any(A.lo[d] == B.lo[d] or A.hi[d] == B.hi[d] for d in range(2))
    proper = any(A.lo[d] > B.lo[d] or A.hi[d] < B.hi[d] for d in range(2))
    assert coincides and proper


# ------------------------------------------------------------------ PO / EQ

def test_po_proper_overlap():
    spec = _spec(["A", "B"], [_c("A", "B", "PO")])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))
    assert real.status == Realizability.REALIZED_IN_D
    A, B = _boxmap(real)["A"], _boxmap(real)["B"]
    # interiors overlap on both axes
    assert _iover(A, B, 0) and _iover(A, B, 1)
    # neither contains the other
    a_in_b = all(A.lo[d] >= B.lo[d] and A.hi[d] <= B.hi[d] for d in range(2))
    b_in_a = all(B.lo[d] >= A.lo[d] and B.hi[d] <= A.hi[d] for d in range(2))
    assert not a_in_b and not b_in_a


def test_eq_equal_boxes():
    spec = _spec(["A", "B"], [_c("A", "B", "EQ")])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))
    assert real.status == Realizability.REALIZED_IN_D
    A, B = _boxmap(real)["A"], _boxmap(real)["B"]
    assert A.lo == B.lo and A.hi == B.hi


# ------------------------------------------------------------------ disjunction mask

def test_disjunction_mask_is_satisfiable():
    # DC|EC: solver may pick either; result must be exactly one of them.
    spec = _spec(["A", "B"], [_c("A", "B", "DC", "EC")])
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))
    assert real.status == Realizability.REALIZED_IN_D
    A, B = _boxmap(real)["A"], _boxmap(real)["B"]
    is_dc = _sep(A, B, 0) or _sep(A, B, 1)
    connected = _closed_intersect(A, B, 0) and _closed_intersect(A, B, 1)
    is_ec = connected and (_touch(A, B, 0) or _touch(A, B, 1)) \
        and not (_iover(A, B, 0) and _iover(A, B, 1))
    assert is_dc or is_ec


# ------------------------------------------------------------------ UNSAT network

def test_unsat_mutual_ntpp():
    # A NTPP B and B NTPP A cannot both hold (mutual strict containment).
    spec = _spec(
        ["A", "B"],
        [_c("A", "B", "NTPP", cid="c1"), _c("B", "A", "NTPP", cid="c2")],
    )
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))
    assert real.status == Realizability.INCONSISTENT
    assert real.boxes == ()
    assert set(real.undecided) == {"c1", "c2"}


def test_unsat_contradictory_pair():
    # Same pair required to be both DC and EC.
    spec = _spec(
        ["A", "B"],
        [_c("A", "B", "DC", cid="c1"), _c("A", "B", "EC", cid="c2")],
    )
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))
    assert real.status == Realizability.INCONSISTENT


# ------------------------------------------------------------------ 3D

def test_ec_face_share_3d():
    w = IncidenceWitness(
        id="w0",
        cell_ids=("f0",),
        incident_regions=("A", "B"),
        intended_contact_dimension=ContactDim.FACE,  # 2D face share in 3D
    )
    spec = _spec(["A", "B"], [_c("A", "B", "EC")], witnesses=[w])
    real = solve_boxes(spec, domain="aabb_3d", bounds=(0, 10))
    assert real.status == Realizability.REALIZED_IN_D
    _assert_exact(real)
    A, B = _boxmap(real)["A"], _boxmap(real)["B"]
    assert len(A.lo) == 3
    touch_axes = [d for d in range(3) if _touch(A, B, d)]
    over_axes = [d for d in range(3) if _iover(A, B, d)]
    assert len(touch_axes) == 1 and len(over_axes) == 2


# ------------------------------------------------------------------ transitive chain

def test_ntpp_chain_transitive():
    # A NTPP B NTPP C  -- all realizable together, nested boxes.
    spec = _spec(
        ["A", "B", "C"],
        [_c("A", "B", "NTPP"), _c("B", "C", "NTPP")],
    )
    real = solve_boxes(spec, domain="aabb_2d", bounds=(0, 20))
    assert real.status == Realizability.REALIZED_IN_D
    bm = _boxmap(real)
    A, B, C = bm["A"], bm["B"], bm["C"]
    for d in range(2):
        assert C.lo[d] < B.lo[d] < A.lo[d] and A.hi[d] < B.hi[d] < C.hi[d]


# ------------------------------------------------------------------ misc / errors

def test_unknown_region_raises():
    spec = _spec(["A"], [_c("A", "B", "DC")])
    with pytest.raises(ValueError):
        solve_boxes(spec, domain="aabb_2d", bounds=(0, 10))


def test_bad_domain_raises():
    spec = _spec(["A", "B"], [_c("A", "B", "DC")])
    with pytest.raises(ValueError):
        solve_boxes(spec, domain="aabb_9d", bounds=(0, 10))
