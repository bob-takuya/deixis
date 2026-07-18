"""Tests for the equality-elimination + boundary-propagation presolver.

The presolver's contract is: *equivalence transforms only*. Its two reductions
(EQ-collapse and subset-domain tightening) must never change the solution set, and
the reduced spec, handed to :func:`solve_boxes`, must have the SAME satisfiability as
the original. It additionally detects *some* early contradictions (soundly, never a
false positive) and delegates disjunctions / cycles to Z3.

These tests verify:
  1. RCC-8 EQ variable identification / region reduction (incl. transitivity, order).
  2. NTPP / TPP containment -> per-axis boundary tightening (both directions, chains).
  3. Equivalence: solve_boxes(original).status == solve_boxes(reduced).status,
     across SAT *and* UNSAT specs (the crux property).
  4. Early contradiction detection (EMPTY mask, non-EQ self-relation, empty mask
     intersection, emptied propagation domain).
  5. Disjunctions / containment cycles are delegated to Z3 (not touched, no crash).
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from deixis.core.ids import F
from deixis.core.types import (
    IncidenceWitness,
    OverlayAtom,
    Realizability,
    Region,
    RelationConstraint,
    RelSpec,
)
from deixis.core import rcc8
from deixis.solver.presolve import presolve, PresolveReport, RegionBounds
from deixis.solver.geom_solver import solve_boxes


# ------------------------------------------------------------------ builders

def _spec(regions, constraints, witnesses=(), overlays=()):
    return RelSpec(
        regions=tuple(Region(id=r) if isinstance(r, str) else r for r in regions),
        constraints=tuple(constraints),
        witnesses=tuple(witnesses),
        overlays=tuple(overlays),
    )


def _c(src, dst, *names, cid=None):
    return RelationConstraint(
        src=src, dst=dst, rcc8_mask=rcc8.mask(*names), id=cid or f"{src}-{dst}",
    )


def _c_mask(src, dst, mask, cid=None):
    return RelationConstraint(src=src, dst=dst, rcc8_mask=mask, id=cid or f"{src}-{dst}")


def _region_ids(spec):
    return [r.id for r in spec.regions]


def _bounds_of(report, rid):
    for b in report.bounds:
        if b.region_id == rid:
            return b
    return None


# ================================================================== 1. EQ elimination

def test_eq_collapses_two_regions_into_one_representative():
    spec = _spec(["A", "B"], [_c("A", "B", "EQ", cid="eq")])
    rs, rep = presolve(spec, bounds=(0, 100))
    assert _region_ids(rs) == ["A"]                       # B eliminated
    assert rep.eliminated_regions == ("B",)
    assert rep.substitutions == (("B", "A"),)
    assert "eq" in rep.dropped_constraints                # rep EQ rep is trivially true
    assert rs.constraints == ()                           # nothing left
    assert rep.consistent is True


def test_eq_is_transitive_whole_class_collapses():
    spec = _spec(["A", "B", "C"],
                 [_c("A", "B", "EQ", cid="ab"), _c("B", "C", "EQ", cid="bc")])
    rs, rep = presolve(spec, bounds=(0, 100))
    assert _region_ids(rs) == ["A"]
    assert set(rep.eliminated_regions) == {"B", "C"}
    # every eliminated region maps to the single representative A
    assert dict(rep.substitutions) == {"B": "A", "C": "A"}


def test_eq_representative_is_earliest_declared():
    # declare in order C, B, A; EQ ties all three -> earliest-declared (C) represents.
    spec = _spec(["C", "B", "A"],
                 [_c("C", "A", "EQ", cid="ca"), _c("B", "A", "EQ", cid="ba")])
    rs, rep = presolve(spec, bounds=(0, 100))
    assert _region_ids(rs) == ["C"]
    assert dict(rep.substitutions) == {"A": "C", "B": "C"}


def test_eq_rewrites_other_constraints_onto_representative():
    # A EQ B ; B NTPP C  =>  A NTPP C (B rewritten to A)
    spec = _spec(["A", "B", "C"],
                 [_c("A", "B", "EQ", cid="eq"), _c("B", "C", "NTPP", cid="bc")])
    rs, rep = presolve(spec, bounds=(0, 100))
    assert _region_ids(rs) == ["A", "C"]
    kept = [(c.src, c.dst, rcc8.names(c.rcc8_mask)) for c in rs.constraints]
    assert kept == [("A", "C", ["NTPP"])]


def test_disjunctive_eq_mask_is_not_collapsed():
    # {EQ, PO} is a disjunction, not a definite equality -> delegate to Z3, keep both regions.
    spec = _spec(["A", "B"], [_c("A", "B", "EQ", "PO", cid="ab")])
    rs, rep = presolve(spec, bounds=(0, 100))
    assert _region_ids(rs) == ["A", "B"]
    assert rep.substitutions == ()
    assert rep.eliminated_regions == ()
    assert len(rs.constraints) == 1


def test_witnesses_and_overlays_are_remapped_onto_representative():
    w = IncidenceWitness(id="w0", cell_ids=("k0",),
                         incident_regions=("B", "C"), intended_contact_dimension=1)
    ov = OverlayAtom(id="o0", members=frozenset({"B", "C"}))
    spec = _spec(["A", "B", "C"],
                 [_c("A", "B", "EQ", cid="eq")],
                 witnesses=(w,), overlays=(ov,))
    rs, rep = presolve(spec, bounds=(0, 100))
    # B was collapsed into A; witness/overlay members must reflect the substitution.
    assert rs.witnesses[0].incident_regions == ("A", "C")
    assert rs.overlays[0].members == frozenset({"A", "C"})


# ================================================================== 2. bounds tightening

def _map_bounds(**kw):
    """region -> ((lo0,lo1),(hi0,hi1)) helper for 2D."""
    return {k: v for k, v in kw.items()}


def test_ntpp_tightens_child_domain_to_parent():
    # A NTPP C : A's box is inside C's box, so A's domain intersects C's (tighter) domain.
    spec = _spec(["A", "C"], [_c("A", "C", "NTPP", cid="ac")])
    bounds = _map_bounds(
        A=((0, 0), (100, 100)),      # child wide open
        C=((10, 20), (40, 50)),      # parent tight
    )
    rs, rep = presolve(spec, bounds=bounds)
    a = _bounds_of(rep, "A")
    assert a.lo == (F(10), F(20))
    assert a.hi == (F(40), F(50))
    assert rep.consistent is True
    # exactness
    for v in (*a.lo, *a.hi):
        assert isinstance(v, Fraction)


def test_tpp_also_tightens_child_domain():
    spec = _spec(["A", "C"], [_c("A", "C", "TPP", cid="ac")])
    bounds = _map_bounds(A=((0, 0), (100, 100)), C=((5, 5), (30, 30)))
    rs, rep = presolve(spec, bounds=bounds)
    a = _bounds_of(rep, "A")
    assert (a.lo, a.hi) == ((F(5), F(5)), (F(30), F(30)))


def test_converse_subset_tightens_the_other_region():
    # A NTPPi C means C is inside A -> C is the child; C's domain tightens to A's.
    spec = _spec(["A", "C"], [_c("A", "C", "NTPPi", cid="ac")])
    bounds = _map_bounds(A=((10, 10), (40, 40)), C=((0, 0), (100, 100)))
    rs, rep = presolve(spec, bounds=bounds)
    c = _bounds_of(rep, "C")
    assert (c.lo, c.hi) == ((F(10), F(10)), (F(40), F(40)))
    # A (the parent) is NOT widened/altered by its child
    a = _bounds_of(rep, "A")
    assert (a.lo, a.hi) == ((F(10), F(10)), (F(40), F(40)))


def test_transitive_containment_chain_propagates_to_leaf():
    # A NTPP B, B NTPP C ; only C carries a tight domain -> must reach A through B.
    spec = _spec(["A", "B", "C"],
                 [_c("A", "B", "NTPP", cid="ab"), _c("B", "C", "NTPP", cid="bc")])
    bounds = _map_bounds(
        A=((0, 0), (100, 100)),
        B=((0, 0), (100, 100)),
        C=((10, 10), (20, 20)),
    )
    rs, rep = presolve(spec, bounds=bounds)
    assert (_bounds_of(rep, "A").lo, _bounds_of(rep, "A").hi) == ((F(10), F(10)), (F(20), F(20)))
    assert (_bounds_of(rep, "B").lo, _bounds_of(rep, "B").hi) == ((F(10), F(10)), (F(20), F(20)))
    assert rep.propagation_passes >= 1


def test_scalar_bounds_apply_to_every_region():
    spec = _spec(["A", "C"], [_c("A", "C", "NTPP", cid="ac")])
    rs, rep = presolve(spec, bounds=(0, 100))
    for rid in ("A", "C"):
        b = _bounds_of(rep, rid)
        assert b.lo == (F(0), F(0)) and b.hi == (F(100), F(100))


def test_bounds_none_skips_tightening_but_eq_still_runs():
    spec = _spec(["A", "B", "C"],
                 [_c("A", "B", "EQ", cid="eq"), _c("B", "C", "NTPP", cid="bc")])
    rs, rep = presolve(spec, bounds=None)
    assert _region_ids(rs) == ["A", "C"]          # EQ collapse happened
    assert rep.bounds == ()                        # no domain tightening
    assert rep.propagation_passes == 0


def test_tightening_is_exact_over_non_integer_fractions():
    # non-integer rational bounds must tighten exactly (no float, no rounding).
    spec = _spec(["A", "C"], [_c("A", "C", "NTPP", cid="ac")])
    bounds = {
        "A": ((Fraction(1, 3), Fraction(-2, 7)), (Fraction(11, 5), Fraction(17, 3))),
        "C": ((Fraction(2, 3), Fraction(-1, 7)), (Fraction(7, 5), Fraction(13, 3))),
    }
    rs, rep = presolve(spec, bounds=bounds)
    a = _bounds_of(rep, "A")
    # child A tightened: lo = max, hi = min, exactly.
    assert a.lo == (Fraction(2, 3), Fraction(-1, 7))    # max(1/3,2/3), max(-2/7,-1/7)
    assert a.hi == (Fraction(7, 5), Fraction(13, 3))    # min(11/5,7/5), min(17/3,13/3)
    for v in (*a.lo, *a.hi):
        assert isinstance(v, Fraction)


def test_eq_members_intersect_their_input_domains():
    # A EQ B with different input domains: representative domain is the intersection
    # (A and B are the same box, so it must lie in both).
    spec = _spec(["A", "B"], [_c("A", "B", "EQ", cid="eq")])
    bounds = _map_bounds(A=((0, 0), (50, 80)), B=((10, 5), (100, 100)))
    rs, rep = presolve(spec, bounds=bounds)
    a = _bounds_of(rep, "A")
    assert a.lo == (F(10), F(5))       # max of lows
    assert a.hi == (F(50), F(80))      # min of highs


# ================================================================== 3. solution-set invariance

# Each spec is (regions, constraints, expected_status_note). We assert that
# solve_boxes(original).status == solve_boxes(reduced).status regardless of what
# that status is -- presolve must never flip satisfiability.
_INVARIANCE_SPECS = {
    "eq_then_containment_sat": _spec(
        ["A", "B", "C"],
        [_c("A", "B", "EQ"), _c("A", "C", "NTPP")],
    ),
    "eq_chain_sat": _spec(
        ["A", "B", "C", "D"],
        [_c("A", "B", "EQ"), _c("B", "C", "EQ"), _c("A", "D", "NTPP")],
    ),
    "plain_containment_sat": _spec(
        ["A", "B"], [_c("A", "B", "NTPP")],
    ),
    "eq_conflict_unsat": _spec(   # A=B but A DC B  -> UNSAT (the fixed-drop bug case)
        ["A", "B"], [_c("A", "B", "EQ"), _c("A", "B", "DC")],
    ),
    "eq_transitive_conflict_unsat": _spec(  # A=B=C but A PO C -> UNSAT after collapse
        ["A", "B", "C"],
        [_c("A", "B", "EQ"), _c("B", "C", "EQ"), _c("A", "C", "PO")],
    ),
    "parallel_conflict_unsat": _spec(  # same pair forced DC and PO
        ["A", "B"], [_c("A", "B", "DC"), _c("A", "B", "PO")],
    ),
    "empty_mask_unsat": _spec(
        ["A", "B"], [_c_mask("A", "B", rcc8.EMPTY)],
    ),
    "containment_cycle_unsat": _spec(  # A TPP B & B TPP A -> both proper subsets -> UNSAT
        ["A", "B"], [_c("A", "B", "TPP"), _c("B", "A", "TPP")],
    ),
    "disjunction_sat": _spec(
        ["A", "B", "C"],
        [_c("A", "B", "PO", "EC"), _c("B", "C", "DC")],
    ),
}


@pytest.mark.parametrize("name", sorted(_INVARIANCE_SPECS))
def test_solve_status_unchanged_by_presolve(name):
    spec = _INVARIANCE_SPECS[name]
    original = solve_boxes(spec, bounds=(0, 100))
    reduced_spec, rep = presolve(spec, bounds=(0, 100))
    reduced = solve_boxes(reduced_spec, bounds=(0, 100))
    assert original.status == reduced.status, (
        f"{name}: presolve changed satisfiability "
        f"{original.status} -> {reduced.status}"
    )
    # if presolve *proved* inconsistency, the true answer must indeed be UNSAT.
    if not rep.consistent:
        assert original.status == Realizability.INCONSISTENT


def test_eq_collapse_preserves_realizability_and_geometry():
    # A EQ B, A NTPP C: SAT. Reduced solves and produces a valid geometry.
    spec = _spec(["A", "B", "C"],
                 [_c("A", "B", "EQ", cid="eq"), _c("A", "C", "NTPP", cid="ac")])
    reduced_spec, rep = presolve(spec, bounds=(0, 100))
    r = solve_boxes(reduced_spec, bounds=(0, 100))
    assert r.status == Realizability.REALIZED_IN_D
    boxes = {b.region_id: b for b in r.boxes}
    # A strictly inside C on every axis (NTPP)
    A, C = boxes["A"], boxes["C"]
    for d in range(2):
        assert C.lo[d] < A.lo[d] and A.hi[d] < C.hi[d]


# ================================================================== 4. early contradiction

def test_empty_mask_flagged_inconsistent():
    spec = _spec(["A", "B"], [_c_mask("A", "B", rcc8.EMPTY, cid="e")])
    rs, rep = presolve(spec, bounds=(0, 100))
    assert rep.consistent is False
    assert "e" in rep.contradiction or "EMPTY" in rep.contradiction


def test_non_eq_self_relation_after_collapse_is_contradiction_and_kept():
    # A EQ B, A DC B -> after collapse A DC A. consistent=False AND the reduced spec
    # keeps the self-constraint so solve_boxes still returns UNSAT.
    spec = _spec(["A", "B"],
                 [_c("A", "B", "EQ", cid="eq"), _c("A", "B", "DC", cid="dc")])
    rs, rep = presolve(spec, bounds=(0, 100))
    assert rep.consistent is False
    assert "dc" not in rep.dropped_constraints          # NOT dropped: it's a contradiction
    kept = [(c.src, c.dst) for c in rs.constraints]
    assert ("A", "A") in kept                            # self-constraint retained
    assert solve_boxes(rs, bounds=(0, 100)).status == Realizability.INCONSISTENT


def test_parallel_masks_intersecting_to_empty_flagged():
    spec = _spec(["A", "B"],
                 [_c("A", "B", "DC", cid="dc"), _c("A", "B", "PO", cid="po")])
    rs, rep = presolve(spec, bounds=(0, 100))
    assert rep.consistent is False
    assert "EMPTY" in rep.contradiction or "intersect" in rep.contradiction


def test_parallel_masks_via_converse_intersect_to_empty():
    # A TPP B (=> A subset B) and B TPP A (=> B subset A). Folded through converse the
    # pair mask is intersect(TPP, converse(TPP)=TPPi) = EMPTY.
    spec = _spec(["A", "B"],
                 [_c("A", "B", "TPP", cid="ab"), _c("B", "A", "TPP", cid="ba")])
    rs, rep = presolve(spec, bounds=(0, 100))
    assert rep.consistent is False


def test_propagation_empties_domain_flagged_inconsistent():
    # child A must be inside parent C, but A's own floor exceeds C's ceiling on axis 0.
    spec = _spec(["A", "C"], [_c("A", "C", "NTPP", cid="ac")])
    bounds = _map_bounds(A=((10, 0), (100, 100)), C=((0, 0), (5, 100)))
    rs, rep = presolve(spec, bounds=bounds)
    assert rep.consistent is False
    assert "empty" in rep.contradiction.lower()


def test_consistent_spec_reports_consistent():
    spec = _spec(["A", "B"], [_c("A", "B", "NTPP", cid="ab")])
    bounds = _map_bounds(A=((0, 0), (100, 100)), B=((0, 0), (100, 100)))
    rs, rep = presolve(spec, bounds=bounds)
    assert rep.consistent is True
    assert rep.contradiction == ""


# ================================================================== 5. delegation (untouched)

def test_disjunctive_non_subset_mask_drives_no_propagation():
    # {PO, EC} is not a subset relation -> no subset edge, bounds untouched, delegated.
    spec = _spec(["A", "B"], [_c("A", "B", "PO", "EC", cid="ab")])
    bounds = _map_bounds(A=((0, 0), (100, 100)), B=((10, 10), (20, 20)))
    rs, rep = presolve(spec, bounds=bounds)
    # A is not tightened by B (relation does not entail containment)
    a = _bounds_of(rep, "A")
    assert (a.lo, a.hi) == ((F(0), F(0)), (F(100), F(100)))
    assert rep.consistent is True
    # constraint preserved verbatim for Z3
    assert rs.constraints[0].rcc8_mask == rcc8.mask("PO", "EC")


def test_containment_cycle_not_resolved_but_delegated_without_crash():
    # A NTPP B, B NTPP A: a containment cycle. presolve does not resolve it to EQ /
    # claim contradiction; it delegates. solve_boxes is the authority (UNSAT here).
    spec = _spec(["A", "B"], [_c("A", "B", "NTPP", cid="ab"), _c("B", "A", "NTPP", cid="ba")])
    rs, rep = presolve(spec, bounds=(0, 100))
    # not collapsed (no definite EQ), both regions survive
    assert set(_region_ids(rs)) == {"A", "B"}
    # presolve's mask fold DOES catch this one (NTPP vs converse NTPP = NTPPi -> EMPTY),
    # but crucially it does not crash and the reduced spec still matches solve_boxes.
    orig = solve_boxes(spec, bounds=(0, 100)).status
    red = solve_boxes(rs, bounds=(0, 100)).status
    assert orig == red == Realizability.INCONSISTENT


# ================================================================== 6. domain / arg handling

def test_aabb_3d_bounds_tightening_uses_three_axes():
    spec = _spec(["A", "C"], [_c("A", "C", "NTPP", cid="ac")])
    bounds = {
        "A": ((0, 0, 0), (100, 100, 100)),
        "C": ((1, 2, 3), (7, 8, 9)),
    }
    rs, rep = presolve(spec, bounds=bounds, domain="aabb_3d")
    a = _bounds_of(rep, "A")
    assert a.lo == (F(1), F(2), F(3))
    assert a.hi == (F(7), F(8), F(9))


def test_unknown_domain_raises():
    spec = _spec(["A"], [])
    with pytest.raises(ValueError):
        presolve(spec, bounds=(0, 100), domain="hexahedron")


def test_bounds_axis_count_mismatch_raises():
    spec = _spec(["A"], [])
    with pytest.raises(ValueError):
        presolve(spec, bounds={"A": ((0, 0, 0), (1, 1, 1))}, domain="aabb_2d")


def test_report_is_frozen_dataclass_with_scope_note():
    spec = _spec(["A", "B"], [_c("A", "B", "EQ", cid="eq")])
    _, rep = presolve(spec, bounds=(0, 100))
    assert isinstance(rep, PresolveReport)
    assert rep.scope_note  # honest-scope statement present
    with pytest.raises(Exception):
        rep.consistent = False  # type: ignore[misc]  frozen


def test_returned_spec_is_new_object_original_unmutated():
    spec = _spec(["A", "B"], [_c("A", "B", "EQ", cid="eq")])
    before = _region_ids(spec)
    rs, _ = presolve(spec, bounds=(0, 100))
    assert rs is not spec
    assert _region_ids(spec) == before  # original untouched
