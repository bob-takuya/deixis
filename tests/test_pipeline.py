"""Tests for the two-stage fail-closed realizability pipeline (deixis.pipeline)."""
from __future__ import annotations

from fractions import Fraction

import pytest

from deixis.core import rcc8
from deixis.core.types import (
    GroundingDecision,
    Realizability,
    Region,
    RelationConstraint,
    RelSpec,
    Status,
)
from deixis.core.ids import F
from deixis.incidence.witness import make_witness
from deixis.grounding.grounding_op import GroundingError, set_invariant
from deixis import pipeline


# ---------------------------------------------------------------- builders
def _regions(*ids: str) -> tuple[Region, ...]:
    return tuple(Region(id=i) for i in ids)


def _c(src: str, dst: str, mask: int, cid: str, status: Status = Status.DELEGATED) -> RelationConstraint:
    return RelationConstraint(src=src, dst=dst, rcc8_mask=mask, status=status, id=cid)


# ---------------------------------------------------------------- Stage B success
def test_consistent_spec_reaches_realized_in_d():
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("EC"), "cAB"),),
    )
    r = pipeline.run(spec, domain="aabb_2d", bounds=(0, 100))
    assert r.status is Realizability.REALIZED_IN_D
    assert "cAB" in r.satisfied
    assert not r.violated
    assert len(r.boxes) == 2
    # exact rational readout, no floats
    for bx in r.boxes:
        for v in (*bx.lo, *bx.hi):
            assert isinstance(v, Fraction)


def test_realized_relation_matches_readback():
    # NTPP: A strictly inside B -> reverse-verify must confirm NTPP membership.
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("NTPP"), "cAB"),),
    )
    r = pipeline.run(spec, bounds=(0, 100))
    assert r.status is Realizability.REALIZED_IN_D
    a = next(bx for bx in r.boxes if bx.region_id == "A")
    b = next(bx for bx in r.boxes if bx.region_id == "B")
    base = pipeline._rcc_base_of_boxes(a, b, 2)
    assert base == "NTPP"


def test_witness_contact_dimension_is_verified():
    # EC with an edge (LINE, dim=1) contact witness in 2D: exactly one axis interior-overlaps.
    w = make_witness(("A", "B"), 1, id="wAB")  # 1 == LINE
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("EC"), "cAB"),),
        witnesses=(w,),
    )
    r = pipeline.run(spec, bounds=(0, 100))
    assert r.status is Realizability.REALIZED_IN_D
    a = next(bx for bx in r.boxes if bx.region_id == "A")
    b = next(bx for bx in r.boxes if bx.region_id == "B")
    assert pipeline._rcc_base_of_boxes(a, b, 2) == "EC"
    assert pipeline._interior_overlap_axes(a, b, 2) == 1


# ---------------------------------------------------------------- Stage A halts
def test_direct_contradiction_is_inconsistent_no_geometry(monkeypatch):
    # EC and DC on the same pair -> intersection EMPTY -> inconsistent.
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(
            _c("A", "B", rcc8.mask("EC"), "cEC"),
            _c("A", "B", rcc8.mask("DC"), "cDC"),
        ),
    )

    def _boom(*a, **k):  # geometry must NOT be attempted after Stage A fails
        raise AssertionError("geom_solver should not run on an INCONSISTENT spec")

    monkeypatch.setattr(pipeline.geom_solver, "solve_boxes", _boom)
    r = pipeline.run(spec)
    assert r.status is Realizability.INCONSISTENT
    assert r.boxes == ()


def test_compositional_contradiction_is_inconsistent(monkeypatch):
    # A TPP B, B TPP C  =>  A rel C in {TPP,NTPP}; declaring A DC C is a contradiction.
    spec = RelSpec(
        regions=_regions("A", "B", "C"),
        constraints=(
            _c("A", "B", rcc8.mask("TPP"), "cAB"),
            _c("B", "C", rcc8.mask("TPP"), "cBC"),
            _c("A", "C", rcc8.mask("DC"), "cAC"),
        ),
    )
    monkeypatch.setattr(
        pipeline.geom_solver, "solve_boxes",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no geometry")),
    )
    r = pipeline.run(spec)
    assert r.status is Realizability.INCONSISTENT
    assert set(r.violated)  # some constraints blamed


def test_witness_contradiction_is_inconsistent():
    # Witness says A,B touch (entails EC); constraint forbids EC (DC) -> symbolic clash.
    w = make_witness(("A", "B"), 2, id="wAB")  # FACE contact
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("DC"), "cAB"),),
        witnesses=(w,),
    )
    r = pipeline.run(spec)
    assert r.status is Realizability.INCONSISTENT
    assert "cAB" in r.violated


# ---------------------------------------------------------------- Stage A terminal (no realize)
def test_rcc_consistent_without_realization():
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("EC"), "cAB"),),
    )
    r = pipeline.run(spec, realize=False)
    assert r.status is Realizability.RCC_CONSISTENT
    assert r.boxes == ()
    assert "cAB" in r.satisfied


# ---------------------------------------------------------------- realized-unsat-in-D => UNKNOWN
def test_geometry_unsat_in_d_is_unknown_not_inconsistent():
    # A NTPP B (strictly interior) but bounds+min_size leave no room -> unsat in D.
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("NTPP"), "cAB"),),
    )
    r = pipeline.run(spec, bounds=(0, 1), min_size=Fraction(1))
    assert r.status is Realizability.UNKNOWN  # NOT inconsistent: symbolic was fine
    assert r.boxes == ()
    assert "cAB" in r.undecided


def test_grounding_to_disjoint_relation_is_unknown():
    # Constraint EC, grounded to DC -> refined mask EMPTY -> geometry unsat -> UNKNOWN.
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("EC"), "cAB"),),
    )
    d = GroundingDecision(id="g0", target="cAB", fixed_value_or_domain="DC")
    r = pipeline.run(spec, groundings=(d,))
    assert r.status is Realizability.UNKNOWN


# ---------------------------------------------------------------- grounding fail-closed
def test_grounding_invariant_constraint_is_fail_closed():
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("EC"), "cAB"),),
    )
    spec = set_invariant(spec, "cAB")  # promote to un-touchable core
    d = GroundingDecision(id="g0", target="cAB", fixed_value_or_domain="DC")
    with pytest.raises(GroundingError):
        pipeline.run(spec, groundings=(d,))


# ---------------------------------------------------------------- family: multiple solutions
def test_family_produces_multiple_distinct_solutions():
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.UNIVERSAL, "cAB"),),  # delegated, open relation
    )
    scenarios = [
        (GroundingDecision(id="gDC", target="cAB", fixed_value_or_domain="DC"),),
        (GroundingDecision(id="gEC", target="cAB", fixed_value_or_domain="EC"),),
    ]
    results = pipeline.family(spec, scenarios, bounds=(0, 100))
    assert len(results) == 2
    assert [r.scenario_id for r in results] == ["s0", "s1"]
    assert all(r.status is Realizability.REALIZED_IN_D for r in results)
    # the two scenarios realize genuinely different geometry (DC vs EC)
    a0 = next(bx for bx in results[0].boxes if bx.region_id == "A")
    b0 = next(bx for bx in results[0].boxes if bx.region_id == "B")
    a1 = next(bx for bx in results[1].boxes if bx.region_id == "A")
    b1 = next(bx for bx in results[1].boxes if bx.region_id == "B")
    assert pipeline._rcc_base_of_boxes(a0, b0, 2) == "DC"
    assert pipeline._rcc_base_of_boxes(a1, b1, 2) == "EC"


def test_family_shares_immutable_base_spec():
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.UNIVERSAL, "cAB"),),
    )
    scenarios = [
        (GroundingDecision(id="gEC", target="cAB", fixed_value_or_domain="EC"),),
    ]
    pipeline.family(spec, scenarios)
    # base spec untouched: still delegated, still one universal constraint, empty ledger
    assert spec.constraints[0].status is Status.DELEGATED
    assert spec.constraints[0].rcc8_mask == rcc8.UNIVERSAL
    assert spec.groundings == ()


# ---------------------------------------------------------------- helpers / edge cases
def test_unknown_domain_raises():
    spec = RelSpec(regions=_regions("A"), constraints=())
    with pytest.raises(pipeline.PipelineError):
        pipeline.run(spec, domain="nope")


def test_parse_relation_mask_ignores_coordinate_groundings():
    assert pipeline._parse_relation_mask("x==4.20") is None
    assert pipeline._parse_relation_mask("EC") == rcc8.mask("EC")
    assert pipeline._parse_relation_mask("EC|DC") == rcc8.mask("EC", "DC")
    assert pipeline._parse_relation_mask("rel:NTPP") == rcc8.mask("NTPP")


def test_witness_only_pair_is_imposed_and_verified():
    # A witnessed edge contact with NO explicit RelationConstraint must still be
    # realized (as EC, dim 1) and reverse-verified -- not left unconstrained.
    w = make_witness(("A", "B"), 1, id="wAB")  # LINE contact, no constraint on the pair
    spec = RelSpec(regions=_regions("A", "B"), constraints=(), witnesses=(w,))
    r = pipeline.run(spec, bounds=(0, 100))
    assert r.status is Realizability.REALIZED_IN_D
    a = next(bx for bx in r.boxes if bx.region_id == "A")
    b = next(bx for bx in r.boxes if bx.region_id == "B")
    assert pipeline._rcc_base_of_boxes(a, b, 2) == "EC"
    assert pipeline._interior_overlap_axes(a, b, 2) == 1
    assert any(cid.startswith("witness:") for cid in r.satisfied)


def test_conflicting_witness_dimensions_is_inconsistent():
    # Same pair declared both edge (1) and point (0) contact: symbolic contradiction.
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(),
        witnesses=(
            make_witness(("A", "B"), 1, id="wLine"),
            make_witness(("A", "B"), 0, id="wPoint"),
        ),
    )
    r = pipeline.run(spec)
    assert r.status is Realizability.INCONSISTENT


def test_grounding_unknown_target_is_fail_closed():
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("EC"), "cAB"),),
    )
    d = GroundingDecision(id="g0", target="does_not_exist", fixed_value_or_domain="south")
    with pytest.raises(GroundingError):
        pipeline.run(spec, groundings=(d,))


def test_grounding_region_target_records_ledger_only():
    # A grounding that fixes a region (not a relation) is recorded but changes no geometry.
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("EC"), "cAB"),),
    )
    d = GroundingDecision(id="g0", target="A", fixed_value_or_domain="x==4")
    r = pipeline.run(spec, groundings=(d,))
    assert r.status is Realizability.REALIZED_IN_D


def test_malformed_witness_unknown_region_raises():
    # Witness references a region not in the spec -> malformed -> fail-closed PipelineError.
    w = make_witness(("A", "Z"), 1, id="wAZ")
    spec = RelSpec(regions=_regions("A", "B"), constraints=(), witnesses=(w,))
    with pytest.raises(pipeline.PipelineError):
        pipeline.run(spec)


def test_empty_spec_realizes_trivially():
    spec = RelSpec(regions=_regions("A", "B"), constraints=())
    r = pipeline.run(spec, bounds=(0, 10))
    assert r.status is Realizability.REALIZED_IN_D
    assert r.satisfied == ()
    assert len(r.boxes) == 2
