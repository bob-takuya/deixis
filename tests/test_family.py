"""Tests for M6 solution-family generation and comparison (deixis.solver.family).

Covers: family generation over grounding scenarios; the demo3 claim that the diff
between two forms is *only* the grounded part (invariant spec preserved) relative to a
pre-registered observation set O; canonical_key identifying symmetric/translated
duplicates while distinguishing genuinely different order-types; and decision_provenance
reverse-tracing which decision grounded which constraint.
"""
from __future__ import annotations

from fractions import Fraction as Fr

import pytest

from deixis.core import rcc8
from deixis.core.types import (
    Box,
    GroundingDecision,
    ObservationQuery,
    Realizability,
    Realization,
    Region,
    RelationConstraint,
    RelSpec,
    Status,
)
from deixis.grounding.grounding_op import set_invariant
from deixis.solver import family


# ---------------------------------------------------------------- builders
def _regions(*ids: str) -> tuple[Region, ...]:
    return tuple(Region(id=i) for i in ids)


def _c(src, dst, mask, cid, status=Status.DELEGATED) -> RelationConstraint:
    return RelationConstraint(src=src, dst=dst, rcc8_mask=mask, status=status, id=cid)


def _box(rid, lo, hi) -> Box:
    return Box(region_id=rid, lo=tuple(Fr(v) for v in lo), hi=tuple(Fr(v) for v in hi))


def _real(*boxes, rid="r") -> Realization:
    return Realization(id=rid, boxes=tuple(boxes), status=Realizability.REALIZED_IN_D)


# ================================================================ solve_family
def test_solve_family_one_realization_per_scenario_with_ids():
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.UNIVERSAL, "cAB"),),
    )
    scenarios = [
        (GroundingDecision(id="gDC", target="cAB", fixed_value_or_domain="DC"),),
        (GroundingDecision(id="gEC", target="cAB", fixed_value_or_domain="EC"),),
    ]
    reals = family.solve_family(spec, scenarios, bounds=(0, 100))
    assert len(reals) == 2
    assert [r.scenario_id for r in reals] == ["s0", "s1"]
    assert all(r.status is Realizability.REALIZED_IN_D for r in reals)


def test_solve_family_shares_immutable_base_spec():
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.UNIVERSAL, "cAB"),),
    )
    scenarios = [
        (GroundingDecision(id="gEC", target="cAB", fixed_value_or_domain="EC"),),
    ]
    family.solve_family(spec, scenarios)
    # base spec untouched by branching.
    assert spec.constraints[0].status is Status.DELEGATED
    assert spec.constraints[0].rcc8_mask == rcc8.UNIVERSAL
    assert spec.groundings == ()


# ================================================================ demo3
def _demo3_spec() -> RelSpec:
    """A un-grounded spec: B-C is the invariant skeleton; A's relations are delegated."""
    return RelSpec(
        regions=_regions("A", "B", "C"),
        constraints=(
            _c("B", "C", rcc8.mask("EC"), "cBC", status=Status.INVARIANT),  # invariant core
            _c("A", "B", rcc8.UNIVERSAL, "cAB"),                             # delegated
            _c("A", "C", rcc8.UNIVERSAL, "cAC"),                             # delegated
        ),
    )


def test_demo3_two_forms_differ_only_in_grounded_part():
    spec = _demo3_spec()
    # form "south": A disconnected from both B and C.
    south = (
        GroundingDecision(id="s_AB", target="cAB", fixed_value_or_domain="DC"),
        GroundingDecision(id="s_AC", target="cAC", fixed_value_or_domain="DC"),
    )
    # form "north": A now touches B (EC), still disconnected from C.
    north = (
        GroundingDecision(id="n_AB", target="cAB", fixed_value_or_domain="EC"),
        GroundingDecision(id="n_AC", target="cAC", fixed_value_or_domain="DC"),
    )
    reals = family.solve_family(spec, [south, north], bounds=(0, 100))
    assert all(r.status is Realizability.REALIZED_IN_D for r in reals)

    diff = family.compare_realizations(reals[0], reals[1])
    changed, preserved = diff["changed"], diff["preserved"]

    # The invariant B-C relation is PRESERVED (EC in both forms).
    assert preserved[("rcc8", "B", "C")] == "EC"
    # A-C was grounded to DC in both forms -> preserved.
    assert preserved[("rcc8", "A", "C")] == "DC"
    # Only the grounded A-B relation CHANGED (DC in south, EC in north).
    assert changed[("rcc8", "A", "B")] == ("DC", "EC")
    # ...and nothing about B-C or A-C leaked into the changed set.
    assert not any(k[1:] == ("B", "C") for k in changed)
    assert not any(k[1:] == ("A", "C") for k in changed)


def test_compare_respects_observation_set_O():
    # If O excludes rcc8, the RCC change is not observed at all.
    spec = _demo3_spec()
    south = (
        GroundingDecision(id="s_AB", target="cAB", fixed_value_or_domain="DC"),
        GroundingDecision(id="s_AC", target="cAC", fixed_value_or_domain="DC"),
    )
    north = (
        GroundingDecision(id="n_AB", target="cAB", fixed_value_or_domain="EC"),
        GroundingDecision(id="n_AC", target="cAC", fixed_value_or_domain="DC"),
    )
    reals = family.solve_family(spec, [south, north], bounds=(0, 100))
    o = ObservationQuery(kinds=frozenset({"cell_identity"}))
    diff = family.compare_realizations(reals[0], reals[1], o)
    assert diff["changed"] == {}
    # both forms have the same three regions -> all preserved under cell_identity
    # (realization_status is always compared and is REALIZED_IN_D for both).
    assert diff["preserved"][("cell_identity", "A")] is True
    assert diff["preserved"][("cell_identity", "B")] is True
    assert diff["preserved"][("cell_identity", "C")] is True
    assert diff["preserved"][("realization_status",)] == "realized_in_selected_domain_D"


def test_compare_occupancy_and_contact_dimension():
    # PO overlap => occupied; EC edge contact => contact dimension 1 (LINE) in 2D.
    po = _real(_box("A", (0, 0), (4, 4)), _box("B", (2, 2), (6, 6)))     # partial overlap
    ec = _real(_box("A", (0, 0), (4, 4)), _box("B", (4, 0), (8, 4)))     # edge-share
    diff = family.compare_realizations(po, ec)
    changed = diff["changed"]
    # occupancy flips true->false (overlap vs boundary-only touch).
    assert changed[("atom_occupancy", "A", "B")] == (True, False)
    # contact_dimension exists only for the EC realization (absent on the PO side).
    assert ("contact_dimension", "A", "B") in changed
    absent_side, dim_side = changed[("contact_dimension", "A", "B")]
    assert absent_side == family._ABSENT and dim_side == 1


def test_compare_unknown_observation_kind_is_fail_closed():
    a = _real(_box("A", (0, 0), (1, 1)))
    b = _real(_box("A", (0, 0), (1, 1)))
    with pytest.raises(ValueError):
        family.compare_realizations(a, b, ObservationQuery(kinds=frozenset({"nope"})))


def test_compare_flags_non_realized_status_not_vacuous_preserve():
    # Two empty (boxes=()) results must NOT vacuously "preserve" everything: their
    # differing realizability status is surfaced as a change.
    unknown = Realization(id="u", boxes=(), status=Realizability.UNKNOWN)
    inconsistent = Realization(id="i", boxes=(), status=Realizability.INCONSISTENT)
    diff = family.compare_realizations(unknown, inconsistent)
    assert diff["changed"][("realization_status",)] == ("unknown", "inconsistent")


def test_compare_status_preserved_when_equal():
    a = _real(_box("A", (0, 0), (1, 1)))
    b = _real(_box("A", (0, 0), (1, 1)))
    diff = family.compare_realizations(a, b)
    assert diff["preserved"][("realization_status",)] == "realized_in_selected_domain_D"


# ================================================================ canonical_key
def test_canonical_key_rejects_duplicate_region():
    dup = _real(_box("A", (0, 0), (1, 1)), _box("A", (2, 0), (3, 1)))
    with pytest.raises(ValueError):
        family.canonical_key(dup)
def test_canonical_key_translation_invariant():
    r1 = _real(_box("A", (0, 0), (2, 2)), _box("B", (2, 0), (4, 2)))
    # same arrangement shifted by +10 on both axes.
    r2 = _real(_box("A", (10, 10), (12, 12)), _box("B", (12, 10), (14, 12)))
    assert family.canonical_key(r1) == family.canonical_key(r2)


def test_canonical_key_reflection_invariant():
    # A left of B, touching.
    r1 = _real(_box("A", (0, 0), (2, 2)), _box("B", (2, 0), (4, 2)))
    # horizontal mirror: A now right of B (same region labels).
    r2 = _real(_box("A", (2, 0), (4, 2)), _box("B", (0, 0), (2, 2)))
    assert family.canonical_key(r1) == family.canonical_key(r2)


def test_canonical_key_scale_invariant():
    r1 = _real(_box("A", (0, 0), (2, 2)), _box("B", (2, 0), (4, 2)))
    # same order-type, non-uniformly stretched.
    r2 = _real(_box("A", (0, 0), (5, 9)), _box("B", (5, 0), (99, 9)))
    assert family.canonical_key(r1) == family.canonical_key(r2)


def test_canonical_key_distinguishes_different_order_types():
    touching = _real(_box("A", (0, 0), (2, 2)), _box("B", (2, 0), (4, 2)))   # EC
    disjoint = _real(_box("A", (0, 0), (2, 2)), _box("B", (5, 0), (7, 2)))   # DC (gap)
    assert family.canonical_key(touching) != family.canonical_key(disjoint)


def test_canonical_key_collapses_symmetric_family_duplicates():
    # Two solver outputs that are mirror images collapse to one key: symmetric solutions
    # are NOT double-counted when we quotient a family by canonical_key.
    left = _real(_box("A", (0, 0), (3, 3)), _box("B", (3, 0), (5, 3)))
    right = _real(_box("A", (2, 0), (5, 3)), _box("B", (0, 0), (2, 3)))
    keys = {family.canonical_key(left), family.canonical_key(right)}
    assert len(keys) == 1


def test_canonical_key_empty_realization():
    assert family.canonical_key(_real()) == ()


# ================================================================ decision_provenance
def test_decision_provenance_traces_constraint_to_decision():
    spec = RelSpec(
        regions=_regions("A", "B", "C"),
        constraints=(
            _c("A", "B", rcc8.UNIVERSAL, "cAB"),
            _c("A", "C", rcc8.UNIVERSAL, "cAC"),
        ),
    )
    scenario = (
        GroundingDecision(id="dEC", target="cAB", fixed_value_or_domain="EC"),
        GroundingDecision(id="dDC", target="cAC", fixed_value_or_domain="DC"),
    )
    reals = family.solve_family(spec, [scenario], bounds=(0, 100))
    prov = family.decision_provenance(reals[0])
    assert prov == {"cAB": "dEC", "cAC": "dDC"}


def test_decision_provenance_ignores_region_target_decisions():
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("EC"), "cAB"),),
    )
    scenario = (
        # targets a REGION, not a constraint -> no constraint mapping recorded.
        GroundingDecision(id="dReg", target="A", fixed_value_or_domain="x==4"),
    )
    reals = family.solve_family(spec, [scenario], bounds=(0, 100))
    assert family.decision_provenance(reals[0]) == {}


def test_decision_provenance_empty_for_non_family_realization():
    # A bare realization (no family provenance) honestly reports no recorded decisions.
    assert family.decision_provenance(_real(_box("A", (0, 0), (1, 1)))) == {}


def test_provenance_survives_invariant_core_untouched():
    # Grounding only the delegated constraints; the invariant one is never in the ledger.
    spec = _demo3_spec()
    spec = set_invariant(spec, "cBC")  # already invariant, registers in the core
    scenario = (
        GroundingDecision(id="g1", target="cAB", fixed_value_or_domain="EC"),
    )
    reals = family.solve_family(spec, [scenario], bounds=(0, 100))
    prov = family.decision_provenance(reals[0])
    assert "cBC" not in prov
    assert prov == {"cAB": "g1"}
