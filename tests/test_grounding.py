"""Tests for deferred-grounding operations (deixis.grounding.grounding_op).

Focus: immutability (input spec never mutated), status transitions, ground<->unground
round-trip, invariant fail-closed protection, and correct I/G/P partition.
"""
import pytest

from deixis.core.types import (
    GroundingDecision,
    RelationConstraint,
    RelSpec,
    Region,
    Status,
)
from deixis.core.rcc8 import mask
from deixis.grounding.grounding_op import (
    GroundingError,
    ground,
    partition,
    set_invariant,
    unground,
)


# ---------------------------------------------------------------- fixtures

def _base_spec() -> RelSpec:
    regions = (Region(id="A"), Region(id="B"), Region(id="C"))
    constraints = (
        RelationConstraint(src="A", dst="B", rcc8_mask=mask("EC"), id="c1"),   # DELEGATED
        RelationConstraint(src="B", dst="C", rcc8_mask=mask("DC"), id="c2"),   # DELEGATED
        RelationConstraint(src="A", dst="C", rcc8_mask=mask("PO"),
                           status=Status.GROUNDED, id="c3"),                   # GROUNDED
    )
    return RelSpec(regions=regions, constraints=constraints)


def _decision(target="c1", did="d1") -> GroundingDecision:
    return GroundingDecision(
        id=did, target=target, fixed_value_or_domain="x==south",
        actor="alice", rationale="site entry faces south",
    )


# ---------------------------------------------------------------- immutability

def test_ground_does_not_mutate_input():
    spec = _base_spec()
    snapshot_constraints = spec.constraints
    snapshot_groundings = spec.groundings

    new = ground(spec, "c1", _decision())

    # original untouched (same tuple objects, same statuses, empty ledger)
    assert spec.constraints is snapshot_constraints
    assert spec.groundings is snapshot_groundings
    assert spec.constraint("c1").status is Status.DELEGATED
    assert spec.groundings == ()
    # new is a different object with the change applied
    assert new is not spec
    assert new.constraint("c1").status is Status.GROUNDED


def test_set_invariant_does_not_mutate_input():
    spec = _base_spec()
    new = set_invariant(spec, "c1")
    assert spec.constraint("c1").status is Status.DELEGATED
    assert spec.invariant.relation_invariants == frozenset()
    assert new.constraint("c1").status is Status.INVARIANT
    assert new.invariant.relation_invariants == frozenset({"c1"})


# ---------------------------------------------------------------- status transitions

def test_ground_delegated_to_grounded():
    spec = _base_spec()
    new = ground(spec, "c1", _decision())
    assert new.constraint("c1").status is Status.GROUNDED
    assert len(new.groundings) == 1
    assert new.groundings[0].id == "d1"
    # provenance recorded on the constraint
    prov = new.constraint("c1").prov
    assert prov is not None
    assert prov.origin == "ground"
    assert prov.inputs == ("d1",)


def test_ground_rejects_non_delegated():
    spec = _base_spec()
    # c3 is already GROUNDED
    with pytest.raises(GroundingError):
        ground(spec, "c3", _decision(target="c3", did="dX"))


def test_ground_rejects_unknown_target():
    spec = _base_spec()
    with pytest.raises(GroundingError):
        ground(spec, "nope", _decision(target="nope"))


def test_ground_rejects_target_mismatch():
    spec = _base_spec()
    with pytest.raises(GroundingError):
        ground(spec, "c1", _decision(target="c2"))


def test_ground_rejects_duplicate_decision_id():
    spec = _base_spec()
    once = ground(spec, "c1", _decision(target="c1", did="d1"))
    with pytest.raises(GroundingError):
        ground(once, "c2", _decision(target="c2", did="d1"))


# ---------------------------------------------------------------- round-trip

def test_ground_then_unground_round_trip():
    spec = _base_spec()
    grounded = ground(spec, "c1", _decision())
    reverted = unground(grounded, "d1")

    assert reverted.constraint("c1").status is Status.DELEGATED
    assert reverted.groundings == ()
    # the intermediate grounded spec is itself unchanged (immutability across the trip)
    assert grounded.constraint("c1").status is Status.GROUNDED
    assert len(grounded.groundings) == 1
    # provenance reflects the revert
    assert reverted.constraint("c1").prov.origin == "unground"


def test_unground_unknown_decision():
    spec = _base_spec()
    with pytest.raises(GroundingError):
        unground(spec, "ghost")


def test_unground_rejects_when_not_grounded():
    # decision points at a constraint that is DELEGATED (inconsistent ledger) -> fail-closed
    regions = (Region(id="A"), Region(id="B"))
    constraints = (RelationConstraint(src="A", dst="B", rcc8_mask=mask("EC"), id="c1"),)
    stray = GroundingDecision(id="d1", target="c1", fixed_value_or_domain="?")
    spec = RelSpec(regions=regions, constraints=constraints, groundings=(stray,))
    with pytest.raises(GroundingError):
        unground(spec, "d1")


# ---------------------------------------------------------------- invariant protection

def test_ground_on_invariant_is_fail_closed():
    spec = set_invariant(_base_spec(), "c1")
    with pytest.raises(GroundingError):
        ground(spec, "c1", _decision())


def test_unground_on_promoted_invariant_is_fail_closed():
    spec = _base_spec()
    grounded = ground(spec, "c1", _decision())          # c1 GROUNDED, decision d1 in ledger
    promoted = set_invariant(grounded, "c1")            # c1 now INVARIANT (protected)
    # the decision is still in the ledger, but unground must refuse to touch the invariant
    assert promoted.constraint("c1").status is Status.INVARIANT
    with pytest.raises(GroundingError):
        unground(promoted, "d1")


def test_set_invariant_idempotent():
    spec = set_invariant(_base_spec(), "c1")
    again = set_invariant(spec, "c1")
    assert again.constraint("c1").status is Status.INVARIANT
    assert again.invariant.relation_invariants == frozenset({"c1"})


def test_set_invariant_from_grounded():
    spec = _base_spec()
    # c3 starts GROUNDED; promoting it should still work and register the id
    new = set_invariant(spec, "c3")
    assert new.constraint("c3").status is Status.INVARIANT
    assert "c3" in new.invariant.relation_invariants


def test_set_invariant_unknown_target():
    spec = _base_spec()
    with pytest.raises(GroundingError):
        set_invariant(spec, "nope")


# ---------------------------------------------------------------- partition

def test_partition_splits_by_status():
    spec = _base_spec()
    spec = set_invariant(spec, "c2")   # c2 -> INVARIANT ; c1 DELEGATED ; c3 GROUNDED
    inv, grn, dele = partition(spec)

    assert tuple(c.id for c in inv) == ("c2",)
    assert tuple(c.id for c in grn) == ("c3",)
    assert tuple(c.id for c in dele) == ("c1",)
    # total coverage: every constraint appears exactly once
    assert len(inv) + len(grn) + len(dele) == len(spec.constraints)


def test_partition_preserves_order():
    regions = (Region(id="A"), Region(id="B"))
    cs = tuple(
        RelationConstraint(src="A", dst="B", rcc8_mask=mask("EC"),
                           status=Status.DELEGATED, id=f"c{i}")
        for i in range(5)
    )
    spec = RelSpec(regions=regions, constraints=cs)
    _, _, dele = partition(spec)
    assert tuple(c.id for c in dele) == ("c0", "c1", "c2", "c3", "c4")


def test_partition_empty():
    spec = RelSpec()
    inv, grn, dele = partition(spec)
    assert inv == () and grn == () and dele == ()
