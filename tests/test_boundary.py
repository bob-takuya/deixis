"""Tests for M4 shared-boundary identity (deixis.incidence.boundary).

Focus: one cell co-referenced by two regions' witnesses, boundary-region
recovery, regrounding through grounding_op (immutable), and the demo2 invariant
that a grounded wall keeps the shared cell id stable.
"""
import pytest

from deixis.core.types import (
    Cell,
    ContactDim,
    GroundingDecision,
    RelationConstraint,
    RelSpec,
    Region,
    Status,
)
from deixis.core.rcc8 import mask
from deixis.incidence.boundary import (
    BoundaryError,
    boundary_regions,
    demo_shared_wall,
    id_stability,
    make_shared_boundary,
    reground_boundary,
)


# ---------------------------------------------------------------- fixtures
def _two_room_spec():
    """Two rooms sharing wall F17 (dim 2), with a DELEGATED EC wall constraint."""
    a = Region(id="A", semantic_type="living")
    b = Region(id="B", semantic_type="bedroom")
    w_a, w_b = make_shared_boundary("A", "B", "F17", dim=2)
    wall = RelationConstraint(
        src="A", dst="B", rcc8_mask=mask("EC"), status=Status.DELEGATED, id="wall_AB"
    )
    spec = RelSpec(regions=(a, b), constraints=(wall,), witnesses=(w_a, w_b))
    return spec, w_a, w_b


def _decision(did="fix_wall", target="wall_AB"):
    return GroundingDecision(
        id=did, target=target, fixed_value_or_domain="x==Fraction(21,5)",
        actor="designer", rationale="pin wall F17",
    )


# ---------------------------------------------------------------- construction
def test_make_shared_boundary_coreferences_one_cell():
    w_a, w_b = make_shared_boundary("A", "B", "F17", dim=2)
    # both witnesses reference the SAME cell id -> the identity anchor
    assert "F17" in w_a.cell_ids
    assert "F17" in w_b.cell_ids
    assert w_a.cell_ids == w_b.cell_ids == ("F17",)
    # each witness sees both regions (shared boundary joins them)
    assert set(w_a.incident_regions) == {"A", "B"}
    assert set(w_b.incident_regions) == {"A", "B"}
    # same contact dimension, distinct stable ids (A-view vs B-view)
    assert w_a.intended_contact_dimension == 2
    assert w_b.intended_contact_dimension == 2
    assert w_a.id != w_b.id


def test_make_shared_boundary_accepts_contactdim_enum():
    w_a, w_b = make_shared_boundary("A", "B", "F17", dim=ContactDim.FACE)
    assert w_a.intended_contact_dimension == int(ContactDim.FACE) == 2
    assert w_b.intended_contact_dimension == 2


def test_make_shared_boundary_rejects_same_region():
    with pytest.raises(BoundaryError):
        make_shared_boundary("A", "A", "F17", dim=2)


def test_make_shared_boundary_rejects_empty_cell_id():
    with pytest.raises(BoundaryError):
        make_shared_boundary("A", "B", "", dim=2)


def test_make_shared_boundary_rejects_float_dim():
    with pytest.raises(TypeError):
        make_shared_boundary("A", "B", "F17", dim=2.0)


def test_make_shared_boundary_rejects_bool_dim():
    with pytest.raises(TypeError):
        make_shared_boundary("A", "B", "F17", dim=True)


def test_make_shared_boundary_rejects_str_dim():
    with pytest.raises(TypeError):
        make_shared_boundary("A", "B", "F17", dim="2")


def test_make_shared_boundary_rejects_out_of_range_dim():
    with pytest.raises(BoundaryError):
        make_shared_boundary("A", "B", "F17", dim=3)


# ---------------------------------------------------------------- boundary_regions
def test_boundary_regions_recovers_shared_pair():
    spec, _, _ = _two_room_spec()
    assert boundary_regions(spec, "F17") == frozenset({"A", "B"})


def test_boundary_regions_empty_for_unknown_cell():
    spec, _, _ = _two_room_spec()
    assert boundary_regions(spec, "F99") == frozenset()


def test_boundary_regions_unions_over_witnesses():
    # a third region C sharing the same cell F17 via its own witness
    a, b, c = Region(id="A"), Region(id="B"), Region(id="C")
    w_a, w_b = make_shared_boundary("A", "B", "F17", dim=2)
    w_c, _ = make_shared_boundary("C", "A", "F17", dim=2)
    spec = RelSpec(regions=(a, b, c), witnesses=(w_a, w_b, w_c))
    assert boundary_regions(spec, "F17") == frozenset({"A", "B", "C"})


# ---------------------------------------------------------------- reground
def test_reground_grounds_wall_constraint():
    spec, _, _ = _two_room_spec()
    after = reground_boundary(spec, "F17", _decision())
    # the boundary constraint moved DELEGATED -> GROUNDED via grounding_op
    assert after.constraint("wall_AB").status is Status.GROUNDED
    assert any(d.id == "fix_wall" for d in after.groundings)


def test_reground_is_immutable():
    spec, _, _ = _two_room_spec()
    snap_constraints = spec.constraints
    snap_groundings = spec.groundings
    snap_witnesses = spec.witnesses
    reground_boundary(spec, "F17", _decision())
    # input spec is untouched
    assert spec.constraints is snap_constraints
    assert spec.groundings is snap_groundings
    assert spec.witnesses is snap_witnesses
    assert spec.constraint("wall_AB").status is Status.DELEGATED
    assert spec.groundings == ()


def test_reground_keeps_cell_id_and_witnesses():
    spec, w_a, w_b = _two_room_spec()
    after = reground_boundary(spec, "F17", _decision())
    # witnesses (identity + cell_ids) carried through unchanged
    assert after.witnesses == (w_a, w_b)
    assert boundary_regions(after, "F17") == frozenset({"A", "B"})


def test_reground_routes_mismatched_target_to_constraint():
    # decision.target names the CELL, not the constraint -> reground re-points it
    spec, _, _ = _two_room_spec()
    dec = _decision(target="F17")
    after = reground_boundary(spec, "F17", dec)
    assert after.constraint("wall_AB").status is Status.GROUNDED
    # the routed decision now targets the wall constraint (grounding_op invariant)
    routed = next(d for d in after.groundings if d.id == "fix_wall")
    assert routed.target == "wall_AB"


def test_reground_without_constraint_is_fail_closed():
    # no constraint linking the two rooms: nothing to fix -> fail-closed
    a, b = Region(id="A"), Region(id="B")
    w_a, w_b = make_shared_boundary("A", "B", "F17", dim=2)
    spec = RelSpec(regions=(a, b), witnesses=(w_a, w_b))
    with pytest.raises(BoundaryError):
        reground_boundary(spec, "F17", _decision(target="F17"))


def test_reground_ignores_non_ec_constraint():
    # a DC constraint between the rooms is not a shared wall -> nothing to fix
    a, b = Region(id="A"), Region(id="B")
    w_a, w_b = make_shared_boundary("A", "B", "F17", dim=2)
    dc = RelationConstraint(
        src="A", dst="B", rcc8_mask=mask("DC"), status=Status.DELEGATED, id="sep_AB"
    )
    spec = RelSpec(regions=(a, b), constraints=(dc,), witnesses=(w_a, w_b))
    with pytest.raises(BoundaryError):
        reground_boundary(spec, "F17", _decision(target="F17"))
    # the DC constraint is left delegated (never grounded as a wall)
    assert spec.constraint("sep_AB").status is Status.DELEGATED


def test_reground_prefers_delegated_over_grounded_wall():
    # two EC constraints on the pair: one already GROUNDED, one DELEGATED.
    # the DELEGATED one must be chosen and grounded.
    a, b = Region(id="A"), Region(id="B")
    w_a, w_b = make_shared_boundary("A", "B", "F17", dim=2)
    g = RelationConstraint(
        src="A", dst="B", rcc8_mask=mask("EC"), status=Status.GROUNDED, id="wall_old"
    )
    d = RelationConstraint(
        src="A", dst="B", rcc8_mask=mask("EC"), status=Status.DELEGATED, id="wall_new"
    )
    spec = RelSpec(regions=(a, b), constraints=(g, d), witnesses=(w_a, w_b))
    after = reground_boundary(spec, "F17", _decision(target="F17"))
    assert after.constraint("wall_new").status is Status.GROUNDED
    assert after.constraint("wall_old").status is Status.GROUNDED  # untouched


def test_reground_regrounds_already_grounded_wall():
    # first fix, then move the wall again: unground+ground, cell id still stable
    spec, _, _ = _two_room_spec()
    once = reground_boundary(spec, "F17", _decision(did="fix1"))
    assert once.constraint("wall_AB").status is Status.GROUNDED
    twice = reground_boundary(once, "F17", _decision(did="fix2", target="F17"))
    assert twice.constraint("wall_AB").status is Status.GROUNDED
    # the old decision was released, the new one pins the wall
    assert not any(d.id == "fix1" for d in twice.groundings)
    assert any(d.id == "fix2" for d in twice.groundings)
    assert id_stability(once, twice, "F17") is True


def test_reground_rejects_invariant_wall():
    spec, _, _ = _two_room_spec()
    wall = RelationConstraint(
        src="A", dst="B", rcc8_mask=mask("EC"), status=Status.INVARIANT, id="wall_AB"
    )
    protected = RelSpec(regions=spec.regions, constraints=(wall,),
                        witnesses=spec.witnesses)
    with pytest.raises(BoundaryError):
        reground_boundary(protected, "F17", _decision())


def test_reground_rejects_unshared_cell():
    spec, _, _ = _two_room_spec()
    with pytest.raises(BoundaryError):
        reground_boundary(spec, "F99", _decision())


def test_reground_rejects_duplicate_decision_id():
    spec, _, _ = _two_room_spec()
    once = reground_boundary(spec, "F17", _decision())
    with pytest.raises(BoundaryError):
        reground_boundary(once, "F17", _decision())  # same id "fix_wall"


# ---------------------------------------------------------------- id stability
def test_id_stability_holds_across_reground():
    spec, _, _ = _two_room_spec()
    after = reground_boundary(spec, "F17", _decision())
    assert id_stability(spec, after, "F17") is True


def test_id_stability_survives_repeated_reground():
    # move (reground) the wall twice; F17 identity stays stable each step
    spec, _, _ = _two_room_spec()
    after1 = reground_boundary(spec, "F17", _decision(did="fix1"))
    assert id_stability(spec, after1, "F17") is True
    # ground is only P->G once; regrounding a second time (no delegatable
    # constraint left) still records the decision and preserves identity
    after2 = reground_boundary(after1, "F17", _decision(did="fix2", target="F17"))
    assert id_stability(after1, after2, "F17") is True
    assert boundary_regions(after2, "F17") == frozenset({"A", "B"})


def test_id_stability_false_when_cell_absent_before():
    spec, _, _ = _two_room_spec()
    assert id_stability(spec, spec, "F99") is False


def test_id_stability_false_when_witness_dropped():
    spec, w_a, w_b = _two_room_spec()
    # drop one room's witness on F17 -> identity no longer shared the same way
    forked = RelSpec(regions=spec.regions, constraints=spec.constraints,
                     witnesses=(w_a,))
    assert id_stability(spec, forked, "F17") is False


def test_id_stability_false_when_cell_renamed():
    spec, _, _ = _two_room_spec()
    # a spec where the wall was forked to a different cell id (identity broken)
    w_a2, w_b2 = make_shared_boundary("A", "B", "F18", dim=2)
    renamed = RelSpec(regions=spec.regions, constraints=spec.constraints,
                      witnesses=(w_a2, w_b2))
    assert id_stability(spec, renamed, "F17") is False


# ---------------------------------------------------------------- demo2
def test_demo_shared_wall():
    before, after, cell = demo_shared_wall()
    assert cell == Cell(id="F17", dim=2)
    # both rooms co-reference F17 before AND after grounding the wall position
    assert boundary_regions(before, "F17") == frozenset({"room_A", "room_B"})
    assert boundary_regions(after, "F17") == frozenset({"room_A", "room_B"})
    # the wall position got grounded (delegated -> grounded), decision recorded
    assert before.constraint("wall_AB").status is Status.DELEGATED
    assert after.constraint("wall_AB").status is Status.GROUNDED
    assert any(d.id == "fix_wall_F17" for d in after.groundings)
    # demo2 invariant: the shared cell id is stable across the reground
    assert id_stability(before, after, "F17") is True
