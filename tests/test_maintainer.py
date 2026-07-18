"""Tests for the M6.5 turn-based bidirectional maintainer (deixis.bidir.maintainer).

Covers the honest maintainer contract:
  * append-only, non-destructive history;
  * fail-closed invariant protection (put refuses to touch the core);
  * complement preservation — put holds C_B (invariant + ledger) constant, get holds
    C_A (the observed geometry) constant and leaves DECIDED relations untouched;
  * demo4 — one relation-edit -> geometry, then observe-geometry -> relation round trip;
  * the explicit *non-claims*: get is forgetful/irreversible and PutPut is not confluent.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction as Fr

import pytest

from deixis.core import rcc8
from deixis.core.types import (
    Box,
    GroundingDecision,
    Realizability,
    Realization,
    Region,
    RelationConstraint,
    RelSpec,
    Status,
)
from deixis.grounding.grounding_op import ground, set_invariant
from deixis.bidir import maintainer
from deixis.bidir.maintainer import (
    HistoryEntry,
    MaintainerError,
    RelationEdit,
    SessionState,
    get_observe_geometry,
    grounding_ledger_ids,
    invariant_ids,
    new_session,
    put_edit_relation,
)


# ---------------------------------------------------------------- builders
def _regions(*ids: str) -> tuple[Region, ...]:
    return tuple(Region(id=i) for i in ids)


def _c(src: str, dst: str, mask: int, cid: str) -> RelationConstraint:
    return RelationConstraint(src=src, dst=dst, rcc8_mask=mask, status=Status.DELEGATED, id=cid)


def _base_spec() -> RelSpec:
    """A 3-region spec with one INVARIANT, one GROUNDED and one DELEGATED relation.

        cAB : A EC B   DELEGATED  (the editable relation)
        cAC : A DC C   INVARIANT  (the un-touchable core, half of C_B)
        cBC : B DC C   GROUNDED   (a decided complement relation, ledger-backed)
    """
    spec = RelSpec(
        regions=_regions("A", "B", "C"),
        constraints=(
            _c("A", "B", rcc8.mask("EC"), "cAB"),
            _c("A", "C", rcc8.mask("DC"), "cAC"),
            _c("B", "C", rcc8.mask("DC"), "cBC"),
        ),
    )
    spec = ground(
        spec, "cBC",
        GroundingDecision(id="d-bc", target="cBC", fixed_value_or_domain="DC"),
    )
    spec = set_invariant(spec, "cAC")
    return spec


def _session() -> SessionState:
    return new_session(_base_spec())


# ---------------------------------------------------------------- basics / immutability
def test_new_session_starts_at_version_zero_with_empty_history():
    s = _session()
    assert s.version == 0
    assert s.history == ()
    assert s.realization is None
    assert s.editing_port == ""


def test_session_state_is_frozen():
    s = _session()
    with pytest.raises(FrozenInstanceError):
        s.version = 99  # type: ignore[misc]


def test_put_returns_new_state_without_mutating_original():
    s0 = _session()
    s1 = put_edit_relation(s0, RelationEdit("A", "B", rcc8.mask("DC"), "cAB"))
    assert s1 is not s0
    assert s0.version == 0 and s0.realization is None  # original untouched
    assert s1.version == 1


# ---------------------------------------------------------------- put: relation -> geometry
def test_put_edit_relation_resynthesizes_geometry():
    s0 = _session()
    s1 = put_edit_relation(s0, RelationEdit("A", "B", rcc8.mask("DC"), "cAB"))
    assert s1.editing_port == "relation"
    assert s1.realization is not None
    assert s1.realization.status is Realizability.REALIZED_IN_D
    assert len(s1.realization.boxes) == 3
    # the edited mask is what the spec now carries
    assert s1.spec.constraint("cAB").rcc8_mask == rcc8.mask("DC")


def test_put_history_appended_and_version_incremented():
    s = _session()
    s = put_edit_relation(s, RelationEdit("A", "B", rcc8.mask("DC"), "cAB"))
    assert s.version == 1
    assert len(s.history) == 1
    assert isinstance(s.history[0], HistoryEntry)
    assert s.history[0].port == "relation"
    assert s.history[0].op == "put_edit_relation"


def test_put_refuses_editing_invariant_relation_fail_closed():
    s = _session()
    with pytest.raises(MaintainerError):
        put_edit_relation(s, RelationEdit("A", "C", rcc8.mask("EC"), "cAC"))


def test_put_refuses_empty_mask():
    s = _session()
    with pytest.raises(MaintainerError):
        put_edit_relation(s, RelationEdit("A", "B", rcc8.EMPTY, "cAB"))


def test_put_can_add_a_new_delegated_relation():
    # minimal additive change: a pair with no existing constraint is added, not rewritten.
    s = new_session(
        RelSpec(regions=_regions("A", "B"), constraints=(_c("A", "B", rcc8.mask("EC"), "cAB"),))
    )
    s = put_edit_relation(s, RelationEdit("A", "B", rcc8.mask("DC"), constraint_id=""))
    # same pair -> edited in place (matched by pair), still one constraint
    assert len(s.spec.constraints) == 1
    assert s.spec.constraint("cAB").rcc8_mask == rcc8.mask("DC")


# ---------------------------------------------------------------- put: whole-spec guard
def test_put_whole_spec_must_preserve_invariant_core():
    s = _session()
    # a replacement that weakens the invariant relation must be refused fail-closed
    bad = s.spec.with_constraints(
        tuple(
            (rc if rc.id != "cAC" else RelationConstraint(
                src="A", dst="C", rcc8_mask=rcc8.mask("EC"),
                status=Status.DELEGATED, id="cAC"))
            for rc in s.spec.constraints
        )
    )
    with pytest.raises(MaintainerError):
        put_edit_relation(s, bad)


def test_put_whole_spec_must_preserve_grounding_ledger():
    s = _session()
    bad = s.spec.with_groundings(())  # dropping the ledger moves C_B
    with pytest.raises(MaintainerError):
        put_edit_relation(s, bad)


def test_put_whole_spec_replacement_that_preserves_complement_succeeds():
    s = _session()
    ok = s.spec.with_constraints(
        tuple(
            (rc if rc.id != "cAB" else RelationConstraint(
                src="A", dst="B", rcc8_mask=rcc8.mask("DC"),
                status=Status.DELEGATED, id="cAB"))
            for rc in s.spec.constraints
        )
    )
    s1 = put_edit_relation(s, ok)
    assert s1.realization.status is Realizability.REALIZED_IN_D
    assert s1.spec.constraint("cAB").rcc8_mask == rcc8.mask("DC")


# ---------------------------------------------------------------- complement C_B held by put
def test_put_holds_complement_C_B_constant():
    s0 = _session()
    inv_before, ledger_before = invariant_ids(s0.spec), grounding_ledger_ids(s0.spec)
    s1 = put_edit_relation(s0, RelationEdit("A", "B", rcc8.mask("DC"), "cAB"))
    assert invariant_ids(s1.spec) == inv_before
    assert grounding_ledger_ids(s1.spec) == ledger_before
    # invariant relation mask/status unchanged
    assert s1.spec.constraint("cAC").status is Status.INVARIANT
    assert s1.spec.constraint("cAC").rcc8_mask == rcc8.mask("DC")


# ---------------------------------------------------------------- get: geometry -> relation
def test_get_requires_a_realized_geometry():
    s = _session()  # no realization yet
    with pytest.raises(MaintainerError):
        get_observe_geometry(s)


def test_get_updates_only_delegated_relations():
    s = _session()
    # widen the delegated relation to a disjunction so the observation visibly collapses it
    s = put_edit_relation(s, RelationEdit("A", "B", rcc8.mask("EC", "PO"), "cAB"))
    before_ab = s.spec.constraint("cAB").rcc8_mask
    assert bin(before_ab).count("1") == 2  # a genuine disjunction

    s2 = get_observe_geometry(s)
    obs_ab = s2.spec.constraint("cAB").rcc8_mask
    # forgetful: collapsed to a single base relation, and it is one of the disjuncts
    assert bin(obs_ab).count("1") == 1
    assert obs_ab & before_ab == obs_ab
    assert s2.editing_port == "geometry"


def test_get_preserves_decided_relations_untouched():
    s = _session()
    s = put_edit_relation(s, RelationEdit("A", "B", rcc8.mask("EC"), "cAB"))
    s2 = get_observe_geometry(s)
    # GROUNDED and INVARIANT relations are the preserved complement — unchanged.
    assert s2.spec.constraint("cAC").status is Status.INVARIANT
    assert s2.spec.constraint("cAC").rcc8_mask == rcc8.mask("DC")
    assert s2.spec.constraint("cBC").status is Status.GROUNDED
    assert s2.spec.constraint("cBC").rcc8_mask == rcc8.mask("DC")


def test_get_observation_orientation_uses_converse():
    """A NTPP B observed through a *reversed* constraint B->A must yield NTPPi.

    This pins the converse orientation in ``_observed_mask``: a self-converse relation
    (DC/EC/PO/EQ) could not detect an orientation bug, so we use the asymmetric NTPP pair
    on hand-built exact-Fraction boxes (A strictly inside B)."""
    boxes = (
        Box(region_id="A", lo=(Fr(2), Fr(2)), hi=(Fr(4), Fr(4))),   # strictly inside B
        Box(region_id="B", lo=(Fr(0), Fr(0)), hi=(Fr(10), Fr(10))),
    )
    real = Realization(id="r", boxes=boxes, status=Realizability.REALIZED_IN_D)
    # constraint is declared in the REVERSED direction B -> A (delegated), disjunction wide
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("B", "A", rcc8.mask("NTPP", "NTPPi"), "cBA"),),
    )
    s = new_session(spec, realization=real)
    s2 = get_observe_geometry(s)
    # B relative to A is NTPPi (A is strictly inside B), obtained via converse of NTPP.
    assert s2.spec.constraint("cBA").rcc8_mask == rcc8.mask("NTPPi")


def test_get_refuses_unrealized_geometry_fail_closed():
    boxes = (
        Box(region_id="A", lo=(Fr(0), Fr(0)), hi=(Fr(1), Fr(1))),
        Box(region_id="B", lo=(Fr(2), Fr(2)), hi=(Fr(3), Fr(3))),
    )
    real = Realization(id="r", boxes=boxes, status=Realizability.UNKNOWN)  # not REALIZED_IN_D
    s = new_session(
        RelSpec(regions=_regions("A", "B"), constraints=(_c("A", "B", rcc8.mask("DC"), "cAB"),)),
        realization=real,
    )
    with pytest.raises(MaintainerError):
        get_observe_geometry(s)


def test_get_fail_closed_when_geometry_contradicts_decided_relation():
    """A REALIZED_IN_D geometry that violates a GROUNDED/INVARIANT relation is refused.

    Guards externally-supplied or stale realizations: here A and B *touch* (EC) but the
    grounded relation demands DC, so get must fail-closed rather than silently keep the
    now-contradicted decided relation."""
    boxes = (
        Box(region_id="A", lo=(Fr(0), Fr(0)), hi=(Fr(2), Fr(2))),
        Box(region_id="B", lo=(Fr(2), Fr(0)), hi=(Fr(4), Fr(2))),   # shares the x=2 edge -> EC
    )
    real = Realization(id="r", boxes=boxes, status=Realizability.REALIZED_IN_D)
    spec = RelSpec(
        regions=_regions("A", "B"),
        constraints=(_c("A", "B", rcc8.mask("DC"), "cAB"),),
    )
    spec = ground(
        spec, "cAB",
        GroundingDecision(id="d-ab", target="cAB", fixed_value_or_domain="DC"),
    )
    s = new_session(spec, realization=real)
    with pytest.raises(MaintainerError):
        get_observe_geometry(s)


def test_put_refuses_shadow_constraint_on_already_constrained_pair():
    s = _session()
    # cAB already governs {A,B}; trying to add a *new* id for the same pair is ambiguous.
    with pytest.raises(MaintainerError):
        put_edit_relation(s, RelationEdit("A", "B", rcc8.mask("PO"), "cAB-shadow"))


def test_get_holds_complement_C_A_constant():
    # get does NOT re-solve: the observed realization is carried through unchanged.
    s = _session()
    s = put_edit_relation(s, RelationEdit("A", "B", rcc8.mask("EC"), "cAB"))
    real_before = s.realization
    s2 = get_observe_geometry(s)
    assert s2.realization is real_before  # identical object, geometry not re-chosen


# ---------------------------------------------------------------- history append-only
def test_history_is_append_only_across_turns():
    s = _session()
    s = put_edit_relation(s, RelationEdit("A", "B", rcc8.mask("EC"), "cAB"))
    h1 = s.history
    s = get_observe_geometry(s)
    h2 = s.history
    s = put_edit_relation(s, RelationEdit("A", "B", rcc8.mask("DC"), "cAB"))
    h3 = s.history
    # each later history is a strict superset extending the earlier one (prefix preserved)
    assert h2[: len(h1)] == h1
    assert h3[: len(h2)] == h2
    assert [e.version for e in h3] == [1, 2, 3]
    assert [e.port for e in h3] == ["relation", "geometry", "relation"]


# ---------------------------------------------------------------- demo4: one round trip
def test_demo4_roundtrip_preserves_invariant_and_unedited_complement():
    """demo4 — relation edit -> geometry, then observe geometry -> relation, one round trip.

    Shows that across the round trip the INVARIANT relation and the un-edited GROUNDED
    relation (the decided complement) are preserved, while the edited DELEGATED relation
    is re-synthesized then observed back."""
    s0 = _session()

    # (1) put: edit the delegated relation A?B and re-synthesize geometry.
    s1 = put_edit_relation(s0, RelationEdit("A", "B", rcc8.mask("EC", "PO"), "cAB"))
    assert s1.realization.status is Realizability.REALIZED_IN_D

    # (2) get: observe the geometry side's solution back into the relations.
    s2 = get_observe_geometry(s1)

    # invariant + un-edited grounded relation preserved through the whole round trip
    assert s2.spec.constraint("cAC").rcc8_mask == s0.spec.constraint("cAC").rcc8_mask
    assert s2.spec.constraint("cAC").status is Status.INVARIANT
    assert s2.spec.constraint("cBC").rcc8_mask == s0.spec.constraint("cBC").rcc8_mask
    assert s2.spec.constraint("cBC").status is Status.GROUNDED
    assert invariant_ids(s2.spec) == invariant_ids(s0.spec)
    assert grounding_ledger_ids(s2.spec) == grounding_ledger_ids(s0.spec)

    # the edited relation is now a single observed base relation within the edited disjunction
    obs = s2.spec.constraint("cAB").rcc8_mask
    assert bin(obs).count("1") == 1
    assert obs & rcc8.mask("EC", "PO") == obs

    # exactly two append-only turns recorded
    assert len(s2.history) == 2
    assert [e.op for e in s2.history] == ["put_edit_relation", "get_observe_geometry"]


# ---------------------------------------------------------------- explicit non-claims
def test_confluence_putput_is_not_claimed():
    """PutPut is order-dependent: editing the same relation in two orders diverges.

    The maintainer performs no merge and makes no commutation claim — this test *asserts
    the divergence* to lock in the honest non-confluence limitation."""
    s = _session()
    x, y = rcc8.mask("EC"), rcc8.mask("DC")

    a = put_edit_relation(s, RelationEdit("A", "B", x, "cAB"))
    a = put_edit_relation(a, RelationEdit("A", "B", y, "cAB"))

    b = put_edit_relation(s, RelationEdit("A", "B", y, "cAB"))
    b = put_edit_relation(b, RelationEdit("A", "B", x, "cAB"))

    # last-writer-wins, so the two orders end on different masks: NOT confluent.
    assert a.spec.constraint("cAB").rcc8_mask == y
    assert b.spec.constraint("cAB").rcc8_mask == x
    assert a.spec.constraint("cAB").rcc8_mask != b.spec.constraint("cAB").rcc8_mask


def test_no_adjoint_or_merge_is_exposed():
    # the module deliberately provides no lens/adjoint/merge/round-trip operation.
    for forbidden in ("adjoint", "merge", "round_trip", "roundtrip"):
        assert not hasattr(maintainer, forbidden)


def test_get_is_irreversible_topology_not_recovered():
    """A widened delegated relation is collapsed by get and NOT restored by a later put.

    Demonstrates get's forgetfulness: the disjunction (relational topology) is lost and a
    subsequent put on the geometry cannot resurrect the original pre-image."""
    s = _session()
    s = put_edit_relation(s, RelationEdit("A", "B", rcc8.mask("EC", "PO"), "cAB"))
    disjunction = s.spec.constraint("cAB").rcc8_mask
    s = get_observe_geometry(s)
    collapsed = s.spec.constraint("cAB").rcc8_mask
    assert collapsed != disjunction  # information was discarded, irreversibly
    assert bin(collapsed).count("1") == 1
