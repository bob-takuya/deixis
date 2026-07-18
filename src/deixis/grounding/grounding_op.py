"""Deferred-grounding operations over the immutable RelSpec (M0 §deferred I/G/P).

Every operation is a *pure function* of a RelSpec: it never mutates the input and
always returns a NEW RelSpec (frozen objects rebuilt via ``dataclasses.replace``).
This is what lets branched Grasshopper wires share a spec without order-dependent
mutation bugs.

The edit-authority lattice of a RelationConstraint is::

    DELEGATED (P)  --ground-->    GROUNDED (G)
    GROUNDED (G)   --unground-->  DELEGATED (P)
    (any)          --set_invariant--> INVARIANT (I)

INVARIANT (I) is the un-touchable core. ``ground`` / ``unground`` are *fail-closed*
against it: an attempt to ground or unground an INVARIANT constraint raises
``GroundingError`` rather than silently corrupting the protected core.
"""
from __future__ import annotations

from dataclasses import replace

from ..core.ids import Provenance
from ..core.types import (
    GroundingDecision,
    Invariant,
    RelationConstraint,
    RelSpec,
    Status,
)

__all__ = [
    "GroundingError",
    "ground",
    "unground",
    "set_invariant",
    "partition",
]


class GroundingError(Exception):
    """Raised when a grounding operation is illegal (fail-closed guard, missing id, ...)."""


# ---------------------------------------------------------------- internal helpers

def _find_constraint_index(spec: RelSpec, cid: str) -> int:
    """Return the index of the constraint with id ``cid`` or raise GroundingError."""
    for i, c in enumerate(spec.constraints):
        if c.id == cid:
            return i
    raise GroundingError(f"no RelationConstraint with id {cid!r} in spec")


def _replace_constraint(
    spec: RelSpec, index: int, new_constraint: RelationConstraint
) -> tuple[RelationConstraint, ...]:
    """Return a NEW constraints tuple with position ``index`` swapped out (input untouched)."""
    cs = list(spec.constraints)
    cs[index] = new_constraint
    return tuple(cs)


# ---------------------------------------------------------------- public operations

def ground(spec: RelSpec, target_id: str, decision: GroundingDecision) -> RelSpec:
    """Fix a delegated constraint NOW: DELEGATED (P) -> GROUNDED (G).

    Appends ``decision`` to the decision ledger and records provenance on the
    constraint. Fail-closed: raises GroundingError if the target is INVARIANT, if
    it is not currently DELEGATED, or if ``decision.target`` disagrees with
    ``target_id``.
    """
    if decision.target and decision.target != target_id:
        raise GroundingError(
            f"decision.target {decision.target!r} != target_id {target_id!r}"
        )
    if any(d.id == decision.id for d in spec.groundings):
        raise GroundingError(f"decision id {decision.id!r} already present in ledger")

    idx = _find_constraint_index(spec, target_id)
    c = spec.constraints[idx]

    if c.status is Status.INVARIANT:
        raise GroundingError(
            f"constraint {target_id!r} is INVARIANT and cannot be grounded (fail-closed)"
        )
    if c.status is not Status.DELEGATED:
        raise GroundingError(
            f"constraint {target_id!r} is {c.status.value}, expected delegated to ground"
        )

    prov = Provenance(
        origin="ground",
        actor=decision.actor,
        activity="ground",
        inputs=(decision.id,),
        note=decision.rationale,
    )
    grounded_c = replace(c, status=Status.GROUNDED, prov=prov)

    new_constraints = _replace_constraint(spec, idx, grounded_c)
    new_groundings = spec.groundings + (decision,)
    return replace(spec, constraints=new_constraints, groundings=new_groundings)


def unground(spec: RelSpec, decision_id: str) -> RelSpec:
    """Roll back a grounding: GROUNDED (G) -> DELEGATED (P), removing the decision.

    Fail-closed: raises GroundingError if the decision is unknown, if its target
    constraint is missing, if the target has since been promoted to INVARIANT
    (protected), or if it is not currently GROUNDED (nothing to undo).
    """
    decision = next((d for d in spec.groundings if d.id == decision_id), None)
    if decision is None:
        raise GroundingError(f"no GroundingDecision with id {decision_id!r} in ledger")

    idx = _find_constraint_index(spec, decision.target)
    c = spec.constraints[idx]

    if c.status is Status.INVARIANT:
        raise GroundingError(
            f"constraint {c.id!r} is INVARIANT and cannot be ungrounded (fail-closed)"
        )
    if c.status is not Status.GROUNDED:
        raise GroundingError(
            f"constraint {c.id!r} is {c.status.value}, expected grounded to unground"
        )

    prov = Provenance(
        origin="unground",
        actor=decision.actor,
        activity="unground",
        inputs=(decision_id,),
        note=f"reverted grounding {decision_id}",
    )
    delegated_c = replace(c, status=Status.DELEGATED, prov=prov)

    new_constraints = _replace_constraint(spec, idx, delegated_c)
    new_groundings = tuple(d for d in spec.groundings if d.id != decision_id)
    return replace(spec, constraints=new_constraints, groundings=new_groundings)


def set_invariant(spec: RelSpec, constraint_id: str) -> RelSpec:
    """Promote a constraint into the un-touchable core: status -> INVARIANT (I).

    Also registers the id in ``spec.invariant.relation_invariants`` so downstream
    fail-closed guards see a single source of truth. Idempotent.
    """
    idx = _find_constraint_index(spec, constraint_id)
    c = spec.constraints[idx]

    if c.status is Status.INVARIANT and constraint_id in spec.invariant.relation_invariants:
        return spec  # already invariant and registered: no-op (still returns a valid spec)

    prov = Provenance(
        origin="set_invariant",
        actor="designer",
        activity="set_invariant",
        inputs=(constraint_id,),
        note="promoted to invariant core",
    )
    inv_c = replace(c, status=Status.INVARIANT, prov=prov)

    new_constraints = _replace_constraint(spec, idx, inv_c)
    new_invariant = replace(
        spec.invariant,
        relation_invariants=spec.invariant.relation_invariants | {constraint_id},
    )
    return replace(spec, constraints=new_constraints, invariant=new_invariant)


def partition(
    spec: RelSpec,
) -> tuple[
    tuple[RelationConstraint, ...],
    tuple[RelationConstraint, ...],
    tuple[RelationConstraint, ...],
]:
    """Split constraints by edit-authority into (I, G, P), preserving spec order.

    Returns three tuples: INVARIANT, GROUNDED, DELEGATED. Together they partition
    ``spec.constraints`` (every constraint appears in exactly one bucket)."""
    inv: list[RelationConstraint] = []
    grn: list[RelationConstraint] = []
    dele: list[RelationConstraint] = []
    for c in spec.constraints:
        if c.status is Status.INVARIANT:
            inv.append(c)
        elif c.status is Status.GROUNDED:
            grn.append(c)
        elif c.status is Status.DELEGATED:
            dele.append(c)
        else:  # fail-closed: never silently bucket an unrecognized status as delegated
            raise GroundingError(
                f"constraint {c.id!r} has unrecognized status {c.status!r}"
            )
    return tuple(inv), tuple(grn), tuple(dele)
