"""M4: shared-boundary identity — one cell, co-referenced by two regions.

Two rooms that share a wall do not each own a private copy of the wall: they
co-reference **one** cell in the shared witnessed complex K. This module makes
that co-reference explicit and *stable*.

Concretely, a single :class:`~deixis.core.types.Cell` (e.g. ``F17``, dim 2 — a
wall face) is referenced by the :class:`~deixis.core.types.IncidenceWitness` of
region ``A`` *and* by the witness of region ``B``, using the **same**
``cell_id``. Fixing the boundary position with a
:class:`~deixis.core.types.GroundingDecision` (grounding the wall constraint)
moves the geometry that will later realise the wall, but must never fork,
rename, or drop that shared cell id: the identity anchor ``F17`` is preserved so
both rooms' boundaries stay linked. This is the demo2 core.

API
---
* :func:`make_shared_boundary` — build the two witnesses (one per region) that
  co-reference a single ``cell_id``.
* :func:`boundary_regions`     — the set of region ids that share a given cell.
* :func:`reground_boundary`    — fix the boundary with a GroundingDecision via
  :mod:`deixis.grounding.grounding_op` (immutable), leaving every witness — and
  hence every ``cell_id`` — untouched.
* :func:`id_stability`         — True iff a cell's shared identity is preserved
  across two specs (same witnesses reference it, same regions share it).
* :func:`demo_shared_wall`     — wall ``F17`` between two rooms; ground the wall
  position and show the shared cell id is invariant.

Everything is structural / exact: no floats, no geometry. Only this file's own
helpers are defined; the M0 contract in ``deixis.core.types`` and the grounding
operations in ``deixis.grounding.grounding_op`` are used, never edited.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Union

from deixis.core.types import (
    Cell,
    ContactDim,
    GroundingDecision,
    IncidenceWitness,
    RelationConstraint,
    RelSpec,
    Region,
    Status,
)
from deixis.core.rcc8 import mask, bit
from deixis.grounding.grounding_op import ground, unground

__all__ = [
    "BoundaryError",
    "make_shared_boundary",
    "boundary_regions",
    "reground_boundary",
    "id_stability",
    "demo_shared_wall",
]

# Valid contact dimensions a boundary witness may record (0=POINT/1=LINE/2=FACE).
_VALID_CONTACT_DIMS: frozenset[int] = frozenset(int(d) for d in ContactDim)
_EC: int = bit("EC")


class BoundaryError(Exception):
    """Raised when a shared-boundary operation is illegal (unshared cell, dup id...)."""


# ---------------------------------------------------------------- construction
def make_shared_boundary(
    region_a: str,
    region_b: str,
    cell_id: str,
    dim: Union[int, ContactDim],
    *,
    boundary_role: str = "shared_boundary",
) -> tuple[IncidenceWitness, IncidenceWitness]:
    """Build the two witnesses that co-reference a single boundary ``cell_id``.

    Returns ``(witness_for_A, witness_for_B)``. Both witnesses:

    * carry the **same** ``cell_id`` in their ``cell_ids`` — this shared id is the
      identity anchor of the boundary (the whole point of M4);
    * are incident to *both* ``region_a`` and ``region_b`` (a shared boundary
      joins them), so :func:`boundary_regions` recovers the full pair from either;
    * record the same ``intended_contact_dimension`` (``dim``).

    The two witnesses differ only in their stable ``id`` (the A-view vs the
    B-view of the same cell) and in the order their incident regions are listed
    (region-first), so each room can address "its" side of the wall while both
    still point at the one cell. ``dim`` accepts a :class:`ContactDim` or a raw
    int (0=POINT, 1=LINE, 2=FACE); ``F17`` in the demo is a 2-cell (wall face).
    """
    if region_a == region_b:
        raise BoundaryError(
            f"a shared boundary needs two distinct regions, got {region_a!r} twice"
        )
    if not cell_id:
        raise BoundaryError("cell_id must be a non-empty stable id")
    # Contact dimension is a discrete ContactDim/int (0/1/2), never a coordinate
    # and never a truthy proxy — reject float/bool/str outright rather than
    # silently coercing (honours the no-float rule and the ContactDim contract).
    if isinstance(dim, bool) or isinstance(dim, float):
        raise TypeError("dim must be a ContactDim or int (0/1/2), not "
                        f"{type(dim).__name__}")
    if not isinstance(dim, (int, ContactDim)):
        raise TypeError("dim must be a ContactDim or int (0/1/2), not "
                        f"{type(dim).__name__}")
    d = int(dim)
    if d not in _VALID_CONTACT_DIMS:
        raise BoundaryError(
            f"dim={d} is not a valid ContactDim {sorted(_VALID_CONTACT_DIMS)}"
        )

    w_a = IncidenceWitness(
        id=f"w:{cell_id}:{region_a}",
        cell_ids=(cell_id,),
        incident_regions=(region_a, region_b),
        intended_contact_dimension=d,
        boundary_role=boundary_role,
    )
    w_b = IncidenceWitness(
        id=f"w:{cell_id}:{region_b}",
        cell_ids=(cell_id,),
        incident_regions=(region_b, region_a),
        intended_contact_dimension=d,
        boundary_role=boundary_role,
    )
    return w_a, w_b


# ---------------------------------------------------------------- queries
def boundary_regions(spec: RelSpec, cell_id: str) -> frozenset[str]:
    """The set of region ids that share the cell ``cell_id`` as a boundary.

    Scans every witness in ``spec``; a witness "uses" the cell iff ``cell_id`` is
    among its ``cell_ids``, and then all of that witness's ``incident_regions``
    are counted as sharing the boundary. Returns a (possibly empty) frozenset —
    empty means the cell is not referenced by any witness in the spec.
    """
    out: set[str] = set()
    for w in spec.witnesses:
        if cell_id in w.cell_ids:
            out.update(w.incident_regions)
    return frozenset(out)


# edit-authority preference: a movable (DELEGATED) wall is grounded directly; a
# GROUNDED wall is re-grounded (unground+ground); an INVARIANT wall cannot move.
_STATUS_PRIORITY = {Status.DELEGATED: 0, Status.GROUNDED: 1, Status.INVARIANT: 2}


def _boundary_constraint(
    spec: RelSpec, shared: frozenset[str]
) -> Optional[RelationConstraint]:
    """The wall constraint to fix for a boundary shared by ``shared`` (or None).

    A boundary/wall constraint is one whose two (distinct) endpoints both lie in
    ``shared`` *and* whose RCC-8 mask admits ``EC`` (a touching boundary) — a
    ``DC``/``PO``/... constraint between the rooms is not a shared wall and is
    never grounded here. Among the candidates the most edit-actionable is
    returned (DELEGATED before GROUNDED before INVARIANT) so a movable wall is
    always preferred over an already-fixed or protected one.
    """
    cands = [
        c
        for c in spec.constraints
        if c.src != c.dst
        and c.src in shared
        and c.dst in shared
        and (c.rcc8_mask & _EC)
    ]
    if not cands:
        return None
    return min(cands, key=lambda c: _STATUS_PRIORITY.get(c.status, 3))


# ---------------------------------------------------------------- regrounding
def reground_boundary(
    spec: RelSpec, cell_id: str, decision: GroundingDecision
) -> RelSpec:
    """Fix the boundary position with ``decision`` while keeping ``cell_id`` stable.

    Semantics: the wall's position is pinned by grounding the *relation* between
    the two rooms — their EC-compatible boundary constraint (see
    :func:`_boundary_constraint`). The decision is routed to that constraint
    through :mod:`deixis.grounding.grounding_op` (pure/immutable), with its
    ``target`` re-pointed to the constraint id whenever it does not already match
    (empty or mismatched), so grounding_op's fail-closed target check is
    satisfied and a later ``unground`` can find the target.

    * a **DELEGATED** wall is grounded directly (P -> G);
    * an already **GROUNDED** wall is genuinely *re-grounded* — the existing
      decision that pinned it is ungrounded (G -> P) and the new decision then
      re-pins it (P -> G) at the moved position; the wall never lingers with a
      decision but no status change.

    In every path the witnesses are never touched, so each witness that
    referenced ``cell_id`` still references the same ``cell_id``: the boundary's
    identity anchor is invariant under regrounding (:func:`id_stability`).

    Fail-closed (raises :class:`BoundaryError`): ``cell_id`` not shared by >=2
    regions; the decision id already in the ledger; no EC-compatible constraint
    links the shared regions (nothing to fix); or the wall constraint is
    INVARIANT (a protected wall cannot be moved).
    """
    shared = boundary_regions(spec, cell_id)
    if len(shared) < 2:
        raise BoundaryError(
            f"cell {cell_id!r} is not a shared boundary (regions sharing it: "
            f"{sorted(shared)})"
        )
    if any(d.id == decision.id for d in spec.groundings):
        raise BoundaryError(
            f"decision id {decision.id!r} already present in ledger"
        )

    c = _boundary_constraint(spec, shared)
    if c is None:
        raise BoundaryError(
            f"no EC-compatible boundary constraint links {sorted(shared)}; "
            f"nothing to fix for cell {cell_id!r}"
        )
    if c.status is Status.INVARIANT:
        raise BoundaryError(
            f"boundary constraint {c.id!r} is INVARIANT; the wall cannot be moved"
        )

    routed = decision if decision.target == c.id else replace(decision, target=c.id)

    base = spec
    if c.status is Status.GROUNDED:
        # re-ground: release the existing fix so grounding_op can re-pin it.
        old = next((d for d in spec.groundings if d.target == c.id), None)
        if old is None:
            raise BoundaryError(
                f"constraint {c.id!r} is grounded but no ledger decision targets "
                f"it (inconsistent spec)"
            )
        base = unground(spec, old.id)

    # ground() rebuilds the spec via dataclasses.replace; witnesses (and their
    # cell_ids) are carried through unchanged.
    return ground(base, c.id, routed)


# ---------------------------------------------------------------- id stability
def _witness_ids_using(spec: RelSpec, cell_id: str) -> frozenset[str]:
    """Ids of witnesses in ``spec`` whose ``cell_ids`` contain ``cell_id``."""
    return frozenset(w.id for w in spec.witnesses if cell_id in w.cell_ids)


def id_stability(spec_before: RelSpec, spec_after: RelSpec, cell_id: str) -> bool:
    """True iff the shared identity of ``cell_id`` is preserved across the two specs.

    Preserved means, jointly:

    * the same set of witness ids reference ``cell_id`` before and after
      (no witness forked, renamed, dropped, or newly attached to the cell), and
    * the same set of regions share the cell (:func:`boundary_regions` equal), and
    * the cell was actually referenced before (a boundary that never existed
      cannot be called "stable").

    This is the demo2 invariant: fixing the wall position must not disturb the
    ``F17`` identity anchor.
    """
    before = _witness_ids_using(spec_before, cell_id)
    if not before:
        return False
    after = _witness_ids_using(spec_after, cell_id)
    if before != after:
        return False
    return boundary_regions(spec_before, cell_id) == boundary_regions(
        spec_after, cell_id
    )


# ---------------------------------------------------------------- demo2
def demo_shared_wall() -> tuple[RelSpec, RelSpec, Cell]:
    """Wall ``F17`` between two rooms; ground its position; show id invariance.

    Builds two rooms that co-reference the single wall cell ``F17`` (dim 2), then
    fixes the wall position with a :class:`GroundingDecision` via
    :func:`reground_boundary`. Returns ``(before, after, cell)`` where:

    * ``before`` — the spec with both rooms' witnesses on ``F17`` and a DELEGATED
      EC wall constraint;
    * ``after`` — the spec after the wall position is grounded (the constraint is
      now GROUNDED and the decision is in the ledger), witnesses unchanged;
    * ``cell`` — the shared :class:`Cell` ``F17``.

    Post-conditions (also asserted in the demo's pytest): both rooms share
    ``F17`` before and after, the wall constraint is grounded after, and
    :func:`id_stability` holds for ``F17``.
    """
    room_a = Region(id="room_A", semantic_type="living")
    room_b = Region(id="room_B", semantic_type="bedroom")
    cell = Cell(id="F17", dim=2)

    w_a, w_b = make_shared_boundary("room_A", "room_B", "F17", dim=2)
    wall = RelationConstraint(
        src="room_A",
        dst="room_B",
        rcc8_mask=mask("EC"),
        status=Status.DELEGATED,
        id="wall_AB",
    )
    before = RelSpec(
        regions=(room_a, room_b),
        constraints=(wall,),
        witnesses=(w_a, w_b),
    )

    decision = GroundingDecision(
        id="fix_wall_F17",
        target="wall_AB",
        fixed_value_or_domain="x==Fraction(21,5)",  # exact wall x-position
        actor="designer",
        rationale="pin shared wall F17 between room_A and room_B",
    )
    after = reground_boundary(before, "F17", decision)
    return before, after, cell
