"""M-field (a) — scalar / membership *fields* and **thresholding-as-grounding**.

The core move of this module is a single thesis restated in code:

    A continuous space is a *field*, not a set of rooms. A room is what you get
    when you **threshold** the field. The threshold is therefore not neutral
    bookkeeping — it is a :class:`~deixis.core.types.GroundingDecision`. Choosing
    ``tau`` (or choosing argmax) is choosing *who gets to decide* where one room
    ends and the next begins.

So the field is held **ungrounded** — first-class, continuous, un-partitioned —
for as long as we can, exactly the way :mod:`deixis.grounding.grounding_op` holds
a relation ``DELEGATED`` until a decision fixes it. Projecting the field to rooms
is deferred, typed, and reversible:

    field (continuous)  --threshold_ground(tau)-->   room (level set L_tau)      [+ GroundingDecision]
    field (continuous)  --argmax_ground()-------->   partition (winner per cell) [a special reading]
    room  (cells)       --region_to_indicator()-->   field (0/1 indicator)       [the inverse]

The same field under two different ``tau`` *can* yield two different rooms (it
does whenever a sampled value lies between the two thresholds; a constant field
would not). That divergence is not a bug to be resolved by picking the "true"
threshold — there is no true threshold. It is the concrete demonstration that
**the threshold is where the decision rights live** (demo5, see
:func:`demo5_same_field_different_rooms`).

Honest limits (do not overclaim):
- What is stored is a **finitely sampled** scalar field on a **finite-extent**
  grid: only per-cell values, with no interpolation or in-cell model. A level
  set here is the set of *cells* whose sampled value clears ``tau`` — a
  discretization; it equals the true continuous level set only under a stated
  sampling/interpolation model this module does not itself fix.
- Physical meaning of a cell is only defined *relative to* ``(origin, spacing)``.
  Two fields with the same values but different spacing are different fields.
- Membership values are unitless and un-normalized: this module does not assume
  they sum to 1 across regions (multi-label overlap is allowed and expected).
- ``region_to_indicator`` is a *right inverse (section)* of thresholding, not a
  true inverse: ``threshold_ground(., tau=1)`` recovers the room, but the
  original non-binary field is NOT recoverable from a room. "Reversible"
  likewise means the field is retained separately and can be re-thresholded —
  not that the field can be reconstructed from the returned room/decision.
- ``argmax_ground`` returns only the partition dict (its signature); the choice
  *to use argmax* and its tie rule are decisions external to that pure return,
  unlike ``threshold_ground`` which externalizes its ``tau`` choice as a
  :class:`GroundingDecision`.
- All values are exact ``fractions.Fraction`` (or ``int``); float (and ``bool``)
  are rejected on construction. Comparisons (``>= tau``) and argmax ties are
  therefore exact.

This module builds only on the fixed M0 contract (``deixis.core.types`` /
``deixis.core.ids``); it edits nothing and mutates nothing (frozen dataclasses).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterator, Optional, Tuple

from ..core.ids import Provenance
from ..core.types import GroundingDecision

__all__ = [
    "ScalarField",
    "MembershipField",
    "Cell",  # a grid-cell multi-index (tuple[int, ...])
    "threshold_ground",
    "argmax_ground",
    "region_to_indicator",
    "ungrounded",
    "demo5_same_field_different_rooms",
]

# A grid cell is a row-major multi-index into the grid (exact, hashable).
Cell = Tuple[int, ...]

# Values we accept are exact rationals. ``bool`` is an int subclass we reject so
# ``True``/``False`` cannot silently masquerade as 1/0 membership values.
_Rational = (Fraction, int)


def _prod(shape: Tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _check_rational(x, where: str) -> None:
    if isinstance(x, bool) or not isinstance(x, _Rational):
        raise TypeError(
            f"{where} must be exact (Fraction or int), got {type(x).__name__}: "
            f"{x!r} (float is rejected — the core is float-free)"
        )


# --------------------------------------------------------------------- ScalarField
@dataclass(frozen=True)
class ScalarField:
    """A rational-valued scalar field sampled on a regular grid (float-free).

    ``values`` is a flat, row-major tuple of length ``prod(grid_shape)``. Cell
    ``(i0, i1, ...)`` sits at physical coordinate
    ``origin[d] + i_d * spacing`` on each axis ``d`` — so the field's meaning is
    only defined *relative to* ``(origin, spacing)``.
    """

    grid_shape: Tuple[int, ...]
    values: Tuple[Fraction, ...]
    origin: Tuple[Fraction, ...]
    spacing: Fraction

    def __post_init__(self) -> None:
        # Normalize sequences to real tuples so a caller-held list can never
        # mutate the field's contents behind its back ("frozen" for real).
        object.__setattr__(self, "grid_shape", tuple(self.grid_shape))
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "origin", tuple(self.origin))
        if not self.grid_shape or any(
            isinstance(n, bool) or (not isinstance(n, int)) or n <= 0 for n in self.grid_shape
        ):
            raise ValueError(f"grid_shape must be a non-empty tuple of positive ints: {self.grid_shape!r}")
        expected = _prod(self.grid_shape)
        if len(self.values) != expected:
            raise ValueError(
                f"values length {len(self.values)} != prod(grid_shape) {expected} "
                f"for shape {self.grid_shape!r}"
            )
        for v in self.values:
            _check_rational(v, "field value")
        if len(self.origin) != len(self.grid_shape):
            raise ValueError(
                f"origin rank {len(self.origin)} != grid rank {len(self.grid_shape)}"
            )
        for o in self.origin:
            _check_rational(o, "origin coordinate")
        _check_rational(self.spacing, "spacing")
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing!r}")

    # -- geometry / indexing (no mutation) --
    @property
    def ndim(self) -> int:
        return len(self.grid_shape)

    def _flat_index(self, cell: Cell) -> int:
        if len(cell) != self.ndim:
            raise IndexError(f"cell rank {len(cell)} != grid rank {self.ndim}: {cell!r}")
        flat = 0
        for axis, (i, n) in enumerate(zip(cell, self.grid_shape)):
            if isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= n:
                raise IndexError(
                    f"cell index {i!r} out of range [0,{n}) on axis {axis}: {cell!r}"
                )
            flat = flat * n + i
        return flat

    def eval(self, cell_index: Cell) -> Fraction:
        """Return the exact value at grid cell ``cell_index`` (a multi-index)."""
        return self.values[self._flat_index(cell_index)]

    def cell_center(self, cell_index: Cell) -> Tuple[Fraction, ...]:
        """Physical coordinate of a cell = ``origin + index * spacing`` (exact).

        Honest note: this is meaningful *only relative to* ``(origin, spacing)``.
        """
        self._flat_index(cell_index)  # bounds-check
        return tuple(o + Fraction(i) * self.spacing for o, i in zip(self.origin, cell_index))

    def cells(self) -> Iterator[Cell]:
        """Iterate every grid cell as a multi-index, in row-major order."""
        shape = self.grid_shape
        total = _prod(shape)
        for flat in range(total):
            idx = []
            rem = flat
            for n in reversed(shape):
                idx.append(rem % n)
                rem //= n
            yield tuple(reversed(idx))


# ----------------------------------------------------------------- MembershipField
@dataclass(frozen=True)
class MembershipField:
    """Per-region membership degrees over one shared grid (multi-label).

    ``fields[k]`` is the membership field of ``region_ids[k]``. Values are NOT
    assumed normalized: regions may overlap (a cell can have high membership in
    several regions at once) — that overlap is exactly the *continuous* content a
    hard room partition throws away. All member fields must share grid geometry.
    """

    region_ids: Tuple[str, ...]
    fields: Tuple[ScalarField, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_ids", tuple(self.region_ids))
        object.__setattr__(self, "fields", tuple(self.fields))
        if len(self.region_ids) != len(self.fields):
            raise ValueError(
                f"region_ids ({len(self.region_ids)}) and fields ({len(self.fields)}) "
                "must have equal length"
            )
        if len(self.region_ids) == 0:
            raise ValueError("MembershipField needs at least one region")
        if len(set(self.region_ids)) != len(self.region_ids):
            raise ValueError(f"region_ids must be unique: {self.region_ids!r}")
        ref = self.fields[0]
        for f in self.fields[1:]:
            if f.grid_shape != ref.grid_shape or f.origin != ref.origin or f.spacing != ref.spacing:
                raise ValueError("all membership fields must share grid_shape / origin / spacing")

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        return self.fields[0].grid_shape

    def field_for(self, region_id: str) -> ScalarField:
        try:
            k = self.region_ids.index(region_id)
        except ValueError:
            raise KeyError(f"no region {region_id!r} in membership field") from None
        return self.fields[k]


# ------------------------------------------------------- thresholding == grounding
def threshold_ground(
    field: ScalarField,
    tau: Fraction,
    region_id: str,
    actor: str = "designer",
    rationale: str = "",
) -> Tuple[frozenset, GroundingDecision]:
    """Project a field to a room by thresholding: ``L_tau = {cell | f(cell) >= tau}``.

    This is the central claim of the module made operational: **thresholding a
    field into a region IS a GroundingDecision** (field -> room). The returned
    :class:`~deixis.core.types.GroundingDecision` records who chose ``tau`` and
    why, and is marked reversible (re-thresholding at another ``tau`` yields a
    different room — nothing about the field is destroyed).

    Returns ``(cells, decision)`` where ``cells`` is the ``frozenset`` of grid
    cells (multi-indices) in the level set, and ``decision`` is the typed,
    externalized ledger entry for this projection.

    Honest limit: ``cells`` is a *finite-resolution* level set — the set of
    sampled cells clearing ``tau``, a discretization of the continuous level set.
    """
    _check_rational(tau, "tau")
    cells = frozenset(c for c in field.cells() if field.eval(c) >= tau)
    # A compact, exact signature of the *source field geometry* so two decisions
    # over different fields/geometries do not collide in the ledger and the entry
    # is auditable (id is only field->room per (region, tau, geometry), not full
    # field values — those are retained separately by the caller).
    geom = f"shape={field.grid_shape};origin={field.origin};spacing={field.spacing}"
    decision = GroundingDecision(
        id=f"gd:threshold:{region_id}:tau={tau}:{geom}",
        target=region_id,
        fixed_value_or_domain=(
            f"L_tau(region={region_id}) = {{cell | f(cell) >= {tau}}} over {geom} "
            f"-> |L_tau|={len(cells)}"
        ),
        actor=actor,
        rationale=rationale or (
            f"threshold field at tau={tau} to project region {region_id!r} "
            "(field -> room); reversible in the sense that the field is retained "
            "and another tau can be re-thresholded to a different room"
        ),
        reversibility="high",
        prov=Provenance(
            origin="threshold-ground",
            actor=actor,
            activity="threshold_ground",
            inputs=(region_id,),
            note=f"tau={tau}; {geom}; |L_tau|={len(cells)} cells of {_prod(field.grid_shape)}",
        ),
    )
    return cells, decision


def argmax_ground(mfield: MembershipField) -> Dict[Cell, str]:
    """Fully ground a membership field by argmax: each cell -> its winning region.

    This is a *special, total reading* of the field: instead of one threshold per
    region it commits every cell to the single region of maximal membership,
    yielding a hard partition (rooms tile the whole grid), leaving no cell
    undecided and no overlap preserved.

    Per its signature this pure function returns only the partition dict; unlike
    :func:`threshold_ground` it does not externalize a :class:`GroundingDecision`.
    The two decisions it embodies — choosing argmax at all, and the tie rule — are
    therefore made explicit *here in the contract* rather than in the ledger: ties
    are broken **deterministically** by ``region_ids`` order (the earliest region
    wins), and comparisons are exact so ties are exact, not float noise.
    """
    order = mfield.region_ids
    result: Dict[Cell, str] = {}
    for cell in mfield.fields[0].cells():
        best_region = order[0]
        best_val = mfield.fields[0].eval(cell)
        for k in range(1, len(order)):
            v = mfield.fields[k].eval(cell)
            if v > best_val:  # strict '>' => first (earliest) region wins ties
                best_val = v
                best_region = order[k]
        result[cell] = best_region
    return result


def region_to_indicator(
    cells: frozenset,
    grid_shape: Tuple[int, ...],
    origin: Optional[Tuple[Fraction, ...]] = None,
    spacing: Fraction = Fraction(1),
) -> ScalarField:
    """Section (right inverse): a room (set of cells) -> its 0/1 indicator field.

    Produces the field ``1`` on cells in the room and ``0`` elsewhere. Composed
    with :func:`threshold_ground` at ``tau == 1`` it round-trips a room exactly:
    ``threshold_ground(region_to_indicator(C, shape), 1, r)[0] == C``. This is a
    *right inverse of thresholding on rooms*, not a full inverse of an arbitrary
    field: the original non-binary field is not recoverable from a room.

    ``origin``/``spacing`` default to a unit grid at the origin; pass the source
    field's geometry to preserve it across the round trip.
    """
    ndim = len(grid_shape)
    if origin is None:
        origin = tuple(Fraction(0) for _ in range(ndim))
    total = _prod(grid_shape)

    # validate cells against the shape (fail-closed on out-of-range / wrong rank)
    def _flat(cell: Cell) -> int:
        if len(cell) != ndim:
            raise IndexError(f"cell rank {len(cell)} != grid rank {ndim}: {cell!r}")
        flat = 0
        for axis, (i, n) in enumerate(zip(cell, grid_shape)):
            if isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= n:
                raise IndexError(f"cell {cell!r} out of range on axis {axis}")
            flat = flat * n + i
        return flat

    inside = {_flat(c) for c in cells}
    one, zero = Fraction(1), Fraction(0)
    values = tuple(one if flat in inside else zero for flat in range(total))
    return ScalarField(grid_shape=grid_shape, values=values, origin=origin, spacing=spacing)


def ungrounded(mfield: MembershipField) -> bool:
    """Is the membership field still **ungrounded** (continuous, un-partitioned)?

    A field is *ungrounded* while it still carries continuous / overlapping
    membership that no grounding decision has collapsed into rooms — this is the
    first-class state we deliberately preserve (the flow / continuous space that a
    room partition cannot represent). It becomes *grounded* only once a decision
    (:func:`threshold_ground` per region, or :func:`argmax_ground`) projects it to
    rooms.

    Concretely this returns ``True`` unless the field is already a crisp one-hot
    partition — i.e. every cell has value 1 in exactly one region and 0 in all
    others (in which case there is nothing left to decide; the projection is
    already made). Any cell with a fractional value, with overlap (>=1 in two
    regions), or with no clear winner (all zero) keeps the field ungrounded.
    """
    one, zero = Fraction(1), Fraction(0)
    for cell in mfield.fields[0].cells():
        hot = 0
        for f in mfield.fields:
            v = f.eval(cell)
            if v == one:
                hot += 1
            elif v != zero:
                return True  # fractional membership => still continuous
        if hot != 1:
            return True  # overlap (>1 hot) or undecided (0 hot) => ungrounded
    return False


# --------------------------------------------------------------------------- demo5
def demo5_same_field_different_rooms() -> dict:  # pragma: no cover - narrative driver
    """demo5 — one field, different thresholds, different rooms == decision rights.

    Builds a single ungrounded membership field and shows that:
      * it is ``ungrounded`` (continuous, overlapping) until projected;
      * two different ``tau`` values threshold the *same* field into two *different*
        rooms — each carrying its own :class:`GroundingDecision`;
      * argmax gives yet another, total partition;
      * ``region_to_indicator`` inverts a room exactly (round trip).

    Returns a small dict of the computed artifacts (also printed by ``main``).
    """
    F = Fraction
    # A 1x5 "corridor" membership for region 'warm', values rising then falling.
    warm = ScalarField(
        grid_shape=(5,),
        values=(F(1, 5), F(3, 5), F(1), F(3, 5), F(1, 5)),
        origin=(F(0),),
        spacing=F(1),
    )
    cool = ScalarField(
        grid_shape=(5,),
        values=(F(4, 5), F(2, 5), F(0), F(2, 5), F(4, 5)),
        origin=(F(0),),
        spacing=F(1),
    )
    mf = MembershipField(region_ids=("warm", "cool"), fields=(warm, cool))

    low_cells, low_dec = threshold_ground(warm, F(1, 2), "warm", rationale="generous")
    high_cells, high_dec = threshold_ground(warm, F(4, 5), "warm", rationale="strict")
    partition = argmax_ground(mf)

    return {
        "ungrounded": ungrounded(mf),
        "tau_low": F(1, 2),
        "room_low": sorted(low_cells),
        "decision_low": low_dec,
        "tau_high": F(4, 5),
        "room_high": sorted(high_cells),
        "decision_high": high_dec,
        "same_field_different_rooms": low_cells != high_cells,
        "argmax_partition": partition,
    }


def main() -> None:  # pragma: no cover - human-readable driver
    d = demo5_same_field_different_rooms()
    print("=" * 68)
    print("demo5 — the threshold is where the decision rights live")
    print("=" * 68)
    print(f"membership field ungrounded (continuous/overlapping)? {d['ungrounded']}")
    print(f"\nsame 'warm' field, two thresholds -> two different rooms:")
    print(f"  tau={d['tau_low']}  (generous): room cells = {d['room_low']}")
    print(f"  tau={d['tau_high']}  (strict):   room cells = {d['room_high']}")
    print(f"  different rooms from the SAME field? {d['same_field_different_rooms']}")
    print(f"\neach projection is a typed GroundingDecision:")
    print(f"  {d['decision_low'].id}")
    print(f"  {d['decision_high'].id}")
    print(f"\nargmax gives a total partition (rooms tile the grid):")
    print(f"  {d['argmax_partition']}")
    print("=" * 68)


if __name__ == "__main__":  # pragma: no cover
    main()
