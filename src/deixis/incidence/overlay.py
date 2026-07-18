"""M5: the overlay poset — a signature-based Boolean overlay IR.

Where :mod:`deixis.incidence.witness` records *how boundaries meet*, this module
records *how interiors overlap*. Given a family of regions ``R1..Rn`` and a set of
sample occupancies, every point ``x`` carries a **membership signature**

    sigma(x) = { i | x in Ri }

Each distinct *non-empty* signature that is actually realised becomes one
:class:`~deixis.core.types.OverlayAtom` — an independently-addressable overlap
region. The atom's ``members`` frozenset *is* its signature. Crucially an atom may
cut across several physical rooms: a program region such as
``circulation & daylight & exhibition`` is a single overlay atom you can name and
constrain, even though no single wall bounds it.

Honest structural commitment (do not overstate this IR):

* Overlay atoms form a **poset** ordered by signature inclusion, *plus* a
  non-empty predicate — they are **NOT a meet-semilattice**. The set of
  *realised* signatures is not closed under intersection: two realised overlaps
  ``{1,2}`` and ``{1,3}`` can both be non-empty while their would-be meet ``{1}``
  is empty (nobody stands in R1 alone). There is therefore no guaranteed
  greatest lower bound inside the realised set. :func:`meet` returns ``None``
  exactly there, and :func:`is_meet_semilattice` reports the failure globally.
* Signature inclusion is *contravariant* with spatial occupancy: a **larger**
  signature (member of more regions) occupies a **smaller**, deeper overlap. The
  order here is the signature/membership order; occupancy shrinks as you go up.
  :func:`occupancy_compare` names the signature relation (``'sub'`` == smaller
  signature) and this docstring is the single source of truth for that choice, so
  :func:`overlay_order` and :func:`occupancy_compare` stay consistent.

Everything is exact set / structural work — no floats, no geometry. Only this
file's own helpers are defined; the M0 contract in :mod:`deixis.core.types` is
used, never edited. Ids are deterministic (derived from the sorted signature) so
equal inputs yield equal atoms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Union

from deixis.core.ids import Provenance
from deixis.core.types import OverlayAtom

# A signature is a frozenset of region ids; ``memberships`` may be given as an
# iterable of such signatures, or as a mapping ``point_id -> signature``.
Signature = frozenset
_Memberships = Union[Mapping[str, Iterable[str]], Iterable[Iterable[str]]]


# ---------------------------------------------------------------- id helper
def _atom_id(members: frozenset[str]) -> str:
    """Deterministic, *injective* id for an atom from its signature.

    Each member is length-prefixed (``<len>~<id>``) before joining, so the id is
    unambiguous even when a region id itself contains the ``~`` or ``;``
    separators: ``{"a;b"}`` and ``{"a", "b"}`` map to different ids. Stable and
    randomness-free (sorted members)."""
    return "o:" + ";".join(f"{len(m)}~{m}" for m in sorted(members))


# ---------------------------------------------------------------- build
def build_overlay(
    region_ids: Iterable[str],
    memberships: _Memberships,
    *,
    prov: Optional[Provenance] = None,
) -> tuple[OverlayAtom, ...]:
    """Build the overlay atoms for one distinct *non-empty* signature each.

    ``region_ids`` is the universe of region ids (the ``Ri``). ``memberships``
    supplies the occupancy of sample points: either an iterable of signatures
    (each an iterable of region ids ``sigma(x)``) or a mapping
    ``point_id -> sigma(x)``. Every distinct non-empty signature that appears
    yields exactly one :class:`OverlayAtom` whose ``members`` frozenset *is* that
    signature and whose ``nonempty`` flag is ``True`` (it was witnessed).

    * The empty signature (a point in no region) is skipped — an overlay atom is
      by definition an overlap.
    * A membership that references a region id not in ``region_ids`` is a caller
      error and raises ``ValueError`` (we never silently drop occupancy).

    The returned tuple is ordered deterministically by ``(len(signature),
    sorted signature)`` so equal inputs give an equal tuple. Ids are derived from
    the sorted signature, so the same overlap addressed from different sample
    sets collapses to the *same* atom id.
    """
    universe = frozenset(region_ids)

    if isinstance(memberships, Mapping):
        raw_signatures: Iterable[Iterable[str]] = memberships.values()
    else:
        raw_signatures = memberships

    seen: set[frozenset[str]] = set()
    for sig in raw_signatures:
        members = frozenset(sig)
        unknown = members - universe
        if unknown:
            raise ValueError(
                f"membership references region id(s) {sorted(unknown)!r} "
                f"not in region_ids {sorted(universe)!r}"
            )
        if not members:
            continue  # empty signature is not an overlap
        seen.add(members)

    ordered = sorted(seen, key=lambda m: (len(m), sorted(m)))
    return tuple(
        OverlayAtom(id=_atom_id(m), members=m, nonempty=True, prov=prov)
        for m in ordered
    )


# ---------------------------------------------------------------- poset order
def overlay_order(atoms: Iterable[OverlayAtom]) -> set[tuple[str, str]]:
    """The strict partial order on atoms by **signature inclusion**.

    Returns the set of ordered pairs ``(sub_id, sup_id)`` such that
    ``sub.members`` is a *proper* subset of ``sup.members`` — i.e. the ``sub``
    atom's signature is contained in the ``sup`` atom's signature. This is the
    full (transitive) strict order, not just the covering relation.

    Consistent with :func:`occupancy_compare`: ``(A.id, B.id)`` is in the result
    iff ``occupancy_compare(A, B) == 'sub'``. Remember (see module docstring) the
    larger signature is the *smaller* spatial overlap: the order is on membership,
    not area.
    """
    items = list(atoms)
    out: set[tuple[str, str]] = set()
    for a in items:
        for b in items:
            if a.id == b.id:
                continue
            if a.members < b.members:  # proper subset
                out.add((a.id, b.id))
    return out


def occupancy_compare(atomA: OverlayAtom, atomB: OverlayAtom) -> str:
    """Compare two atoms in the signature-inclusion poset.

    Returns:

    * ``'sub'``          — ``atomA.members`` is a proper subset of
      ``atomB.members`` (A has the smaller signature / larger spatial overlap).
    * ``'sup'``          — ``atomA.members`` is a proper superset of
      ``atomB.members``.
    * ``'incomparable'`` — neither contains the other (this includes equal
      signatures, which for distinct atoms should not occur, and any genuinely
      crossing pair such as ``{1,2}`` vs ``{1,3}``).
    """
    a, b = atomA.members, atomB.members
    if a < b:
        return "sub"
    if a > b:
        return "sup"
    return "incomparable"


# ---------------------------------------------------------------- (non-)semilattice
def realized_signatures(atoms: Iterable[OverlayAtom]) -> frozenset[frozenset[str]]:
    """The set of signatures actually realised as non-empty atoms."""
    return frozenset(a.members for a in atoms if a.nonempty)


def meet(
    atoms: Iterable[OverlayAtom],
    atomA: OverlayAtom,
    atomB: OverlayAtom,
) -> Optional[OverlayAtom]:
    """Greatest lower bound of two atoms *within the realised set*, or ``None``.

    The meet is the unique realised atom ``m`` such that ``m.members`` is
    contained in both ``atomA.members`` and ``atomB.members`` and contains every
    other such lower bound. It need **not** exist: the natural candidate is the
    intersection ``atomA.members & atomB.members``, but that signature may be
    empty *or simply not realised*, and the remaining lower bounds may have no
    greatest element. In that case ``None`` is returned — this is precisely why
    overlay atoms are a poset and not a meet-semilattice.
    """
    items = list(atoms)
    inter = atomA.members & atomB.members
    lowers = [m for m in items if m.nonempty and m.members <= inter]
    if not lowers:
        return None
    for cand in lowers:
        if all(other.members <= cand.members for other in lowers):
            return cand
    return None


def is_meet_semilattice(atoms: Iterable[OverlayAtom]) -> bool:
    """True iff *every* pair of atoms has a meet inside the realised set.

    Honest self-check: for the overlay IR this is generally ``False``. A single
    pair with no greatest lower bound (see :func:`meet`) is enough to make the
    realised signatures not closed under meet. Only *realised* (``nonempty``)
    atoms are elements of the poset — atoms explicitly marked empty are not
    members and never make the answer ``False``.
    """
    items = [a for a in atoms if a.nonempty]
    for i, a in enumerate(items):
        for b in items[i:]:
            if meet(items, a, b) is None:
                return False
    return True


# ---------------------------------------------------------------- forbidden/empty
@dataclass(frozen=True)
class OverlayViolation:
    """A mismatch between an atom's non-empty predicate and an emptiness policy."""
    signature: frozenset[str]
    code: str          # "forbidden_nonempty" | "required_nonempty_but_empty"
    detail: str
    atom_id: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.code}] {sorted(self.signature)}: {self.detail}"


def diagnose_forbidden_empty(
    atoms: Iterable[OverlayAtom],
    forbidden: set[frozenset],
) -> list[OverlayViolation]:
    """Diagnose emptiness-policy violations in *both* directions.

    ``forbidden`` is the set of signatures that MUST be empty (an overlap the
    design forbids — e.g. ``exhibition & storage`` must never coincide). For each
    atom two checks run:

    * **forbidden_nonempty** ("should be empty but is non-empty") — the atom's
      signature is in ``forbidden`` yet the atom is marked ``nonempty=True``: the
      forbidden overlap is realised.
    * **required_nonempty_but_empty** ("should be non-empty but is empty") — the
      atom is marked ``nonempty=False`` (recorded as an absent overlap) while its
      signature is *not* forbidden: an overlap we tracked collapsed even though
      nothing declared it forbidden.

    Returns a list of :class:`OverlayViolation` (empty == policy satisfied). The
    ``forbidden`` entries are compared as frozensets, so pass raw sets/frozensets
    of region ids.
    """
    forbidden_fs = {frozenset(s) for s in forbidden}
    out: list[OverlayViolation] = []
    for a in atoms:
        is_forbidden = a.members in forbidden_fs
        if a.nonempty and is_forbidden:
            out.append(OverlayViolation(
                a.members, "forbidden_nonempty",
                "overlap is forbidden (must be empty) but is realised", a.id))
        elif (not a.nonempty) and (not is_forbidden):
            out.append(OverlayViolation(
                a.members, "required_nonempty_but_empty",
                "overlap is not forbidden yet is marked empty", a.id))
    return out
