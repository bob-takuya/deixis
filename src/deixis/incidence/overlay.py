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


# =================================================================
# Guards / diagnostics for the honest non-semilattice structure.
#
# These four helpers are *additive* and do not change any existing
# behaviour. They make the honest structural commitments of the module
# (poset + non-empty predicate, NOT a meet-semilattice; exact atoms A_S
# distinct from the cumulative domain C_S = ⋂ Ri) machine-checkable, and
# guard against a regression that quietly re-asserts a semilattice.
# =================================================================


# ---------------------------------------------------------------- ∩-closure
@dataclass(frozen=True)
class ClosureDefect:
    """A witness that the realised signatures are not closed under ``∩``.

    ``left`` and ``right`` are two *realised* signatures whose (non-empty)
    intersection ``missing = left & right`` is **not** itself realised. This is a
    defect of the *family of signatures* — nothing is said here about whether the
    poset pair has a meet (see :class:`MeetDefect`; the two notions are NOT
    equivalent)."""
    left: frozenset[str]
    right: frozenset[str]
    missing: frozenset[str]     # left & right — non-empty but not realised

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"[intersection_not_realised] {sorted(self.left)} ∩ "
                f"{sorted(self.right)} = {sorted(self.missing)} is not realised")


def intersection_closure_defects(
    atoms: Iterable[OverlayAtom],
) -> list[ClosureDefect]:
    """Report where the realised family is not closed under *non-empty* ``∩``.

    Note the qualification: this diagnoses closure under intersections *within the
    universe of non-empty signatures* only. An empty intersection (``S ∩ T = ∅``,
    disjoint region sets) is never a defect — the empty signature is a point in no
    region, not an overlap atom — so ``{R1}`` and ``{R2}`` are reported clean even
    though the strict set-theoretic notion of an intersection-closed family would
    demand ``∅``.

    For every unordered pair of *realised* signatures ``S, T`` whose intersection
    ``S ∩ T`` is **non-empty**, a :class:`ClosureDefect` is emitted iff ``S ∩ T``
    is not itself a realised signature. The empty intersection is never a defect:
    the empty signature is a point in no region, not an overlap atom.

    This is a statement about the family of signatures only. It is a *necessary*
    condition for a missing meet but **not** a sufficient one: ``S ∩ T`` may be
    absent from the family while the pair still has a greatest lower bound via a
    different, *lower* realised signature — a realised proper subset of ``S ∩ T``
    (see :func:`meet_defects`). The two APIs are deliberately separate because
    ``∩``-closure failure and meet-absence are non-equivalent.

    Output is deterministic, ordered by ``(len, sorted)`` of ``left`` then
    ``right``.
    """
    real = realized_signatures(atoms)
    sigs = sorted(real, key=lambda m: (len(m), sorted(m)))
    out: list[ClosureDefect] = []
    for i, S in enumerate(sigs):
        for T in sigs[i + 1:]:
            inter = S & T
            if not inter:
                continue  # empty signature is not an overlap atom
            if inter not in real:
                out.append(ClosureDefect(S, T, inter))
    return out


# ---------------------------------------------------------------- meet defects
@dataclass(frozen=True)
class MeetDefect:
    """A witness that a poset pair has *no* greatest lower bound.

    ``left`` and ``right`` are realised atoms with no meet inside the realised
    poset. ``intersection = left & right`` is the signature-lattice candidate (it
    is either not realised, or realised but not itself a lower bound of every
    other lower bound). ``maximal_lower_bounds`` is the set of realised signatures
    that are lower bounds and are not dominated by any other lower bound — a meet
    exists iff this set has exactly one element, so a defect always shows either
    zero or ``>= 2`` maximal lower bounds."""
    left: frozenset[str]
    right: frozenset[str]
    intersection: frozenset[str]
    maximal_lower_bounds: frozenset[frozenset[str]]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lbs = sorted((sorted(m) for m in self.maximal_lower_bounds))
        return (f"[no_meet] {sorted(self.left)} ∧ {sorted(self.right)}: "
                f"maximal lower bounds = {lbs}")


def meet_defects(atoms: Iterable[OverlayAtom]) -> list[MeetDefect]:
    """Report every realised pair that has no greatest lower bound (meet).

    Iterates unordered pairs of *realised* (``nonempty``) atoms and, whenever
    :func:`meet` returns ``None``, emits a :class:`MeetDefect` carrying the
    signature-intersection candidate and the maximal lower bounds that witness the
    ambiguity (either none, or two-or-more incomparable ones).

    Distinct from :func:`intersection_closure_defects`: a ``∩``-closure defect can
    exist while the meet still exists (a *lower* realised signature — a realised
    proper subset of ``S ∩ T`` — is the unique greatest lower bound even though
    ``S ∩ T`` itself is unrealised). Conversely a
    missing meet always implies ``S ∩ T`` is unrealised, but the reverse fails —
    hence two separate APIs. Output is deterministic.
    """
    items = sorted(
        (a for a in atoms if a.nonempty),
        key=lambda a: (len(a.members), sorted(a.members)),
    )
    out: list[MeetDefect] = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a.members == b.members:
                continue  # same signature (ill-formed input) has itself as meet
            if meet(items, a, b) is not None:
                continue
            inter = a.members & b.members
            lowers = [m for m in items if m.members <= inter]
            maximal = [
                m for m in lowers
                if not any(other.members > m.members for other in lowers)
            ]
            out.append(MeetDefect(
                a.members, b.members, inter,
                frozenset(m.members for m in maximal),
            ))
    return out


# ---------------------------------------------------------------- Hasse / covering
def covering_relation(atoms: Iterable[OverlayAtom]) -> set[tuple[str, str]]:
    """The **covering relation** (Hasse diagram) of the signature-inclusion order.

    Returns the ordered pairs ``(sub_id, sup_id)`` with ``sub.members`` a proper
    subset of ``sup.members`` such that *no* atom ``c`` lies strictly between them
    (``sub.members < c.members < sup.members``). This is the transitive reduction
    of :func:`overlay_order`: ``covering_relation(atoms)`` is a subset of
    ``overlay_order(atoms)`` and its transitive closure equals it (on a finite
    poset without duplicate signatures).

    Consistent with :func:`overlay_order`, all supplied atoms are treated as poset
    elements (no ``nonempty`` filtering here); pre-filter if you want only realised
    atoms.
    """
    items = list(atoms)
    out: set[tuple[str, str]] = set()
    for a in items:
        for b in items:
            if not (a.members < b.members):  # proper subset only
                continue
            between = any(
                a.members < c.members < b.members for c in items
            )
            if not between:
                out.add((a.id, b.id))
    return out


# ---------------------------------------------------------------- axiom guard
@dataclass(frozen=True)
class AxiomViolation:
    """A structural well-formedness violation of the overlap-atom algebra."""
    code: str          # see check_overlay_axioms for the closed vocabulary
    detail: str
    atom_id: str = ""
    signature: frozenset[str] = frozenset()

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.code}] {self.atom_id} {sorted(self.signature)}: {self.detail}"


def check_overlay_axioms(atoms: Iterable[OverlayAtom]) -> list[AxiomViolation]:
    """Check structural well-formedness of an overlay-atom family; return violations.

    The checks are exactly the structural commitments of this module — and,
    deliberately, *no more*. In particular there is **no** "every pair has a meet"
    check: a non-meet-semilattice input (e.g. ``{R1,R2}`` and ``{R1,R3}`` with
    ``{R1}`` unrealised) is well-formed and yields an empty result. This is the
    regression guard against silently re-asserting a semilattice.

    Violation codes (closed vocabulary):

    * ``members_not_frozenset`` — ``members`` is not a ``frozenset`` (the
      signature *is* the atom's identity and must be an exact, hashable set).
    * ``empty_signature`` — the **non-empty predicate**: an overlay atom is an
      overlap, so its signature must contain ``>= 1`` region; the empty signature
      (a point in no region) is not an atom.
    * ``nonempty_not_bool`` — the ``nonempty`` predicate is not a genuine ``bool``.
    * ``duplicate_signature`` — **canonical uniqueness**: the signature uniquely
      identifies an exact atom ``A_S`` (the points whose signature is *exactly*
      ``S``), so a normalised family carries at most one record per signature; two
      records for the same ``S`` are a denormalised duplicate. (Separately, and
      not the ground of this check: the exact atom ``A_S`` is *not* the cumulative
      domain ``C_S = ⋂_{i∈S} R_i`` — the union of all ``A_T`` for ``T ⊇ S`` — a
      distinction the signature-as-exact-membership convention keeps clean.)
    * ``id_not_canonical`` — a *builder-canonicality* check (not an algebra axiom):
      the public :class:`OverlayAtom` type and existing serialisers accept
      arbitrary ids, but atoms produced by :func:`build_overlay` carry the
      deterministic length-prefixed encoding of their signature; a mismatch flags
      an atom that would not collapse to the builder's normal-form identity.

    Returns a list (empty == well-formed). Per-atom violations follow input order;
    ``duplicate_signature`` violations are appended last in ``(len, sorted)``
    signature order.
    """
    items = list(atoms)
    out: list[AxiomViolation] = []
    by_sig: dict[frozenset[str], list[OverlayAtom]] = {}

    for a in items:
        if not isinstance(a.members, frozenset):
            out.append(AxiomViolation(
                "members_not_frozenset",
                f"members is {type(a.members).__name__}, not frozenset",
                a.id))
            continue
        if not isinstance(a.nonempty, bool):
            out.append(AxiomViolation(
                "nonempty_not_bool",
                f"nonempty is {type(a.nonempty).__name__}, not bool",
                a.id, a.members))
        if len(a.members) == 0:
            out.append(AxiomViolation(
                "empty_signature",
                "an overlay atom must occupy >= 1 region; the empty signature "
                "is a point in no region, not an overlap atom",
                a.id, a.members))
            continue
        if a.id != _atom_id(a.members):
            out.append(AxiomViolation(
                "id_not_canonical",
                f"id {a.id!r} is not the canonical encoding "
                f"{_atom_id(a.members)!r} of its exact signature",
                a.id, a.members))
        by_sig.setdefault(a.members, []).append(a)

    for sig in sorted(by_sig, key=lambda m: (len(m), sorted(m))):
        group = by_sig[sig]
        if len(group) > 1:
            out.append(AxiomViolation(
                "duplicate_signature",
                f"{len(group)} atoms share signature {sorted(sig)}; a signature "
                "uniquely identifies one exact atom A_S, so a normalised family "
                "holds at most one record per signature (A_S is not the "
                "cumulative domain C_S = ⋂ Ri)",
                group[0].id, sig))
    return out
