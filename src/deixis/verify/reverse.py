"""Reverse-verification: re-extract RCC-8 relations and contact dimensions from
*realized geometry* (exact-rational AABBs) and check them against the spec.

Where :mod:`deixis.solver.geom_solver` goes **spec -> geometry** (lowering RCC-8
masks + contact-dimension witnesses to linear constraints and asking Z3 for boxes),
this module goes the other way, **geometry -> spec**, entirely in exact integer /
rational arithmetic (never float):

* :func:`extract_relations` reads every pair of boxes in a
  :class:`~deixis.core.types.Realization` and returns the *definite* RCC-8 base
  relation between them, as an 8-bit mask, keyed by the ordered region-id pair.
* :func:`extract_contact_dimension` reports the dimension of the shared boundary
  of two touching boxes (POINT 0 / LINE 1 / FACE 2) — the information RCC-8's
  single ``EC`` symbol collapses, recovered from the geometry.
* :func:`verify` re-derives the relations, checks each
  :class:`~deixis.core.types.RelationConstraint` against its realized relation,
  and — the whole point of the Witnessed IR — flags every
  :class:`~deixis.core.types.IncidenceWitness` whose *intended* contact dimension
  is not the one the geometry actually realizes. So an ``EC``-satisfying
  realization whose contact is an **edge** while the witness demanded a **face**
  is caught as a ``witness_violation`` even though RCC-8 alone reports success.

Exactness
---------
For two axis-aligned boxes the RCC-8 relation is fully determined, axis by axis,
by the relation between the two *closed* coordinate intervals (the Rectangle
Algebra view — mirrored in geom_solver's forward encoding). Every comparison here
is a comparison of :class:`fractions.Fraction` endpoints, so the extracted relation
is exact and total: each in-domain box pair maps to exactly one JEPD base relation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Optional

from ..core.types import (
    Box,
    ContactDim,
    IncidenceWitness,
    Realization,
    RelSpec,
)
from ..core import rcc8


# ------------------------------------------------------------------ per-axis relation
# Each box interval on axis d is the closed interval [lo[d], hi[d]] with lo < hi
# (the solver guarantees non-degenerate boxes). We classify each axis into exactly
# one of {interior-overlap, boundary-touch, separated} using only Fraction compares.

def _iover(a: Box, b: Box, d: int) -> bool:
    """Interiors overlap on axis d (positive-length overlap)."""
    return a.lo[d] < b.hi[d] and b.lo[d] < a.hi[d]


def _touch(a: Box, b: Box, d: int) -> bool:
    """Closed intervals meet at exactly one shared endpoint on axis d (no interior overlap)."""
    return a.hi[d] == b.lo[d] or b.hi[d] == a.lo[d]


def _sep(a: Box, b: Box, d: int) -> bool:
    """Strictly separated on axis d (a positive gap)."""
    return a.hi[d] < b.lo[d] or b.hi[d] < a.lo[d]


def _asub(a: Box, b: Box, d: int) -> bool:
    """a's interval contained in b's interval on axis d (closed)."""
    return a.lo[d] >= b.lo[d] and a.hi[d] <= b.hi[d]


def _strict_inside(a: Box, b: Box, d: int) -> bool:
    """a's interval strictly interior to b's on axis d (no shared boundary)."""
    return a.lo[d] > b.lo[d] and a.hi[d] < b.hi[d]


def _eq_axis(a: Box, b: Box, d: int) -> bool:
    return a.lo[d] == b.lo[d] and a.hi[d] == b.hi[d]


def _check_box(a: Box) -> None:
    """Enforce the precondition that makes classification total: a non-degenerate
    box with exact-rational bounds and lo < hi on every axis."""
    if len(a.lo) != len(a.hi):
        raise ValueError(f"box {a.region_id!r} lo/hi have mismatched arity")
    for d, (lo, hi) in enumerate(zip(a.lo, a.hi)):
        if not isinstance(lo, Fraction) or not isinstance(hi, Fraction):
            raise TypeError(
                f"box {a.region_id!r} axis {d} bounds must be exact Fraction, "
                f"got {type(lo).__name__}/{type(hi).__name__}"
            )
        if not lo < hi:
            raise ValueError(
                f"box {a.region_id!r} is degenerate/inverted on axis {d}: "
                f"lo={lo} hi={hi} (need lo < hi)"
            )


def _check_ndim(a: Box, b: Box) -> int:
    _check_box(a)
    _check_box(b)
    if len(a.lo) != len(b.lo):
        raise ValueError(
            f"boxes {a.region_id!r} and {b.region_id!r} have different dimensionality "
            f"({len(a.lo)} vs {len(b.lo)})"
        )
    return len(a.lo)


# ------------------------------------------------------------------ box -> RCC-8 base

def relation_between_boxes(a: Box, b: Box) -> int:
    """Exact RCC-8 base relation of ``a`` w.r.t. ``b`` as a single-bit mask.

    Returns exactly one of the 8 JEPD base relations (DC/EC/PO/TPP/NTPP/TPPi/
    NTPPi/EQ). The classification is the reverse of geom_solver's forward encoding:

    * any axis separated                       -> **DC**
    * else if some axis only touches           -> **EC** (interiors disjoint,
                                                  closures meet on a lower-dim set)
    * else (interiors overlap on every axis):
        - a == b on every axis                 -> **EQ**
        - a's closed box ⊆ b's, proper:
            strictly interior on every axis     -> **NTPP**
            otherwise (some boundary coincides) -> **TPP**
        - b's closed box ⊆ a's, proper          -> **NTPPi / TPPi**
        - otherwise                            -> **PO**
    """
    ndim = _check_ndim(a, b)
    idx = range(ndim)

    # 1. separated on any axis => closures disjoint.
    if any(_sep(a, b, d) for d in idx):
        return rcc8.bit("DC")

    # 2. no separation; if not every axis has interior overlap then some axis only
    #    touches => closures meet but interiors are disjoint => EC.
    if not all(_iover(a, b, d) for d in idx):
        return rcc8.bit("EC")

    # 3. interiors overlap on every axis: resolve by containment.
    a_in_b = all(_asub(a, b, d) for d in idx)
    b_in_a = all(_asub(b, a, d) for d in idx)

    if a_in_b and b_in_a:
        # mutual containment of closed intervals on every axis <=> equal boxes.
        return rcc8.bit("EQ")
    if a_in_b:  # proper (not b_in_a): a is a proper part of b
        if all(_strict_inside(a, b, d) for d in idx):
            return rcc8.bit("NTPP")
        return rcc8.bit("TPP")
    if b_in_a:  # proper: b is a proper part of a
        if all(_strict_inside(b, a, d) for d in idx):
            return rcc8.bit("NTPPi")
        return rcc8.bit("TPPi")

    return rcc8.bit("PO")


# ------------------------------------------------------------------ realization helpers

def _boxmap(realization: Realization) -> dict[str, Box]:
    return {b.region_id: b for b in realization.boxes}


# ------------------------------------------------------------------ public: relations

def extract_relations(realization: Realization) -> dict[tuple[str, str], int]:
    """Re-extract the definite RCC-8 relation of every ordered region pair.

    Returns ``{(rid_i, rid_j): mask}`` for every pair ``i < j`` in the box order
    of ``realization``. ``mask`` is the single-base relation of box ``i`` w.r.t.
    box ``j`` (its converse gives ``j`` w.r.t. ``i``). Region ids must be unique
    within the realization.
    """
    boxes = realization.boxes
    seen: set[str] = set()
    for b in boxes:
        if b.region_id in seen:
            raise ValueError(f"duplicate region id in realization: {b.region_id!r}")
        seen.add(b.region_id)

    out: dict[tuple[str, str], int] = {}
    for a, b in combinations(boxes, 2):
        out[(a.region_id, b.region_id)] = relation_between_boxes(a, b)
    return out


# ------------------------------------------------------------------ public: contact dim

def extract_contact_dimension(realization: Realization, A: str, B: str) -> ContactDim:
    """Dimension of the shared boundary of the touching boxes ``A`` and ``B``.

    The contact set is the intersection of the two closed boxes; its dimension is
    the number of axes on which the intervals overlap with *positive* length (on
    the remaining axes the intervals meet at a single coincident endpoint). For an
    ``EC`` pair this is what the RCC-8 ``EC`` symbol collapses: 0 => POINT (corner),
    1 => LINE (edge), 2 => FACE.

    Raises :class:`ValueError` if either box is missing, if the boxes are disjoint
    (no contact), or if their interiors overlap on every axis (a full-dimensional
    overlap, i.e. not a boundary contact).
    """
    bm = _boxmap(realization)
    if A not in bm:
        raise ValueError(f"region {A!r} not in realization")
    if B not in bm:
        raise ValueError(f"region {B!r} not in realization")
    a, b = bm[A], bm[B]
    ndim = _check_ndim(a, b)

    positive = 0
    for d in range(ndim):
        lo = a.lo[d] if a.lo[d] > b.lo[d] else b.lo[d]   # max of the two lows
        hi = a.hi[d] if a.hi[d] < b.hi[d] else b.hi[d]   # min of the two highs
        if lo > hi:
            raise ValueError(
                f"regions {A!r} and {B!r} are disjoint on axis {d}: no shared boundary"
            )
        if lo < hi:
            positive += 1
        # lo == hi: the intervals meet at a single point on this axis.

    if positive >= ndim:
        raise ValueError(
            f"regions {A!r} and {B!r} overlap on every axis: full-dimensional "
            f"overlap, not a boundary contact"
        )
    return ContactDim(positive)


# ------------------------------------------------------------------ witness violations

@dataclass(frozen=True)
class WitnessViolation:
    """A realized-geometry violation of an :class:`IncidenceWitness`.

    ``code`` is one of:

    * ``"missing_region"``    — an incident region has no box in the realization.
    * ``"not_externally_connected"`` — the incident pair is not ``EC`` in the
      geometry, so the intended boundary contact is not realized at all.
    * ``"contact_dim_mismatch"`` — the pair *is* ``EC`` but the realized contact
      dimension differs from ``intended_contact_dimension`` (e.g. edge vs face).
    """
    witness_id: str
    code: str
    regions: tuple[str, str]
    intended: Optional[int] = None
    actual: Optional[int] = None
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.code}] witness {self.witness_id!r} {self.regions}: {self.detail}"


def _witness_pair_violations(
    w: IncidenceWitness, bm: dict[str, Box]
) -> list[WitnessViolation]:
    """Check every incident region pair of a witness against the geometry."""
    out: list[WitnessViolation] = []
    intended = int(w.intended_contact_dimension)
    ec = rcc8.bit("EC")
    for A, B in combinations(w.incident_regions, 2):
        if A not in bm or B not in bm:
            missing = A if A not in bm else B
            out.append(WitnessViolation(
                w.id, "missing_region", (A, B), intended, None,
                f"region {missing!r} has no box in the realization"))
            continue
        rel = relation_between_boxes(bm[A], bm[B])
        if rel != ec:
            out.append(WitnessViolation(
                w.id, "not_externally_connected", (A, B), intended, None,
                f"witness requires EC contact but geometry realizes "
                f"{rcc8.names(rel)}"))
            continue
        actual = int(extract_contact_dimension(
            Realization(id="", boxes=(bm[A], bm[B])), A, B))
        if actual != intended:
            out.append(WitnessViolation(
                w.id, "contact_dim_mismatch", (A, B), intended, actual,
                f"intended contact dimension {ContactDim(intended).name} "
                f"({intended}) but geometry realizes {ContactDim(actual).name} "
                f"({actual})"))
    return out


# ------------------------------------------------------------------ public: verify

@dataclass(frozen=True)
class VerifyReport:
    """Result of :func:`verify`. Also behaves like the requested dict via mapping."""
    satisfied: tuple[str, ...] = ()
    violated: tuple[str, ...] = ()
    undecided: tuple[str, ...] = ()
    witness_violations: tuple[WitnessViolation, ...] = ()

    def as_dict(self) -> dict:
        return {
            "satisfied": self.satisfied,
            "violated": self.violated,
            "undecided": self.undecided,
            "witness_violations": self.witness_violations,
        }

    # dict-style access so callers can do report["satisfied"] etc.
    def __getitem__(self, key: str):
        return self.as_dict()[key]

    def keys(self):
        return self.as_dict().keys()


def verify(spec: RelSpec, realization: Realization) -> VerifyReport:
    """Reverse-verify a realization against its spec, entirely in exact arithmetic.

    For each :class:`RelationConstraint` the realized RCC-8 relation of ``src``
    w.r.t. ``dst`` is re-extracted from the geometry:

    * if a referenced region has no box                    -> **undecided**
    * else if the realized base relation is *in* the mask  -> **satisfied**
    * else                                                 -> **violated**

    (An ``EMPTY`` mask admits no base relation, so any realized pair violates it.)

    Independently, every :class:`IncidenceWitness` is checked with
    :func:`extract_contact_dimension`: an ``EC``-but-wrong-dimension contact (the
    canonical *face demanded / edge realized* case) is reported in
    ``witness_violations`` even though the RCC-8 constraint itself is satisfied.
    """
    bm = _boxmap(realization)

    satisfied: list[str] = []
    violated: list[str] = []
    undecided: list[str] = []

    for c in spec.constraints:
        cid = c.id or f"{c.src}->{c.dst}"
        if c.src not in bm or c.dst not in bm:
            undecided.append(cid)
            continue
        rel = relation_between_boxes(bm[c.src], bm[c.dst])
        if rel & c.rcc8_mask:
            satisfied.append(cid)
        else:
            violated.append(cid)

    witness_violations: list[WitnessViolation] = []
    for w in spec.witnesses:
        witness_violations.extend(_witness_pair_violations(w, bm))

    return VerifyReport(
        satisfied=tuple(satisfied),
        violated=tuple(violated),
        undecided=tuple(undecided),
        witness_violations=tuple(witness_violations),
    )


__all__ = [
    "relation_between_boxes",
    "extract_relations",
    "extract_contact_dimension",
    "verify",
    "VerifyReport",
    "WitnessViolation",
]
