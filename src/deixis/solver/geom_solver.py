"""Bounded geometric solver: RCC-8 relations + incidence witnesses -> exact AABBs.

We take a :class:`~deixis.core.types.RelSpec` (its regions, RelationConstraints and
IncidenceWitnesses) and *realize* it as axis-aligned boxes (AABBs) with **exact
rational** coordinates, by lowering every RCC-8 base relation to a set of *linear*
constraints over the box bounds and discharging them with Z3 (``z3-solver``) in the
theory of linear real arithmetic (LRA).

Why this is sound within the selected domain
--------------------------------------------
For two AABBs ``A`` and ``B`` the RCC-8 relation is fully determined, axis by axis,
by the relation between the two closed coordinate intervals (this is the *Rectangle
Algebra* view). Writing, per axis ``d``:

* ``iover_d``  (interiors overlap):  ``A.lo[d] < B.hi[d]  and  B.lo[d] < A.hi[d]``
* ``touch_d``  (share only a boundary): ``A.hi[d] == B.lo[d]  or  B.hi[d] == A.lo[d]``
* ``sep_d``    (strictly apart):       ``A.hi[d] < B.lo[d]  or  B.hi[d] < A.lo[d]``
* ``asub_d``   (A's interval inside B's): ``A.lo[d] >= B.lo[d]  and  A.hi[d] <= B.hi[d]``

the 8 jointly-exhaustive / pairwise-disjoint (JEPD) base relations become:

* **DC**   : ``Or_d sep_d``                              (separated on some axis)
* **EC**   : connected on every axis AND ``Or_d touch_d``  (touch, interiors disjoint)
* **PO**   : ``And_d iover_d`` AND not(A subset B) AND not(B subset A)
* **TPP**  : A subset B, proper, and tangential (a boundary coincides)
* **NTPP** : A strictly interior to B on every axis
* **TPPi/NTPPi** : the converse (swap A,B)
* **EQ**   : ``And_d (A.lo[d]==B.lo[d] and A.hi[d]==B.hi[d])``

Because the 8 encodings are exactly the JEPD partition, asking Z3 for a model of the
encoding of a single base relation yields boxes *in exactly that relation*.

Contact dimension (what EC collapses)
-------------------------------------
An :class:`IncidenceWitness` records ``intended_contact_dimension`` (a
:class:`ContactDim` value 0/1/2). For an EC pair we honour it: a contact of
dimension ``k`` means **exactly ``k`` axes have interior overlap and the remaining
``ndim - k`` axes touch at a coincident boundary**. So in 2D, ``k=1`` (LINE) is an
*edge share* and ``k=0`` (POINT) is a *corner touch*; in 3D ``k=2`` (FACE) is a face
share, ``k=1`` an edge, ``k=0`` a corner.

All coordinates in the returned :class:`Realization` are exact ``Fraction`` values
read out of the Z3 rational model (never float).
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Optional

import z3

from ..core.types import (
    Box,
    ContactDim,
    IncidenceWitness,
    Realizability,
    Realization,
    RelSpec,
)
from ..core import rcc8

# ------------------------------------------------------------------ domains

_DOMAIN_NDIM = {
    "aabb_2d": 2,
    "aabb_3d": 3,
}


# ------------------------------------------------------------------ z3 helpers

def _q(x) -> z3.ArithRef:
    """Exact rational Z3 numeral from an int / Fraction / str."""
    return z3.RealVal(Fraction(x) if not isinstance(x, str) else Fraction(x))


def _z3_to_fraction(v) -> Fraction:
    """Exactly convert a Z3 numeral (from a model) to a :class:`Fraction`."""
    if z3.is_int_value(v):
        return Fraction(v.as_long())
    if z3.is_rational_value(v):
        return Fraction(v.numerator_as_long(), v.denominator_as_long())
    # Linear real arithmetic never yields irrationals; guard defensively.
    raise ValueError(f"non-rational value in model: {v!r}")


class _BoxVars:
    """Z3 real variables for one region's AABB bounds."""

    __slots__ = ("region_id", "lo", "hi", "ndim")

    def __init__(self, region_id: str, ndim: int):
        self.region_id = region_id
        self.ndim = ndim
        safe = region_id.replace(" ", "_")
        self.lo = tuple(z3.Real(f"{safe}__lo{d}") for d in range(ndim))
        self.hi = tuple(z3.Real(f"{safe}__hi{d}") for d in range(ndim))


# ------------------------------------------------------------------ per-axis predicates

def _iover(a: _BoxVars, b: _BoxVars, d: int):
    """Interiors overlap on axis d (positive-length overlap)."""
    return z3.And(a.lo[d] < b.hi[d], b.lo[d] < a.hi[d])


def _closed_intersect(a: _BoxVars, b: _BoxVars, d: int):
    """Closed intervals intersect on axis d (overlap or touch)."""
    return z3.And(a.lo[d] <= b.hi[d], b.lo[d] <= a.hi[d])


def _touch(a: _BoxVars, b: _BoxVars, d: int):
    """Closed intervals meet at exactly a shared endpoint on axis d (no interior overlap)."""
    return z3.Or(a.hi[d] == b.lo[d], b.hi[d] == a.lo[d])


def _sep(a: _BoxVars, b: _BoxVars, d: int):
    """Strictly separated on axis d (a gap)."""
    return z3.Or(a.hi[d] < b.lo[d], b.hi[d] < a.lo[d])


def _asub(a: _BoxVars, b: _BoxVars, d: int):
    """a's interval contained in b's interval on axis d."""
    return z3.And(a.lo[d] >= b.lo[d], a.hi[d] <= b.hi[d])


def _strict_inside(a: _BoxVars, b: _BoxVars, d: int):
    """a's interval strictly interior to b's on axis d (no shared boundary)."""
    return z3.And(a.lo[d] > b.lo[d], a.hi[d] < b.hi[d])


def _eq_axis(a: _BoxVars, b: _BoxVars, d: int):
    return z3.And(a.lo[d] == b.lo[d], a.hi[d] == b.hi[d])


def _subseteq(a: _BoxVars, b: _BoxVars, ndim: int):
    return z3.And(*[_asub(a, b, d) for d in range(ndim)])


def _equal(a: _BoxVars, b: _BoxVars, ndim: int):
    return z3.And(*[_eq_axis(a, b, d) for d in range(ndim)])


# ------------------------------------------------------------------ base-relation encodings
# Each returns a Z3 BoolRef that holds iff (A base B), where A=src box, B=dst box.

def _enc_DC(a, b, ndim):
    return z3.Or(*[_sep(a, b, d) for d in range(ndim)])


def _enc_EC(a, b, ndim, contact_dim: Optional[int] = None):
    """Externally connected: touch, interiors disjoint.

    ``contact_dim`` (a ContactDim value) pins the dimension of the contact set:
    exactly ``contact_dim`` axes have interior overlap and the rest touch at a
    coincident boundary. If ``None`` we accept *any* EC (union over contact dims).
    """
    connected = z3.And(*[_closed_intersect(a, b, d) for d in range(ndim)])
    if contact_dim is None:
        some_touch = z3.Or(*[_touch(a, b, d) for d in range(ndim)])
        return z3.And(connected, some_touch)

    if not (0 <= contact_dim < ndim):
        # A contact of dimension ndim would be interior overlap (=> not EC), and a
        # negative dimension is meaningless: no EC configuration can satisfy it.
        return z3.BoolVal(False)

    axes = range(ndim)
    clauses = []
    for overlap_axes in combinations(axes, contact_dim):
        oset = set(overlap_axes)
        conj = []
        for d in axes:
            conj.append(_iover(a, b, d) if d in oset else _touch(a, b, d))
        clauses.append(z3.And(*conj))
    return z3.Or(*clauses) if clauses else z3.BoolVal(False)


def _enc_PO(a, b, ndim):
    interiors = z3.And(*[_iover(a, b, d) for d in range(ndim)])
    not_a_in_b = z3.Not(_subseteq(a, b, ndim))
    not_b_in_a = z3.Not(_subseteq(b, a, ndim))
    return z3.And(interiors, not_a_in_b, not_b_in_a)


def _enc_NTPP(a, b, ndim):
    return z3.And(*[_strict_inside(a, b, d) for d in range(ndim)])


def _enc_TPP(a, b, ndim):
    subset = _subseteq(a, b, ndim)
    tangential = z3.Or(*[z3.Or(a.lo[d] == b.lo[d], a.hi[d] == b.hi[d]) for d in range(ndim)])
    proper = z3.Or(*[z3.Or(a.lo[d] > b.lo[d], a.hi[d] < b.hi[d]) for d in range(ndim)])
    return z3.And(subset, tangential, proper)


def _enc_NTPPi(a, b, ndim):
    return _enc_NTPP(b, a, ndim)


def _enc_TPPi(a, b, ndim):
    return _enc_TPP(b, a, ndim)


def _enc_EQ(a, b, ndim):
    return _equal(a, b, ndim)


_BASE_ENCODERS = {
    "DC": _enc_DC,
    "PO": _enc_PO,
    "NTPP": _enc_NTPP,
    "TPP": _enc_TPP,
    "NTPPi": _enc_NTPPi,
    "TPPi": _enc_TPPi,
    "EQ": _enc_EQ,
}


def _encode_base(name: str, a: _BoxVars, b: _BoxVars, ndim: int,
                 contact_dim: Optional[int]):
    if name == "EC":
        return _enc_EC(a, b, ndim, contact_dim)
    return _BASE_ENCODERS[name](a, b, ndim)


# ------------------------------------------------------------------ witness lookup

def _contact_dim_for(spec: RelSpec, src: str, dst: str) -> Optional[int]:
    """intended_contact_dimension of a witness relating exactly {src,dst}, if any."""
    pair = frozenset((src, dst))
    for w in spec.witnesses:
        if pair <= frozenset(w.incident_regions):
            return int(w.intended_contact_dimension)
    return None


# ------------------------------------------------------------------ public API

def solve_boxes(
    spec: RelSpec,
    domain: str = "aabb_2d",
    bounds: tuple = (0, 100),
    *,
    min_size: Optional[Fraction] = None,
    realization_id: str = "geom:0",
) -> Realization:
    """Realize ``spec`` as exact-rational AABBs satisfying its RCC-8 constraints.

    Parameters
    ----------
    spec : RelSpec
        Source of ``regions``, ``constraints`` (RCC-8 masks) and ``witnesses``
        (contact dimensions).
    domain : str
        ``'aabb_2d'`` or ``'aabb_3d'``.
    bounds : (lo, hi)
        Inclusive coordinate box every AABB must live inside (all axes).
    min_size : Fraction, optional
        Minimum edge length for every box (default: strictly positive area only).
    realization_id : str
        Id for the produced :class:`Realization`.

    Returns
    -------
    Realization
        On SAT: ``status = REALIZED_IN_D``, exact-Fraction ``boxes``, and every
        constraint id in ``satisfied``. On UNSAT: ``status = INCONSISTENT`` with
        the constraint ids in ``undecided`` (no attribution of blame). On timeout /
        unknown: ``status = UNKNOWN``.
    """
    if domain not in _DOMAIN_NDIM:
        raise ValueError(f"unsupported domain {domain!r}; use one of {sorted(_DOMAIN_NDIM)}")
    ndim = _DOMAIN_NDIM[domain]

    blo = Fraction(bounds[0])
    bhi = Fraction(bounds[1])
    if not (blo < bhi):
        raise ValueError(f"empty bounds: {bounds!r}")

    # region variables (stable order = declaration order)
    boxvars: dict[str, _BoxVars] = {}
    for r in spec.regions:
        boxvars[r.id] = _BoxVars(r.id, ndim)

    solver = z3.Solver()

    # well-formedness: inside bounds and non-degenerate (positive edge length).
    for bv in boxvars.values():
        for d in range(ndim):
            solver.add(bv.lo[d] >= _q(blo))
            solver.add(bv.hi[d] <= _q(bhi))
            if min_size is not None:
                solver.add(bv.hi[d] - bv.lo[d] >= _q(Fraction(min_size)))
            else:
                solver.add(bv.lo[d] < bv.hi[d])

    constraint_ids: list[str] = []
    for c in spec.constraints:
        cid = c.id or f"{c.src}->{c.dst}"
        constraint_ids.append(cid)

        a = boxvars.get(c.src)
        b = boxvars.get(c.dst)
        if a is None or b is None:
            missing = c.src if a is None else c.dst
            raise ValueError(f"constraint {cid!r} references unknown region {missing!r}")

        base_names = rcc8.names(c.rcc8_mask)
        if not base_names:
            # EMPTY mask = contradictory relation: unsatisfiable by construction.
            solver.add(z3.BoolVal(False))
            continue

        contact_dim = _contact_dim_for(spec, c.src, c.dst)
        disjunction = [
            _encode_base(name, a, b, ndim, contact_dim) for name in base_names
        ]
        solver.add(z3.Or(*disjunction) if len(disjunction) > 1 else disjunction[0])

    result = solver.check()

    if result == z3.sat:
        model = solver.model()
        boxes = []
        for r in spec.regions:
            bv = boxvars[r.id]
            lo = tuple(_z3_to_fraction(model.eval(bv.lo[d], model_completion=True))
                       for d in range(ndim))
            hi = tuple(_z3_to_fraction(model.eval(bv.hi[d], model_completion=True))
                       for d in range(ndim))
            boxes.append(Box(region_id=r.id, lo=lo, hi=hi))
        return Realization(
            id=realization_id,
            boxes=tuple(boxes),
            status=Realizability.REALIZED_IN_D,
            satisfied=tuple(constraint_ids),
            violated=(),
            undecided=(),
        )

    if result == z3.unsat:
        return Realization(
            id=realization_id,
            boxes=(),
            status=Realizability.INCONSISTENT,
            satisfied=(),
            violated=(),
            undecided=tuple(constraint_ids),
        )

    # z3.unknown (timeout / incompleteness)
    return Realization(
        id=realization_id,
        boxes=(),
        status=Realizability.UNKNOWN,
        satisfied=(),
        violated=(),
        undecided=tuple(constraint_ids),
    )


__all__ = ["solve_boxes"]
