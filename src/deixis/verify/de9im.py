"""DE-9IM (Dimensionally Extended 9-Intersection Model) for axis-aligned boxes.

Where :mod:`deixis.verify.reverse` collapses a box pair to a single RCC-8 base
relation, this module recovers the *finer* topological signature RCC-8 discards:
the full 3x3 matrix of the **dimension** of the intersection of the interior /
boundary / exterior of one region with those of the other.

Model
-----
Each :class:`~deixis.core.types.Box` is treated as a **filled, full-dimensional**
axis-aligned region ``R = ∏_d [lo_d, hi_d]`` in ``ndim``-space. Its point set
decomposes topologically into three disjoint, exhaustive parts:

* **Interior**   ``I(R)`` — the open box ``∏_d (lo_d, hi_d)`` (dimension ``ndim``).
* **Boundary**   ``∂(R)`` — closure minus interior: the ``(ndim-1)``-dimensional
  shell (faces/edges/corners).
* **Exterior**   ``E(R)`` — the complement of the closed box (dimension ``ndim``).

The DE-9IM matrix of ``a`` w.r.t. ``b`` is, in row-major order,

    ┌ dim(Iₐ∩I_b)  dim(Iₐ∩∂_b)  dim(Iₐ∩E_b) ┐
    │ dim(∂ₐ∩I_b)  dim(∂ₐ∩∂_b)  dim(∂ₐ∩E_b) │
    └ dim(Eₐ∩I_b)  dim(Eₐ∩∂_b)  dim(Eₐ∩E_b) ┘

where each entry is the topological dimension of the intersection set: ``-1`` for
the empty set (rendered ``'F'``), else ``0`` point / ``1`` line / ``2`` area /
``3`` volume … up to ``ndim``.

Exactness
---------
Every dimension is computed by an **exact arrangement** of the two boxes: the four
per-axis endpoints partition each coordinate line into open segments and points;
their Cartesian product tiles ``ℝ^ndim`` into cells on each of which the
(interior/boundary/exterior) membership w.r.t. *both* boxes is constant. The
dimension of an intersection set is the max dimension over the cells that fall in
it. All comparisons are on :class:`fractions.Fraction` endpoints — never float —
so the matrix is exact and total for every non-degenerate box pair.

Completeness / limits
---------------------
This is **not** a general DE-9IM engine. It is sound and complete only for the
selected geometric domain of this project:

* **Axis-aligned boxes only** (AABBs). Arbitrary polygons/polylines/points are
  out of scope.
* Each box is a **regular** region: its interior is homeomorphic to an open ball
  and its closure to a closed ball (one connected, simply-connected full-
  dimensional cell), with the boundary their shared frontier. (The *exterior*
  need not be so well behaved — e.g. in 1D it is disconnected, in 2D it is not
  simply connected — but only interior/boundary regularity is needed here.) DE-9IM
  is strictly *finer* than RCC-8, so the forward map DE-9IM → RCC-8 is always a
  (many-to-one) function while the inverse RCC-8 → DE-9IM is not. What the AABB
  restriction buys is that the finitely many 9-IM matrices this module actually
  emits fall into 8 classes that :func:`rcc8_from_de9im` reads off correctly and
  totally; for general regions with disconnected interiors or holes those same
  classes would not carve out the RCC-8 relations, so we do not handle them.
* No completeness is claimed outside AABBs. ``rcc8_from_de9im`` is a total
  function on the 9-IM matrices *produced by this module* (it agrees, box for box,
  with :func:`deixis.verify.reverse.relation_between_boxes`); it is not claimed
  correct on arbitrary 3x3 patterns from other geometry kinds.

``shapely_interop``: none is provided. Shapely's ``object.relate(other)`` returns
the same 9-character DE-9IM string for planar geometries; for two rectangles its
string equals :func:`de9im_string` here. We only note the correspondence — we do
not import shapely (it uses floating-point GEOS and would break exactness).
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Sequence

from ..core.types import Box
from ..core import rcc8

# Row/column order of the matrix: 0=Interior, 1=Boundary, 2=Exterior.
_INTERIOR = 0
_BOUNDARY = 1
_EXTERIOR = 2

#: Human-readable labels for the 9 entries, row-major (matches the tuple order).
DE9IM_LABELS: tuple[str, ...] = (
    "II", "IB", "IE",
    "BI", "BB", "BE",
    "EI", "EB", "EE",
)

EMPTY_DIM = -1  # the "F" (empty intersection) sentinel


# ------------------------------------------------------------------ validation

def _check_box(a: Box, ndim: int) -> None:
    if len(a.lo) != ndim or len(a.hi) != ndim:
        raise ValueError(
            f"box {a.region_id!r} has arity {len(a.lo)}/{len(a.hi)}, expected ndim={ndim}"
        )
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


# ------------------------------------------------------------------ per-axis pieces

def _axis_state(r: Fraction, lo: Fraction, hi: Fraction) -> int:
    """Membership of representative ``r`` w.r.t. interval [lo, hi]:
    _INTERIOR (lo<r<hi), _BOUNDARY (r==lo or r==hi), _EXTERIOR (r<lo or r>hi)."""
    if r < lo or r > hi:
        return _EXTERIOR
    if r == lo or r == hi:
        return _BOUNDARY
    return _INTERIOR


def _axis_pieces(a: Box, b: Box, d: int) -> list[tuple[int, int, int]]:
    """Partition coordinate axis ``d`` into pieces induced by the 4 box endpoints.

    Returns a list of ``(piece_dim, state_a, state_b)`` — one entry per open
    segment (dim 1) and per cut point (dim 0), covering ``(-inf, +inf)``. The
    infinite end segments are dimension 1 as well. States are computed from an
    exact rational representative of each piece.
    """
    cuts = sorted({a.lo[d], a.hi[d], b.lo[d], b.hi[d]})
    reps: list[tuple[Fraction, int]] = []  # (representative, piece_dim)
    # left open ray (-inf, cuts[0])
    reps.append((cuts[0] - 1, 1))
    for i, c in enumerate(cuts):
        reps.append((c, 0))                       # the cut point itself
        if i + 1 < len(cuts):
            reps.append(((c + cuts[i + 1]) / 2, 1))  # open segment to next cut
    # right open ray (cuts[-1], +inf)
    reps.append((cuts[-1] + 1, 1))

    out: list[tuple[int, int, int]] = []
    for r, pdim in reps:
        out.append((pdim,
                    _axis_state(r, a.lo[d], a.hi[d]),
                    _axis_state(r, b.lo[d], b.hi[d])))
    return out


# ------------------------------------------------------------------ region type

def _region_type(axis_states: Sequence[int]) -> int:
    """Global interior/boundary/exterior of a point from its per-axis states.

    * all axes interior            -> _INTERIOR
    * any axis exterior            -> _EXTERIOR   (outside the closed box)
    * else (no exterior, ≥1 axis boundary, rest interior) -> _BOUNDARY
    These three are disjoint and exhaustive, so every cell maps to exactly one.
    """
    if any(s == _EXTERIOR for s in axis_states):
        return _EXTERIOR
    if all(s == _INTERIOR for s in axis_states):
        return _INTERIOR
    return _BOUNDARY


# ------------------------------------------------------------------ public: matrix

def de9im_matrix(box_a: Box, box_b: Box, ndim: int) -> tuple[int, ...]:
    """Exact DE-9IM matrix of ``box_a`` w.r.t. ``box_b`` as a 9-tuple.

    The tuple is row-major (see :data:`DE9IM_LABELS`): entry ``k`` is the
    dimension of ``type_a(k//3) ∩ type_b(k%3)`` where ``0=interior, 1=boundary,
    2=exterior``. Each value is in ``-1..ndim`` (``-1`` = empty).
    """
    if ndim < 1:
        raise ValueError(f"ndim must be >= 1, got {ndim}")
    _check_box(box_a, ndim)
    _check_box(box_b, ndim)

    # per-axis pieces
    pieces = [_axis_pieces(box_a, box_b, d) for d in range(ndim)]

    # matrix[type_a][type_b], initialized to empty (-1)
    mat = [[EMPTY_DIM, EMPTY_DIM, EMPTY_DIM] for _ in range(3)]

    for cell in product(*pieces):
        # cell is a tuple of (piece_dim, state_a, state_b), one per axis.
        cell_dim = 0
        states_a: list[int] = []
        states_b: list[int] = []
        for pdim, sa, sb in cell:
            cell_dim += pdim
            states_a.append(sa)
            states_b.append(sb)
        ta = _region_type(states_a)
        tb = _region_type(states_b)
        if cell_dim > mat[ta][tb]:
            mat[ta][tb] = cell_dim

    return (
        mat[0][0], mat[0][1], mat[0][2],
        mat[1][0], mat[1][1], mat[1][2],
        mat[2][0], mat[2][1], mat[2][2],
    )


# ------------------------------------------------------------------ public: string

def de9im_string(matrix: Sequence[int]) -> str:
    """9-character DE-9IM pattern string for ``matrix`` (row-major 9-tuple).

    ``-1`` -> ``'F'`` (empty); ``0/1/2/3`` -> its digit. E.g. two equal 2D boxes
    give ``"2FFF1FFF2"``; two 2D boxes sharing an edge give ``"FF2F11212"``.

    The one-char-per-entry pattern is only defined for dimensions ``-1..9`` (an
    entry above 9 raises); this covers every realistic ``ndim`` (the project's
    domain is ``aabb_2d`` / ``aabb_3d``), but is *not* total for the pathological
    ``ndim > 9`` boxes that :func:`de9im_matrix` would otherwise accept.
    """
    if len(matrix) != 9:
        raise ValueError(f"DE-9IM matrix must have 9 entries, got {len(matrix)}")
    chars: list[str] = []
    for v in matrix:
        if v == EMPTY_DIM:
            chars.append("F")
        elif 0 <= v <= 9:
            chars.append(str(v))
        else:
            raise ValueError(f"DE-9IM entry out of range -1..9: {v}")
    return "".join(chars)


# ------------------------------------------------------------------ public: 9-IM -> RCC-8

def rcc8_from_de9im(matrix: Sequence[int]) -> int:
    """Map a DE-9IM matrix (as produced here for two AABBs) to an RCC-8 base mask.

    Returns exactly one JEPD base relation (single-bit mask), agreeing box-for-box
    with :func:`deixis.verify.reverse.relation_between_boxes`. The classification
    uses only whether entries are empty (``F``), never their dimension:

    * ``II = F``:
        - boundaries also disjoint (``IB=BI=BB=F``)  -> **DC**
        - else                                       -> **EC**
    * ``II ≠ F`` (interiors overlap):
        - ``IE=F`` and ``EI=F``                      -> **EQ**
        - ``IE≠F`` and ``EI≠F``                      -> **PO**
        - ``IE=F`` and ``EI≠F`` (a ⊆ b, proper):
            ``BB=F`` -> **NTPP**  else -> **TPP**
        - ``IE≠F`` and ``EI=F`` (b ⊆ a, proper):
            ``BB=F`` -> **NTPPi** else -> **TPPi**

    .. warning::
       This inverse is a *function* only for the disk-homeomorphic AABB regions
       this module emits (see the module docstring). It is **not** claimed correct
       for 9-IM patterns arising from other geometry kinds.
    """
    if len(matrix) != 9:
        raise ValueError(f"DE-9IM matrix must have 9 entries, got {len(matrix)}")
    II, IB, IE, BI, BB, BE, EI, EB, EE = matrix
    F = EMPTY_DIM

    if II == F:
        if IB == F and BI == F and BB == F:
            return rcc8.bit("DC")
        return rcc8.bit("EC")

    a_out = IE != F   # a's interior reaches b's exterior
    b_out = EI != F   # b's interior reaches a's exterior

    if not a_out and not b_out:
        return rcc8.bit("EQ")
    if a_out and b_out:
        return rcc8.bit("PO")
    if not a_out and b_out:              # a is a proper part of b
        return rcc8.bit("TPP" if BB != F else "NTPP")
    # a_out and not b_out                 # b is a proper part of a
    return rcc8.bit("TPPi" if BB != F else "NTPPi")


__all__ = [
    "de9im_matrix",
    "de9im_string",
    "rcc8_from_de9im",
    "DE9IM_LABELS",
    "EMPTY_DIM",
]
