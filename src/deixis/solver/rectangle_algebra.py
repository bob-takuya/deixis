"""Rectangle Algebra — the two-dimensional product of Allen's interval algebra,
lifted into the symbolic layer as a *sibling* of RCC-8 over the same grounding.

Where :mod:`deixis.solver.geom_solver` encodes an axis-aligned box as a pair of
1-D coordinate-interval constraints (one per axis) and hands them to Z3, this
module raises that *per-axis interval* view up into a purely symbolic relation
algebra. Following Guesgen (1989) and Balbiani-Condotti-Fariñas del Cerro (1998),
the qualitative relation between two axis-aligned rectangles is the **pair** of
Allen relations holding on the x-axis and the y-axis independently:

    RectRel = (allen_x_mask, allen_y_mask)

With 13 Allen base relations per axis there are 13x13 = 169 rectangle base
relations. Each component is an Allen 13-bit mask (see :mod:`deixis.solver.allen`),
so a component may be a *disjunction* under incomplete knowledge exactly as in the
1-D algebra; a fully-determined rectangle relation has a single base bit on each
axis. This module never re-implements Allen — it *imports* :mod:`allen` unchanged
and only combines its per-axis results.

Two distinct symbolic readings of the SAME geometry
---------------------------------------------------
RCC-8 (:mod:`deixis.core.rcc8`) and Rectangle Algebra are two different qualitative
abstractions of one box pair. RCC-8 keeps mereotopological class (touch / overlap /
containment) but *collapses direction*; Rectangle Algebra keeps directional /
projection detail per axis but is specific to axis-aligned rectangles.
:func:`rcc8_from_rect` is the exact base-level projection from the finer rectangle
relation down to RCC-8 — a surjective, many-to-one map (NOT claimed to be a
relation-algebra homomorphism) that reproduces, base for base, the RCC-8 relation
:func:`deixis.verify.reverse.relation_between_boxes` extracts from the same boxes.
There is no inverse (RCC-8 -> RectRel loses the axis split); no completeness of the
rectangle algebra as a reasoning system is claimed here.

Exactness / honesty
-------------------
:func:`rect_from_boxes` classifies each axis by comparing the two closed intervals'
:class:`fractions.Fraction` endpoints — exact and total for non-degenerate boxes
(lo < hi on every axis), never float. :func:`converse` and :func:`compose` delegate
to the audited Allen tables component-wise, so ``compose`` inherits Allen's
*weak-composition* semantics: it returns the tightest entailed mask per axis, which
is sound (a superset of every realizable relation) but not a satisfiability test.
"""
from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

from ..core import rcc8
from ..core.types import Box
from . import allen


# ---------------------------------------------------------------- the relation type

class RectRel(NamedTuple):
    """A rectangle-algebra relation: an Allen mask per axis (x first, then y).

    Each field is a 13-bit Allen relation mask (:mod:`deixis.solver.allen`). A
    fully-determined relation extracted from geometry has exactly one base bit set
    on each axis; :func:`compose` may yield disjunction masks. Being a
    :class:`~typing.NamedTuple`, a ``RectRel`` also unpacks as the plain pair
    ``(allen_x_mask, allen_y_mask)``.
    """

    allen_x_mask: int
    allen_y_mask: int


def _check_mask(m: int, axis: str) -> int:
    if not isinstance(m, int) or isinstance(m, bool):
        raise TypeError(f"{axis}-axis Allen mask must be a plain int, got {type(m)!r}")
    if m < 0 or m > allen.UNIVERSAL:
        raise ValueError(
            f"{axis}-axis Allen mask out of range 0..{allen.UNIVERSAL}: {m}"
        )
    return m


def make_rect(allen_x_mask: int, allen_y_mask: int) -> RectRel:
    """Construct a validated :class:`RectRel` from two Allen masks."""
    return RectRel(
        _check_mask(allen_x_mask, "x"),
        _check_mask(allen_y_mask, "y"),
    )


# ---------------------------------------------------------------- axis extraction

def _axis_allen_name(a_lo: Fraction, a_hi: Fraction,
                     b_lo: Fraction, b_hi: Fraction) -> str:
    """Exact Allen base relation of closed interval A=[a_lo,a_hi] w.r.t. B=[b_lo,b_hi].

    Both intervals must be non-degenerate (lo < hi). Returns exactly one of the 13
    JEPD base-relation names. Pure Fraction comparison; the whole classification is
    driven by the four endpoint orderings.
    """
    if not (a_lo < a_hi):
        raise ValueError(f"degenerate interval A: [{a_lo}, {a_hi}] (need lo < hi)")
    if not (b_lo < b_hi):
        raise ValueError(f"degenerate interval B: [{b_lo}, {b_hi}] (need lo < hi)")

    # 1. no interior overlap: A entirely before / after B (with meet as the boundary).
    if a_hi < b_lo:
        return "b"
    if a_hi == b_lo:
        return "m"
    if a_lo > b_hi:
        return "bi"
    if a_lo == b_hi:
        return "mi"

    # 2. interiors overlap: resolve by comparing the two starts and the two ends.
    #    cs = sign(a_lo - b_lo), ce = sign(a_hi - b_hi).
    if a_lo == b_lo:
        if a_hi == b_hi:
            return "eq"
        return "s" if a_hi < b_hi else "si"
    if a_hi == b_hi:
        return "f" if a_lo > b_lo else "fi"
    # both starts and both ends strictly ordered.
    if a_lo < b_lo:
        # a starts first; ends before => overlaps, ends after => contains.
        return "o" if a_hi < b_hi else "di"
    # a starts after b.
    return "d" if a_hi < b_hi else "oi"


def _check_2d_box(box: Box) -> None:
    if len(box.lo) != 2 or len(box.hi) != 2:
        raise ValueError(
            f"rectangle algebra requires 2-D boxes; box {box.region_id!r} has "
            f"arity lo={len(box.lo)} hi={len(box.hi)}"
        )
    for d in range(2):
        lo, hi = box.lo[d], box.hi[d]
        if not isinstance(lo, Fraction) or not isinstance(hi, Fraction):
            raise TypeError(
                f"box {box.region_id!r} axis {d} bounds must be exact Fraction, "
                f"got {type(lo).__name__}/{type(hi).__name__}"
            )
        if not lo < hi:
            raise ValueError(
                f"box {box.region_id!r} is degenerate on axis {d}: lo={lo} hi={hi}"
            )


def rect_from_boxes(box_a: Box, box_b: Box) -> RectRel:
    """Exact rectangle relation of ``box_a`` w.r.t. ``box_b`` (both 2-D AABBs).

    Returns a :class:`RectRel` whose x and y components are each a single Allen base
    bit — the projection of the two rectangles onto that axis, classified exactly
    from Fraction endpoints. Corner / edge contact and area overlap are all captured
    faithfully: e.g. a shared corner has each axis *meeting* — a single base ``m``
    or ``mi`` per axis depending on orientation — while a shared vertical edge is one
    axis *meets* and the other axis interior-overlaps.
    """
    _check_2d_box(box_a)
    _check_2d_box(box_b)
    x = allen.bit(_axis_allen_name(box_a.lo[0], box_a.hi[0], box_b.lo[0], box_b.hi[0]))
    y = allen.bit(_axis_allen_name(box_a.lo[1], box_a.hi[1], box_b.lo[1], box_b.hi[1]))
    return RectRel(x, y)


# ---------------------------------------------------------------- converse / compose

def converse(rect: RectRel) -> RectRel:
    """Converse of ``rect``: if ``rect`` is (A rel B) then this is (B rel A).

    Rectangle converse is component-wise Allen converse — the axes do not interact,
    so it is exactly ``(allen.converse(x), allen.converse(y))``.
    """
    return RectRel(
        allen.converse(_check_mask(rect.allen_x_mask, "x")),
        allen.converse(_check_mask(rect.allen_y_mask, "y")),
    )


def compose(rect_ab: RectRel, rect_bc: RectRel) -> RectRel:
    """Compose rectangle relations: (A rect_ab B) then (B rect_bc C) -> (A ? C).

    Because the two axes are independent, rectangle composition is Allen composition
    applied separately on each axis (Balbiani et al. 1998). This inherits Allen's
    *weak* composition: each component is the tightest entailed Allen mask (a sound
    superset of the realizable relations), not a satisfiability decision.
    """
    return RectRel(
        allen.compose(_check_mask(rect_ab.allen_x_mask, "x"),
                      _check_mask(rect_bc.allen_x_mask, "x")),
        allen.compose(_check_mask(rect_ab.allen_y_mask, "y"),
                      _check_mask(rect_bc.allen_y_mask, "y")),
    )


# ---------------------------------------------------------------- rect -> RCC-8

# Per-axis Allen base -> its role in the mereotopological classification of a box
# pair. These mirror, one-to-one, the axis predicates in
# deixis.verify.reverse.relation_between_boxes (_sep / _touch / _iover / _asub /
# _strict_inside), just expressed over the Allen base name of that axis.
_SEP_NAMES = frozenset({"b", "bi"})            # positive gap on this axis  -> DC driver
_TOUCH_NAMES = frozenset({"m", "mi"})          # meet at one endpoint, no interior overlap
# interior overlap = every other name.
_A_SUB_B = frozenset({"s", "d", "f", "eq"})    # A's closed interval subset of B's
_A_STRICT_IN_B = frozenset({"d"})              # A strictly interior to B (no shared endpoint)
_B_SUB_A = frozenset({"si", "di", "fi", "eq"})  # B's closed interval subset of A's
_B_STRICT_IN_A = frozenset({"di"})             # B strictly interior to A


def _base_rcc8(x_name: str, y_name: str) -> int:
    """RCC-8 base bit of the box pair whose axes realize Allen bases ``x_name``/``y_name``.

    This is the exact 2-D reconstruction used by reverse.relation_between_boxes,
    reproduced here from the two axis relations alone.
    """
    names = (x_name, y_name)

    # 1. separated on either axis => closures disjoint.
    if any(n in _SEP_NAMES for n in names):
        return rcc8.bit("DC")

    # 2. some axis only touches (no interior overlap) => boundaries meet, interiors
    #    disjoint => externally connected.
    if any(n in _TOUCH_NAMES for n in names):
        return rcc8.bit("EC")

    # 3. interiors overlap on both axes: resolve by containment of closed intervals.
    a_in_b = all(n in _A_SUB_B for n in names)
    b_in_a = all(n in _B_SUB_A for n in names)

    if a_in_b and b_in_a:
        return rcc8.bit("EQ")
    if a_in_b:  # proper part
        if all(n in _A_STRICT_IN_B for n in names):
            return rcc8.bit("NTPP")
        return rcc8.bit("TPP")
    if b_in_a:  # proper part (converse)
        if all(n in _B_STRICT_IN_A for n in names):
            return rcc8.bit("NTPPi")
        return rcc8.bit("TPPi")
    return rcc8.bit("PO")


def rcc8_from_rect(rect: RectRel) -> int:
    """Project a rectangle relation down onto its entailed RCC-8 relation mask.

    For a fully-determined ``rect`` (a single Allen base bit on each axis) this
    returns the single RCC-8 base bit that
    :func:`deixis.verify.reverse.relation_between_boxes` extracts from any boxes
    realizing that rectangle relation. If a component is a *disjunction* (as
    :func:`compose` may produce) the result is the union of the RCC-8 bits over
    every base combination — the tightest RCC-8 mask entailed. The map is
    surjective onto RCC-8 but not injective: the axis split is not recoverable.
    """
    xs = allen.names(_check_mask(rect.allen_x_mask, "x"))
    ys = allen.names(_check_mask(rect.allen_y_mask, "y"))
    if not xs or not ys:
        # EMPTY on either axis => no realizable base combination.
        return rcc8.EMPTY
    out = 0
    for xn in xs:
        for yn in ys:
            out |= _base_rcc8(xn, yn)
    return out


__all__ = [
    "RectRel",
    "make_rect",
    "rect_from_boxes",
    "converse",
    "compose",
    "rcc8_from_rect",
]
