"""M-field (c) — the **visibility field** (isovist), held as a plain scalar field.

An *isovist* (Benedikt, 1979, "To take hold of space: isovists and isovist
fields") is the set of points visible from a vantage point. Its scalar summaries
— area, perimeter, occlusivity — form a *field* when computed at every vantage
across a space. This module computes that field on a regular grid and stores it
as a :class:`~deixis.field.scalar.ScalarField`.

Why a ScalarField and **not** a DEC cochain (deliberate, load-bearing):
    :mod:`deixis.field.flow` reads space as *flow* and puts values on a Discrete
    Exterior Calculus (DEC) complex — 0/1/2-cochains linked by a coboundary, with
    an intended **conservation reading** (flux balance at internal vertices). This
    module deliberately does **not** model visibility as a flow or a conserved
    flux: it defines *no* semantic correspondence to the flow module's DEC
    1-cochain API (no coboundary/divergence meaning is assigned to the isovist
    field), and carries visibility as a bare per-cell scalar with no cochain
    structure attached. (This is a modelling choice, not a theorem: one *could*
    load any scalar onto a DEC 0-cochain — we simply decline to, because the
    conservation/coboundary reading would be meaningless for visibility.)
    Concretely: **this is a plain ScalarField, not a DEC 1-cochain**, and it is
    not wired into the flow operators.

What is computed (and its honest limits — do not overclaim):
  * The isovist boundary is probed with a **finite** set of ``n_dirs`` rays.
    Everything between two adjacent rays is interpolated by a straight chord, so
    the polygon **under-approximates** a true (curved-arc-free) isovist. More
    directions ⇒ tighter approximation; the value is exact *for the sampled
    directions*, not for the continuous isovist.
  * Obstacles are **axis-aligned boxes only** (:class:`~deixis.core.types.Box`),
    2-D only. Ray/box occlusion is solved by the exact rational slab method, so
    the ray parameter ``t`` to the first blocking surface is an **exact**
    ``Fraction`` (no float). Occupancy — "is this direction blocked, and where" —
    is therefore exact.
  * **area** is the exact shoelace area of the ordered ray-hit polygon: exact
    ``Fraction`` *given the sampled rays*, but an under-approximation of the true
    isovist area (finite-direction sampling). Occupancy is exact; the *area* it
    feeds is the approximation.
  * **occlusivity** is the fraction of *sampled rays* blocked by an obstacle
    before reaching the domain boundary — ``occluded / n_dirs``, an exact
    ``Fraction``. This is loosely inspired by Benedikt's occlusivity (which
    measures the *length share* of the isovist perimeter lying on occluding
    surface) but is NOT that quantity: rays are equal-arc on the Chebyshev square,
    not equal-angle and not perimeter-length-weighted. It is exactly "the share of
    these sampled directions that hit a wall", nothing more.
  * **perimeter** is the Euclidean perimeter of the ray-hit polygon. Edge lengths
    are ``√(·)`` and hence irrational; to stay float-free they are replaced by a
    **rational lower-bound approximation** via integer square root at a fixed
    precision (error < ``1e-6`` per edge; total-perimeter error < ``n_dirs·1e-6``).
    Perimeter is therefore the one metric
    whose angle/length-dependent part is an explicit *rational approximation*,
    marked as such — unlike area (exact-for-sampled-rays) and occlusivity (exact).

Directions are chosen as **rational** vectors on the perimeter of the unit
Chebyshev square, so no trigonometry (and no float) is ever needed: walking the
square counter-clockwise from ``(1,0)`` is strictly monotincreasing in polar
angle, giving rays already ordered by angle (required by the shoelace formula)
with exactly rational components.

Builds only on the fixed core (``deixis.core.types.Box`` / ``deixis.core.ids``)
and :class:`~deixis.field.scalar.ScalarField`; edits nothing, mutates nothing.
"""
from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Iterable, List, Optional, Sequence, Tuple

from ..core.types import Box
from .scalar import ScalarField

__all__ = [
    "isovist_field",
    "isovist_at_point",
    "square_directions",
]

_Point = Tuple[Fraction, Fraction]
_Dir = Tuple[Fraction, Fraction]

# Precision denominator for the rational square-root approximation used by the
# 'perimeter' metric only. isqrt(k * P**2) / P is a lower bound on sqrt(k) with
# absolute error < 1/P. This is the ONLY place an approximation (beyond finite
# direction sampling) enters, and it is confined to perimeter edge lengths.
_SQRT_PREC = 10 ** 6


def _as_frac(x, where: str) -> Fraction:
    if isinstance(x, bool):
        raise TypeError(f"{where} must be exact (Fraction/int), got bool")
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    raise TypeError(
        f"{where} must be exact (Fraction or int), got {type(x).__name__}: {x!r} "
        "(float is rejected — the field core is float-free)"
    )


def _norm_boxes(obstacles: Iterable[Box]) -> List[Tuple[Tuple[Fraction, Fraction], Tuple[Fraction, Fraction]]]:
    """Validate + coerce obstacles to exact 2-D ``(lo, hi)`` pairs with ``lo <= hi``.

    ``Box`` itself does not enforce rationality or ``lo <= hi``, so we coerce every
    bound through :func:`_as_frac` *at the entrance* — that is what makes the
    downstream slab computation genuinely ``Fraction``-exact — and reject inverted
    or non-2-D boxes rather than silently mis-occluding.
    """
    out: List[Tuple[Tuple[Fraction, Fraction], Tuple[Fraction, Fraction]]] = []
    for b in obstacles:
        if len(b.lo) != 2 or len(b.hi) != 2:
            raise ValueError(f"isovist is 2-D only; obstacle {b.region_id!r} is not 2-D")
        lx = _as_frac(b.lo[0], f"obstacle {b.region_id!r} lo.x")
        ly = _as_frac(b.lo[1], f"obstacle {b.region_id!r} lo.y")
        hx = _as_frac(b.hi[0], f"obstacle {b.region_id!r} hi.x")
        hy = _as_frac(b.hi[1], f"obstacle {b.region_id!r} hi.y")
        if lx > hx or ly > hy:
            raise ValueError(f"obstacle {b.region_id!r} has inverted bounds (lo > hi)")
        out.append(((lx, ly), (hx, hy)))
    return out


# --------------------------------------------------------------- rational directions
def _point_on_square(s: Fraction) -> _Dir:
    """Point on the unit Chebyshev square boundary at CCW arc-parameter ``s`` in [0,8).

    Starts at ``(1,0)`` and walks counter-clockwise; the polar angle is strictly
    monotone in ``s`` so mapping equally-spaced ``s`` gives rays *ordered by
    angle* with exactly rational components. Each returned vector lies on the
    square (Chebyshev norm 1), hence is non-zero — a valid ray direction.
    """
    one = Fraction(1)
    if s < 1:                       # right edge, going up: (1, 0)->(1,1)
        return (one, s)
    if s < 3:                       # top edge, right->left: (1,1)->(-1,1)
        return (2 - s, one)
    if s < 5:                       # left edge, going down: (-1,1)->(-1,-1)
        return (-one, 4 - s)
    if s < 7:                       # bottom edge, left->right: (-1,-1)->(1,-1)
        return (s - 6, -one)
    return (one, s - 8)             # right edge, up to close: (1,-1)->(1,0)


def square_directions(n_dirs: int) -> List[_Dir]:
    """``n_dirs`` rational ray directions, ordered CCW by angle from ``(1,0)``.

    Directions sit at equal arc-spacing on the Chebyshev square (``s = 8k/n``),
    not at equal *angular* spacing — the angular non-uniformity is part of the
    finite-sampling approximation, but every component is an exact ``Fraction``
    and the ordering is monotone in true angle (so the shoelace polygon is well
    formed). ``n_dirs`` must be >= 3 (a polygon needs three vertices). Common
    exact-symmetric choices: 4 (axes), 8 (axes + diagonals), any multiple of 8.
    """
    if isinstance(n_dirs, bool) or not isinstance(n_dirs, int) or n_dirs < 3:
        raise ValueError(f"n_dirs must be an int >= 3, got {n_dirs!r}")
    return [_point_on_square(Fraction(8 * k, n_dirs)) for k in range(n_dirs)]


# --------------------------------------------------------------- exact ray/box slab
def _ray_box_enter(p: _Point, d: _Dir, lo: Sequence[Fraction], hi: Sequence[Fraction]) -> Optional[Fraction]:
    """First forward parameter ``t>0`` at which ray ``p+t·d`` enters box [lo,hi].

    Exact rational slab method. Returns:
      * ``None`` if the ray never meets the box in the forward direction;
      * ``Fraction(0)`` if ``p`` is inside/on the box (the vantage sits in the
        obstacle — the caller treats this as fully occluded);
      * else the exact entry parameter ``t_enter > 0``.
    """
    lo_t: Optional[Fraction] = None   # largest per-axis entry t
    hi_t: Optional[Fraction] = None   # smallest per-axis exit t
    for a in range(2):
        pa, da, la, ha = p[a], d[a], lo[a], hi[a]
        if da == 0:
            if pa < la or pa > ha:
                return None            # parallel to slab and outside it: no hit ever
            continue                   # inside this slab for all t: imposes no bound
        ta = (la - pa) / da
        tb = (ha - pa) / da
        t_in, t_out = (ta, tb) if da > 0 else (tb, ta)
        lo_t = t_in if lo_t is None else (t_in if t_in > lo_t else lo_t)
        hi_t = t_out if hi_t is None else (t_out if t_out < hi_t else hi_t)
    if lo_t is None or hi_t is None:   # d had a zero on every axis => not a ray
        return None
    if lo_t > hi_t or hi_t < 0:        # slabs disjoint, or box entirely behind p
        return None
    if lo_t <= 0:                      # 0 in [lo_t,hi_t] => p is inside/on the box
        return Fraction(0)
    return lo_t                        # enters the box ahead of p


def _domain_exit(p: _Point, d: _Dir, lo: Sequence[Fraction], hi: Sequence[Fraction]) -> Fraction:
    """Forward parameter at which the ray leaves the domain box (``p`` inside it)."""
    hi_t: Optional[Fraction] = None
    for a in range(2):
        pa, da, la, ha = p[a], d[a], lo[a], hi[a]
        if da == 0:
            continue
        ta = (la - pa) / da
        tb = (ha - pa) / da
        t_out = tb if da > 0 else ta
        hi_t = t_out if hi_t is None else (t_out if t_out < hi_t else hi_t)
    if hi_t is None or hi_t <= 0:
        return Fraction(0)
    return hi_t


# --------------------------------------------------------------- one vantage isovist
def _rational_sqrt(k: Fraction) -> Fraction:
    """Rational lower-bound approximation of ``sqrt(k)`` (k>=0), error < 1/_SQRT_PREC.

    Uses integer ``isqrt`` on the numerator/denominator scaled by ``_SQRT_PREC**2``
    so the result stays an exact ``Fraction`` (float-free) while approximating an
    irrational length. Used only for 'perimeter' edge lengths.
    """
    if k < 0:
        raise ValueError("sqrt of negative")
    if k == 0:
        return Fraction(0)
    a, b = k.numerator, k.denominator     # k = a/b, both > 0
    # sqrt(a/b) = sqrt(a*b)/b; approximate P*sqrt(a*b) by isqrt(a*b*P^2).
    root = isqrt(a * b * _SQRT_PREC * _SQRT_PREC)
    return Fraction(root, b * _SQRT_PREC)


def isovist_at_point(
    point: _Point,
    obstacles: Iterable[Box],
    domain_lo: _Point,
    domain_hi: _Point,
    directions: Sequence[_Dir],
    metric: str,
) -> Fraction:
    """Scalar isovist ``metric`` at one vantage ``point`` (exact where the metric is).

    Casts each ray in ``directions`` to the first blocking obstacle box or to the
    domain boundary and reduces the ordered hit-point polygon to a single scalar.

    Convention: a vantage that lies *inside or on the closed boundary of* an
    obstacle box is treated as an invalid, fully-occluded observation —
    area/perimeter are ``0`` and occlusivity is ``1`` (boxes are closed sets; a
    point on a wall face is counted as in the wall).

    ``directions`` must be a non-empty sequence of non-zero rational vectors
    ordered CCW by angle (as produced by :func:`square_directions`); the area
    shoelace and occlusivity denominator both rely on that. ``metric`` and box
    bounds are validated *before* any computation.
    """
    if metric not in ("area", "occlusivity", "perimeter"):
        raise ValueError(f"unknown metric {metric!r}; expected 'area'|'occlusivity'|'perimeter'")
    if len(directions) == 0:
        raise ValueError("directions must be a non-empty sequence")
    px, py = _as_frac(point[0], "point.x"), _as_frac(point[1], "point.y")
    return _isovist_core((px, py), _norm_boxes(obstacles), domain_lo, domain_hi, directions, metric)


def _isovist_core(
    p: _Point,
    boxes: Sequence[Tuple[Tuple[Fraction, Fraction], Tuple[Fraction, Fraction]]],
    domain_lo: _Point,
    domain_hi: _Point,
    directions: Sequence[_Dir],
    metric: str,
) -> Fraction:
    """Compute the isovist metric given *already-normalized* boxes (internal)."""
    hits: List[_Point] = []
    occluded = 0
    for d in directions:
        t_dom = _domain_exit(p, d, domain_lo, domain_hi)
        t_hit = t_dom
        blocked = False
        inside_obstacle = False
        for lo, hi in boxes:
            te = _ray_box_enter(p, d, lo, hi)
            if te is None:
                continue
            if te == 0:
                inside_obstacle = True
                break
            if te < t_hit:
                t_hit = te
                blocked = True
        if inside_obstacle:
            # Vantage sits in a solid: nothing is visible from within it.
            if metric == "occlusivity":
                return Fraction(1)
            return Fraction(0)
        if blocked:
            occluded += 1
        hits.append((p[0] + t_hit * d[0], p[1] + t_hit * d[1]))

    if metric == "occlusivity":
        return Fraction(occluded, len(directions))

    if metric == "area":
        # Shoelace area of the CCW-ordered hit polygon: EXACT for the sampled rays
        # (an under-approximation of the true isovist area).
        acc = Fraction(0)
        n = len(hits)
        for i in range(n):
            x0, y0 = hits[i]
            x1, y1 = hits[(i + 1) % n]
            acc += x0 * y1 - x1 * y0
        return abs(acc) / 2

    if metric == "perimeter":
        # Euclidean perimeter; edge lengths are a rational approximation of sqrt
        # (error < 1/_SQRT_PREC per edge), marked approximate in the docstring.
        acc = Fraction(0)
        n = len(hits)
        for i in range(n):
            x0, y0 = hits[i]
            x1, y1 = hits[(i + 1) % n]
            dx, dy = x1 - x0, y1 - y0
            acc += _rational_sqrt(dx * dx + dy * dy)
        return acc

    raise ValueError(f"unknown metric {metric!r}; expected 'area'|'occlusivity'|'perimeter'")


# --------------------------------------------------------------- the field
def isovist_field(
    sample_points: Optional[Iterable[_Point]],
    obstacles: Iterable[Box],
    grid_shape: Tuple[int, ...],
    origin: Tuple[Fraction, ...],
    spacing: Fraction,
    metric: str = "area",
    n_dirs: int = 16,
) -> ScalarField:
    """Visibility (isovist) field over a 2-D grid, stored as a :class:`ScalarField`.

    For each grid cell the isovist ``metric`` is computed at the cell *center*
    (``origin + index·spacing``) against the axis-aligned ``obstacles`` and stored
    as that cell's scalar value. Rays are limited to the domain bounding box
    (cell centers padded by ``spacing/2`` on each side), so the space is treated
    as finite-extent — an honest limit, not an infinite plane.

    ``sample_points``: if ``None`` (default), every cell center is evaluated. If a
    list of physical points is given, each is mapped to the grid cell whose
    *center is nearest* (exact rational round-half-up per axis, consistent with
    the ``spacing/2`` cell extent); only those cells are evaluated (the isovist is
    written to the cell center, not the raw point) and all other cells are ``0``.
    Points outside the grid are ignored. This keeps the result a well-formed
    ScalarField over the whole grid while letting a caller probe a sparse subset.

    ``metric``: ``'area'`` (exact-for-sampled-rays shoelace area — the default),
    ``'occlusivity'`` (exact occluded-ray fraction), or ``'perimeter'`` (rational
    approximation of Euclidean perimeter). See the module docstring for the exact
    vs. approximate breakdown.

    **This is a plain ScalarField, not a DEC 1-cochain**: this module assigns no
    conservation/flux reading to visibility and defines no correspondence to the
    flow module's DEC operators (see the module docstring).

    2-D only (``grid_shape`` must have rank 2); obstacles must be 2-D boxes.
    """
    if len(grid_shape) != 2:
        raise ValueError(f"isovist_field is 2-D only; got grid rank {len(grid_shape)}")
    if metric not in ("area", "occlusivity", "perimeter"):
        raise ValueError(f"unknown metric {metric!r}; expected 'area'|'occlusivity'|'perimeter'")
    # Validate the shape to ScalarField's own contract *before* using it (do not
    # let bool / float / non-positive extents be silently coerced by int()).
    for n in grid_shape:
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError(f"grid_shape must be positive ints, got {grid_shape!r}")
    nx, ny = grid_shape[0], grid_shape[1]
    ox, oy = _as_frac(origin[0], "origin.x"), _as_frac(origin[1], "origin.y")
    sp = _as_frac(spacing, "spacing")
    if sp <= 0:
        raise ValueError(f"spacing must be positive, got {sp!r}")

    boxes = _norm_boxes(obstacles)
    directions = square_directions(n_dirs)

    half = sp / 2
    dom_lo: _Point = (ox - half, oy - half)
    dom_hi: _Point = (ox + Fraction(nx - 1) * sp + half, oy + Fraction(ny - 1) * sp + half)

    def cell_center(ix: int, iy: int) -> _Point:
        return (ox + Fraction(ix) * sp, oy + Fraction(iy) * sp)

    def flat(ix: int, iy: int) -> int:
        return ix * ny + iy  # row-major, matching ScalarField's flat indexing

    # Which flat cells to evaluate (others stay 0).
    if sample_points is None:
        wanted = None
    else:
        wanted = set()
        for pt in sample_points:
            px = _as_frac(pt[0], "sample_point.x")
            py = _as_frac(pt[1], "sample_point.y")
            # nearest cell center: cell centers sit at origin + i*spacing and each
            # owns the half-open interval (i-1/2, i+1/2]*spacing, so the nearest
            # index is round((coord-origin)/spacing) = floor(q + 1/2) (exact).
            ix = int(((px - ox) / sp + Fraction(1, 2)) // 1)
            iy = int(((py - oy) / sp + Fraction(1, 2)) // 1)
            if 0 <= ix < nx and 0 <= iy < ny:
                wanted.add(flat(ix, iy))

    values: List[Fraction] = []
    for ix in range(nx):
        for iy in range(ny):
            f = flat(ix, iy)
            if wanted is not None and f not in wanted:
                values.append(Fraction(0))
                continue
            values.append(
                _isovist_core(cell_center(ix, iy), boxes, dom_lo, dom_hi, directions, metric)
            )

    return ScalarField(
        grid_shape=(nx, ny),
        values=tuple(values),
        origin=(ox, oy),
        spacing=sp,
    )
