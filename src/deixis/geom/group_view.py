"""Transformation-group *views*: read-only projection of a realized geometry onto
the invariants of a chosen transformation group, with an honest ``LossReport``.

What this module is
-------------------
Given a :class:`~deixis.core.types.Realization` (exact-rational axis-aligned boxes),
:func:`project` observes **only** the quantities that are invariant under a chosen
:class:`TransformGroup`, and returns a :class:`LossReport` that *discloses* which
information was deliberately not observed at that coarseness:

* ``EUCLIDEAN``   — absolute coordinates & metric (raw boxes, side lengths, squared
  centroid distances).  Nothing is dropped.
* ``SIMILARITY``  — ratios only (per-region aspect ratios, pairwise size ratios).
  Absolute coordinates and absolute scale (metric) are dropped.
* ``AFFINE``      — parallelism & ordinal (betweenness / nesting) structure; **no
  ratios**.  Absolute coordinates, metric, ratios and angles are dropped.
* ``PROJECTIVE``  — incidence, collinearity and cross-ratios of *genuinely collinear*
  vertex sets.  Parallelism (in addition to the affine drops) is dropped.
* ``HOMEO``       — contact relations only (the RCC-8 topology).  Orientation and
  linking (in addition to the projective drops) are dropped.

Three honesty commitments (do not relax these)
----------------------------------------------
1. **Read-only.**  :func:`project` observes; it never mutates the realization and
   there is deliberately no ``put``/inverse.  The input :class:`Realization` is
   returned unchanged by identity.

2. **Forgetting is not reverse-grounding.**  Projecting to a coarser group *forgets*
   (hides) finer information; it does **not** reconstruct, invert, or re-ground it.
   A :class:`LossReport` only *names the categories* that were dropped — it does not
   promise they can be recovered.  There is no coarse→fine map here.

3. **No over-claim of algebraic structure.**

   * ``canonical_key`` (see :func:`canonical_key`) is a coordinate on the **orbit
     quotient** ``X/G`` — the space of realizations modulo the group action — **not**
     a coset of a homogeneous space ``G/H``.  It is a *sound but incomplete* orbit
     separator: equal orbits give equal keys, but equal keys do **not** prove the
     same orbit unless the observed invariant set happens to be complete (it is not
     claimed to be).
   * Calling the family of per-group views a "lattice / bundle (束)" is a **metaphor**.
     :class:`TransformGroup` is a genuine finite poset (a total chain, with real
     ``leq``/``meet``/``join``), but the mathematical *fibre-bundle* over that poset
     — total space, projection, structure group — is **not constructed** here. Only
     the poset of groups and the per-group invariant read-outs exist.

Exactness: every observed quantity is an exact :class:`fractions.Fraction` (or a
finite combination of them). Euclidean distance is reported *squared* because a true
distance is generally irrational; squaring keeps it rational.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from itertools import combinations, product
from typing import Any

from ..core.types import Realization
from ..verify import reverse
from ..core import rcc8


# --------------------------------------------------------------------- the poset

class TransformGroup(Enum):
    """The finite poset of transformation groups, ordered by *subgroup inclusion*::

        EUCLIDEAN < SIMILARITY < AFFINE < PROJECTIVE < HOMEO

    This is a **total chain**, so ``meet`` (greatest lower bound = intersection of
    subgroups) is just the lower of two elements and ``join`` (least upper bound) the
    higher.  A *smaller* group has *fewer* transformations and therefore *more*
    invariants, so :func:`project` observes the most at ``EUCLIDEAN`` and the least
    (contact only) at ``HOMEO``.

    The ``value`` is the poset rank (0 = finest / smallest group).
    """
    EUCLIDEAN = 0
    SIMILARITY = 1
    AFFINE = 2
    PROJECTIVE = 3
    HOMEO = 4

    @property
    def rank(self) -> int:
        return self.value

    def leq(self, other: "TransformGroup") -> bool:
        """``self <= other`` in the poset (``self`` is a subgroup of ``other``)."""
        return self.value <= other.value

    def meet(self, other: "TransformGroup") -> "TransformGroup":
        """Greatest lower bound = subgroup intersection (the finer of the two)."""
        return self if self.value <= other.value else other

    def join(self, other: "TransformGroup") -> "TransformGroup":
        """Least upper bound (the coarser of the two)."""
        return self if self.value >= other.value else other


def groups_chain() -> tuple[TransformGroup, ...]:
    """The groups in ascending poset order (finest → coarsest)."""
    return (
        TransformGroup.EUCLIDEAN,
        TransformGroup.SIMILARITY,
        TransformGroup.AFFINE,
        TransformGroup.PROJECTIVE,
        TransformGroup.HOMEO,
    )


# --------------------------------------------------------------- the loss report

# Categories of information, finest → coarsest, retained by each group. A category is
# "dropped" at group g iff it is retained by some strictly finer group but not by g.
_RETAINED: dict[TransformGroup, tuple[str, ...]] = {
    TransformGroup.EUCLIDEAN: (
        "absolute_coordinates", "metric", "ratios", "angles",
        "parallelism", "collinearity", "incidence", "cross_ratio",
        "orientation", "linking", "contact",
    ),
    TransformGroup.SIMILARITY: (
        "ratios", "angles",
        "parallelism", "collinearity", "incidence", "cross_ratio",
        "orientation", "linking", "contact",
    ),
    TransformGroup.AFFINE: (
        "parallelism", "betweenness",
        "collinearity", "incidence", "cross_ratio",
        "orientation", "linking", "contact",
    ),
    TransformGroup.PROJECTIVE: (
        "collinearity", "incidence", "cross_ratio",
        "orientation", "linking", "contact",
    ),
    TransformGroup.HOMEO: (
        "contact",
    ),
}


@dataclass(frozen=True)
class LossReport:
    """Read-only disclosure of what a group projection did *not* observe.

    ``dropped`` names the information categories a strictly finer group would have
    retained but this group does not (e.g. ``absolute_coordinates``, ``metric``,
    ``orientation``, ``linking``).  It is a *disclosure*, not an inverse: nothing in
    this object reconstructs the dropped information (forgetting != reverse-grounding).
    """
    group: TransformGroup
    retained: tuple[str, ...]
    dropped: tuple[str, ...]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group.name,
            "rank": self.group.rank,
            "retained": list(self.retained),
            "dropped": list(self.dropped),
            "notes": self.notes,
        }


def _loss_for(group: TransformGroup) -> LossReport:
    retained = _RETAINED[group]
    retained_set = set(retained)
    # Everything any finer group keeps but this group does not = dropped.
    finer_keep: set[str] = set()
    for g in groups_chain():
        if g.rank < group.rank:
            finer_keep |= set(_RETAINED[g])
    dropped_set = finer_keep - retained_set
    # Preserve a stable, coarsest-first ordering for the dropped list.
    order = _RETAINED[TransformGroup.EUCLIDEAN]
    dropped = tuple(c for c in order if c in dropped_set)
    notes = (
        f"{group.name} view: observed only {group.name}-invariants. Dropped categories "
        f"are disclosed, NOT reconstructed (forgetting is not reverse-grounding); "
        f"this projection is read-only and has no inverse."
    )
    return LossReport(group=group, retained=retained, dropped=dropped, notes=notes)


# ----------------------------------------------------------------- box geometry

def _side_lengths(box) -> tuple[Fraction, ...]:
    return tuple(h - l for l, h in zip(box.lo, box.hi))


def _centroid(box) -> tuple[Fraction, ...]:
    return tuple((l + h) / 2 for l, h in zip(box.lo, box.hi))


def _vertices(box) -> tuple[tuple[Fraction, ...], ...]:
    """All 2^ndim corner points of an axis-aligned box (exact)."""
    axes = tuple((box.lo[d], box.hi[d]) for d in range(len(box.lo)))
    return tuple(product(*axes))


def _cross_ratio(t0: Fraction, t1: Fraction, t2: Fraction, t3: Fraction) -> Fraction:
    """Cross-ratio (t0,t1;t2,t3) of four distinct collinear parameter values."""
    return ((t2 - t0) * (t3 - t1)) / ((t2 - t1) * (t3 - t0))


# ----------------------------------------------------------- per-group observers

def _observe_euclidean(realization: Realization) -> dict[str, Any]:
    boxes = realization.boxes
    coordinates = {b.region_id: (tuple(b.lo), tuple(b.hi)) for b in boxes}
    centroids = {b.region_id: _centroid(b) for b in boxes}
    side_lengths = {b.region_id: _side_lengths(b) for b in boxes}
    squared_distances: dict[tuple[str, str], Fraction] = {}
    for a, b in combinations(boxes, 2):
        ca, cb = centroids[a.region_id], centroids[b.region_id]
        squared_distances[(a.region_id, b.region_id)] = sum(
            (x - y) * (x - y) for x, y in zip(ca, cb)
        )
    return {
        "group": TransformGroup.EUCLIDEAN.name,
        "coordinates": coordinates,
        "centroids": centroids,
        "side_lengths": side_lengths,
        "squared_centroid_distances": squared_distances,
    }


def _observe_similarity(realization: Realization) -> dict[str, Any]:
    boxes = realization.boxes
    # Per-region aspect ratios: side[d]/side[0] (scale-invariant shape).
    aspect_ratios: dict[str, tuple[Fraction, ...]] = {}
    for b in boxes:
        sides = _side_lengths(b)
        base = sides[0]
        aspect_ratios[b.region_id] = tuple(s / base for s in sides)
    # Pairwise size ratio along axis 0 (scale-invariant; absolute scale dropped).
    size_ratios: dict[tuple[str, str], Fraction] = {}
    for a, b in combinations(boxes, 2):
        size_ratios[(a.region_id, b.region_id)] = (
            _side_lengths(a)[0] / _side_lengths(b)[0]
        )
    return {
        "group": TransformGroup.SIMILARITY.name,
        "aspect_ratios": aspect_ratios,
        "pairwise_size_ratios": size_ratios,
    }


def _axis_ordinal(a, b, d: int) -> str:
    """Ordinal (affine-invariant) relation of two intervals on axis ``d``.

    Nesting *direction* is affine-invariant (inclusion is preserved), so it is kept;
    left/right order is not (a reflection swaps it), so ``separated`` is symmetric.
    No metric or ratio is used.
    """
    alo, ahi, blo, bhi = a.lo[d], a.hi[d], b.lo[d], b.hi[d]
    if ahi < blo or bhi < alo:
        return "separated"
    if ahi == blo or bhi == alo:
        return "meets"
    # interiors overlap
    a_in_b = blo <= alo and ahi <= bhi
    b_in_a = alo <= blo and bhi <= ahi
    if a_in_b and b_in_a:
        return "equal"
    if a_in_b:
        return "a_inside_b"
    if b_in_a:
        return "b_inside_a"
    return "overlaps"


def _observe_affine(realization: Realization) -> dict[str, Any]:
    boxes = realization.boxes
    ndim = min((len(b.lo) for b in boxes), default=0)
    # Every box is axis-aligned, so all boxes' edges along axis d are mutually
    # parallel: report the parallel class per axis (parallelism is affine-invariant).
    parallel_edge_classes = {
        d: tuple(b.region_id for b in boxes) for d in range(ndim)
    }
    axis_ordinal: dict[tuple[str, str], tuple[str, ...]] = {}
    for a, b in combinations(boxes, 2):
        axis_ordinal[(a.region_id, b.region_id)] = tuple(
            _axis_ordinal(a, b, d) for d in range(ndim)
        )
    return {
        "group": TransformGroup.AFFINE.name,
        "parallel_edge_classes": parallel_edge_classes,
        "axis_ordinal": axis_ordinal,
    }


def _observe_projective(realization: Realization) -> dict[str, Any]:
    boxes = realization.boxes
    ndim = min((len(b.lo) for b in boxes), default=0)

    # Incidence: corner points shared by two or more regions.
    point_regions: dict[tuple[Fraction, ...], set[str]] = {}
    for b in boxes:
        for v in _vertices(b):
            point_regions.setdefault(v, set()).add(b.region_id)
    coincident_vertices = [
        (pt, tuple(sorted(rs))) for pt, rs in point_regions.items() if len(rs) >= 2
    ]

    # Genuinely collinear vertex sets: vertices sharing all-but-one coordinate lie on
    # a common axis-parallel line, so their cross-ratio is a real projective invariant.
    collinear_sets: list[tuple[int, tuple[Fraction, ...], tuple[Fraction, ...]]] = []
    cross_ratios: list[tuple[int, tuple[Fraction, ...], tuple[Fraction, ...]]] = []
    for d in range(ndim):
        line_groups: dict[tuple[Fraction, ...], set[Fraction]] = {}
        for b in boxes:
            for v in _vertices(b):
                perp = v[:d] + v[d + 1:]
                line_groups.setdefault(perp, set()).add(v[d])
        for perp, coords in line_groups.items():
            ordered = tuple(sorted(coords))
            if len(ordered) >= 3:
                collinear_sets.append((d, perp, ordered))
            if len(ordered) >= 4:
                crs = tuple(
                    _cross_ratio(ordered[i], ordered[i + 1], ordered[i + 2], ordered[i + 3])
                    for i in range(len(ordered) - 3)
                )
                cross_ratios.append((d, perp, crs))

    return {
        "group": TransformGroup.PROJECTIVE.name,
        "coincident_vertices": coincident_vertices,
        "collinear_vertex_sets": collinear_sets,
        "cross_ratios": cross_ratios,
        # Honest scope note: for axis-aligned boxes only genuinely-collinear vertex
        # sets yield cross-ratios; the projective read-out is partial, not complete.
        "partial": True,
    }


def _observe_homeo(realization: Realization) -> dict[str, Any]:
    relations = reverse.extract_relations(realization)
    rcc8_names = {pair: tuple(rcc8.names(m)) for pair, m in relations.items()}
    return {
        "group": TransformGroup.HOMEO.name,
        "rcc8": relations,
        "rcc8_names": rcc8_names,
    }


_OBSERVERS = {
    TransformGroup.EUCLIDEAN: _observe_euclidean,
    TransformGroup.SIMILARITY: _observe_similarity,
    TransformGroup.AFFINE: _observe_affine,
    TransformGroup.PROJECTIVE: _observe_projective,
    TransformGroup.HOMEO: _observe_homeo,
}


# ------------------------------------------------------------------- public API

def project(
    realization: Realization, group: TransformGroup
) -> tuple[dict[str, Any], LossReport]:
    """Observe only ``group``'s invariants of ``realization`` (READ-ONLY).

    Returns ``(observed, loss)`` where ``observed`` is a dict of the group-invariant
    quantities (exact :class:`~fractions.Fraction`) and ``loss`` is a
    :class:`LossReport` disclosing the categories not observed at this coarseness.

    This is a pure observation: ``realization`` is **not** modified and there is no
    inverse (``put``).  Coarser groups forget more; forgetting is *not* the reverse of
    grounding — the dropped categories are disclosed, never reconstructed.
    """
    if not isinstance(group, TransformGroup):
        raise TypeError(f"group must be a TransformGroup, got {type(group).__name__}")
    observed = _OBSERVERS[group](realization)
    return observed, _loss_for(group)


def _freeze(obj: Any) -> Any:
    """Recursively convert an ``observed`` structure into a hashable canonical form.

    Fractions become ``(numerator, denominator)`` pairs; dicts become sorted tuples of
    key/value pairs; sets and lists become sorted/ordered tuples.
    """
    if isinstance(obj, Fraction):
        return (obj.numerator, obj.denominator)
    if isinstance(obj, dict):
        return tuple(sorted(
            (_freeze(k), _freeze(v)) for k, v in obj.items()
        ))
    if isinstance(obj, (set, frozenset)):
        return tuple(sorted(_freeze(x) for x in obj))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    return obj


def canonical_key(realization: Realization, group: TransformGroup) -> Any:
    """A hashable representative of the ``group``-orbit of ``realization``.

    The key is built solely from ``group``-invariant observations, so it is constant
    along the group action: it is a coordinate on the **orbit quotient X/G**, NOT a
    coset of a homogeneous space ``G/H``.  It is a *sound but incomplete* orbit
    separator — equal orbits yield equal keys, but equal keys do not by themselves
    prove the same orbit (the observed invariant set is not claimed to be complete).
    """
    observed, _ = project(realization, group)
    return (group.name, _freeze(observed))
