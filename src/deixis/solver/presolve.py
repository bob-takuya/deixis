"""Equality-elimination + boundary-propagation presolver (equivalence transforms only).

This module runs *before* :func:`deixis.solver.geom_solver.solve_boxes` and shrinks
the problem it hands to Z3, without ever changing the solution set. Two reductions:

1. **Equality elimination.** An RCC-8 ``EQ`` relation (the definite ``bit('EQ')`` mask)
   between two regions means their boxes must be identical, so the two coordinate
   variable groups are one. We union-find all ``EQ``-related regions and collapse each
   equivalence class to a single representative region. Constraints, witnesses and
   overlays are rewritten onto the representative ids. This is a pure *variable
   substitution*: any model of the reduced spec extends to a model of the original by
   copying the representative box onto its EQ-partners, and vice-versa, so SAT/UNSAT is
   preserved exactly.

2. **Boundary propagation (bounds tightening).** For a *subset* relation ``A sub B``
   (any mask whose every base relation is one of ``TPP / NTPP / EQ``, i.e. ``A``'s box
   is contained in ``B``'s box) the confinement domain of ``A`` can only be inside the
   confinement domain of ``B``. We therefore intersect ``A``'s per-axis domain with
   ``B``'s. Because ``box(A) ⊆ box(B) ⊆ domain(B)`` already holds in every solution,
   intersecting the domains removes no solutions — it only makes an *implied* bound
   explicit. Iterated to a fixpoint (domains only shrink, so it converges).

   The tightened per-region domains are reported (``PresolveReport.bounds``) but are
   **advisory**: :func:`~deixis.solver.geom_solver.solve_boxes` currently takes a single
   scalar ``(lo, hi)`` box for all regions, so it does not consume per-region domains. A
   caller who wants the tightening to actually shrink Z3's search must apply the reported
   bounds itself. Only reduction (1), EQ variable-collapse, shrinks the Z3 problem through
   ``reduced_spec`` directly (fewer region variables).

Early contradiction detection (sound, never a false positive):

* an ``EMPTY`` mask (``rcc8_mask == 0``) constraint;
* a constraint that, after EQ-collapse, relates a region to *itself* with a mask that
  excludes ``EQ`` (a region cannot be non-equal to itself);
* two constraints on the same collapsed pair whose masks (folded through ``converse``)
  intersect to ``EMPTY``;
* a domain that propagation empties (``lo[d] >= hi[d]`` on some axis — no room for a
  positive-size box).

Honest scope (over-claim suppression)
-------------------------------------
* **Equivalence only / solution-set unchanged.** Every transform preserves the solution
  set *relative to the supplied bounds*; presolve never adds or drops a satisfying model.
* **Not complete.** presolve is a fast local pre-pass, **not** a decision procedure. It
  finds *some* contradictions, not all; a spec it calls consistent may still be UNSAT.
  Z3 (``solve_boxes``) remains the final authority on realizability.
* **Local propagation is limited to equality and containment bounds.** This is a
  deliberately narrow, DeltaBlue-style *local* propagation restricted to the two rules
  above; it does **not** do full path-consistency or arc-consistency.
* **Disjunctions and cycles are delegated to Z3.** A disjunctive mask (more than one
  base relation, other than a pure subset disjunction) drives no propagation, and no
  attempt is made to *resolve* containment cycles — they are left for the solver.
* geom_solver / pipeline are untouched; this is an opt-in standalone module whose output
  (``reduced_spec`` + report) you may feed to ``solve_boxes`` yourself.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Mapping, Optional

from ..core import rcc8
from ..core.types import (
    IncidenceWitness,
    OverlayAtom,
    Region,
    RelationConstraint,
    RelSpec,
)

_DOMAIN_NDIM = {"aabb_2d": 2, "aabb_3d": 3}

# masks whose every base relation entails ``src`` box ⊆ ``dst`` box.
_SUBSET_FWD = rcc8.mask("TPP", "NTPP", "EQ")     # src ⊆ dst
_SUBSET_REV = rcc8.mask("TPPi", "NTPPi", "EQ")   # dst ⊆ src
_EQ_BIT = rcc8.bit("EQ")


# ------------------------------------------------------------------ report types

@dataclass(frozen=True)
class RegionBounds:
    """Confinement domain of one region's AABB (mirrors :class:`~deixis.core.types.Box`).

    Semantics: ``box.lo[d] >= lo[d]`` and ``box.hi[d] <= hi[d]`` per axis. ``None`` on a
    side means unbounded there. All present values are exact ``Fraction``.
    """

    region_id: str
    lo: tuple[Optional[Fraction], ...]
    hi: tuple[Optional[Fraction], ...]


@dataclass(frozen=True)
class PresolveReport:
    """What the presolver did — a transparent, replayable ledger.

    ``consistent`` is ``False`` only when an *early* contradiction was proven; ``True``
    never claims global satisfiability (Z3 is final — see module docstring)."""

    consistent: bool = True
    contradiction: str = ""
    substitutions: tuple[tuple[str, str], ...] = ()   # (region_id -> representative_id)
    eliminated_regions: tuple[str, ...] = ()          # regions removed by EQ-collapse
    dropped_constraints: tuple[str, ...] = ()         # constraint ids dropped as trivially-true
    bounds: tuple[RegionBounds, ...] = ()             # tightened per (representative) region
    propagation_passes: int = 0
    scope_note: str = (
        "equivalence-preserving only; solution set unchanged relative to given bounds; "
        "incomplete (Z3 is the final authority on realizability); local propagation "
        "limited to EQ-substitution and subset-domain tightening; disjunctions and "
        "containment cycles are delegated to Z3."
    )


# ------------------------------------------------------------------ union-find

class _UnionFind:
    def __init__(self, items) -> None:
        self._parent = {x: x for x in items}

    def add(self, x) -> None:
        self._parent.setdefault(x, x)

    def find(self, x):
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


# ------------------------------------------------------------------ bounds helpers

def _maxb(a: Optional[Fraction], b: Optional[Fraction]) -> Optional[Fraction]:
    """Tighter (larger) lower bound; ``None`` = unbounded below."""
    if a is None:
        return b
    if b is None:
        return a
    return a if a >= b else b


def _minb(a: Optional[Fraction], b: Optional[Fraction]) -> Optional[Fraction]:
    """Tighter (smaller) upper bound; ``None`` = unbounded above."""
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


def _normalize_bounds(bounds, region_ids, ndim: int) -> dict[str, list]:
    """Return ``{region_id: [lo_list, hi_list]}`` of Optional[Fraction], length ndim.

    ``bounds`` may be:
      * ``None`` — no domain info (no tightening);
      * a scalar ``(lo, hi)`` pair — applied to every region on every axis;
      * a mapping ``region_id -> (lo_corner, hi_corner)`` of per-axis tuples
        (entries may be ``None`` for an unbounded side).
    """
    if bounds is None:
        return {}

    if isinstance(bounds, Mapping):
        out: dict[str, list] = {}
        for rid, val in bounds.items():
            lo_corner, hi_corner = val
            lo = [None if x is None else Fraction(x) for x in lo_corner]
            hi = [None if x is None else Fraction(x) for x in hi_corner]
            if len(lo) != ndim or len(hi) != ndim:
                raise ValueError(
                    f"bounds for {rid!r} have {len(lo)}/{len(hi)} axes, expected {ndim}"
                )
            out[rid] = [lo, hi]
        return out

    lo_s, hi_s = bounds
    lo_f = None if lo_s is None else Fraction(lo_s)
    hi_f = None if hi_s is None else Fraction(hi_s)
    return {rid: [[lo_f] * ndim, [hi_f] * ndim] for rid in region_ids}


def _intersect_into(dst: list, src: list, ndim: int) -> bool:
    """Intersect domain ``src`` into ``dst`` (in place). Return True if ``dst`` changed."""
    changed = False
    for d in range(ndim):
        nlo = _maxb(dst[0][d], src[0][d])
        nhi = _minb(dst[1][d], src[1][d])
        if nlo != dst[0][d]:
            dst[0][d] = nlo
            changed = True
        if nhi != dst[1][d]:
            dst[1][d] = nhi
            changed = True
    return changed


def _domain_empty(dom: list, ndim: int) -> bool:
    """A domain with no room for a positive-size box on some axis (lo >= hi)."""
    for d in range(ndim):
        lo, hi = dom[0][d], dom[1][d]
        if lo is not None and hi is not None and lo >= hi:
            return True
    return False


# ------------------------------------------------------------------ public API

def presolve(
    spec: RelSpec,
    bounds=None,
    *,
    domain: str = "aabb_2d",
) -> tuple[RelSpec, PresolveReport]:
    """Reduce ``spec`` by equality elimination and boundary propagation.

    Parameters
    ----------
    spec : RelSpec
        The abstract relational spec (regions + RCC-8 constraints [+ witnesses/overlays]).
    bounds : optional
        Per-region confinement domains that tightening starts from. Either a scalar
        ``(lo, hi)`` pair (applied to every region on every axis), a mapping
        ``region_id -> (lo_corner, hi_corner)`` of per-axis tuples, or ``None`` to skip
        tightening entirely (EQ elimination + mask-level contradiction detection still run).
    domain : str
        ``'aabb_2d'`` or ``'aabb_3d'`` — fixes the number of axes.

    Returns
    -------
    (reduced_spec, PresolveReport)
        ``reduced_spec`` is a NEW :class:`RelSpec` with each EQ class collapsed to one
        representative region and all id references rewritten; trivially-true EQ
        self-relations are dropped. Its solution set (given the same ``bounds``) equals
        the original's. The report carries the substitution map, tightened bounds, dropped
        constraints and, if a contradiction was proven, ``consistent == False``.

    Notes
    -----
    Equivalence-preserving only; **not** complete. See the module docstring for the full
    honest-scope statement (Z3 stays the final authority; disjunctions/cycles delegated).
    """
    if domain not in _DOMAIN_NDIM:
        raise ValueError(
            f"unsupported domain {domain!r}; use one of {sorted(_DOMAIN_NDIM)}"
        )
    ndim = _DOMAIN_NDIM[domain]

    declared_ids = [r.id for r in spec.regions]
    order = {rid: i for i, rid in enumerate(declared_ids)}

    # ---- 1. equality elimination (union-find over definite EQ relations) ----
    uf = _UnionFind(declared_ids)
    for c in spec.constraints:
        uf.add(c.src)
        uf.add(c.dst)
        if c.rcc8_mask == _EQ_BIT:  # definite equality only (disjunctive EQ -> Z3)
            uf.union(c.src, c.dst)

    # representative of each class = earliest-declared member (a real Region), else the id.
    classes: dict[str, list[str]] = {}
    for rid in declared_ids:
        classes.setdefault(uf.find(rid), []).append(rid)
    remap: dict[str, str] = {}
    reps: dict[str, str] = {}  # root -> representative id
    for root, members in classes.items():
        rep = min(members, key=lambda m: order.get(m, len(order)))
        reps[root] = rep
        for m in members:
            remap[m] = rep

    def canon(rid: str) -> str:
        root = uf.find(rid)
        return reps.get(root, remap.get(rid, rid))

    substitutions = tuple(
        (rid, canon(rid)) for rid in declared_ids if canon(rid) != rid
    )
    eliminated = tuple(rid for rid, _ in substitutions)

    consistent = True
    contradiction = ""

    # ---- rewrite constraints onto representatives; detect self-pair contradictions ----
    reduced_constraints: list[RelationConstraint] = []
    dropped: list[str] = []
    for c in spec.constraints:
        cid = c.id or f"{c.src}->{c.dst}"
        if c.rcc8_mask == rcc8.EMPTY:
            consistent = False
            contradiction = contradiction or f"constraint {cid!r} has an EMPTY (impossible) mask"
        ns, nd = canon(c.src), canon(c.dst)
        if ns == nd and (c.rcc8_mask & _EQ_BIT):
            # a self-pair after EQ-collapse whose mask permits EQ: ``rep EQ rep`` holds
            # trivially, so the constraint is satisfied by construction -> drop it.
            dropped.append(cid)
            continue
        if ns == nd:
            # a region forced into a non-EQ relation with itself: an early contradiction.
            # We KEEP the rewritten self-constraint (rather than dropping it) so the reduced
            # spec, handed to solve_boxes, reproduces the SAME UNSAT -- the solution set stays
            # unchanged even in the contradictory case (solve_boxes encodes ``rep r rep`` as
            # unsatisfiable for every non-EQ base relation r).
            consistent = False
            contradiction = contradiction or (
                f"constraint {cid!r} forces region {ns!r} into a non-EQ relation with "
                f"itself (mask={rcc8.names(c.rcc8_mask)})"
            )
        reduced_constraints.append(replace(c, src=ns, dst=nd))

    # ---- fold parallel constraints per collapsed pair to catch empty intersections ----
    pair_mask: dict[tuple[str, str], int] = {}
    for c in reduced_constraints:
        if c.src <= c.dst:
            key, m = (c.src, c.dst), c.rcc8_mask
        else:
            key, m = (c.dst, c.src), rcc8.converse(c.rcc8_mask)
        pair_mask[key] = rcc8.intersect(pair_mask.get(key, rcc8.UNIVERSAL), m)
    for (a, b), m in pair_mask.items():
        if m == rcc8.EMPTY:
            consistent = False
            contradiction = contradiction or (
                f"constraints between {a!r} and {b!r} intersect to an EMPTY relation"
            )

    # ---- 2. boundary propagation over collapsed subset edges ----
    rep_ids = [reps[uf.find(rid)] for rid in declared_ids]
    rep_ids = list(dict.fromkeys(rep_ids))  # dedup, keep declaration order

    norm = _normalize_bounds(bounds, declared_ids, ndim)
    # build each representative's starting domain = intersection of its members' inputs.
    domains: dict[str, list] = {}
    if norm:
        for rid in declared_ids:
            rep = canon(rid)
            src = norm.get(rid)
            if src is None:
                continue
            src_copy = [list(src[0]), list(src[1])]
            if rep not in domains:
                domains[rep] = src_copy
            else:
                _intersect_into(domains[rep], src_copy, ndim)

    subset_edges: list[tuple[str, str]] = []  # (child, parent): child box ⊆ parent box
    for c in reduced_constraints:
        m = c.rcc8_mask
        if m == rcc8.EMPTY:
            continue
        if (m & ~_SUBSET_FWD) == 0:      # every base relation => src ⊆ dst
            subset_edges.append((c.src, c.dst))
        elif (m & ~_SUBSET_REV) == 0:    # every base relation => dst ⊆ src
            subset_edges.append((c.dst, c.src))

    passes = 0
    if domains and subset_edges:
        max_passes = len(subset_edges) + 1
        while passes < max_passes:
            passes += 1
            changed = False
            for child, parent in subset_edges:
                pdom = domains.get(parent)
                if pdom is None:
                    continue
                cdom = domains.get(child)
                if cdom is None:
                    cdom = [list(pdom[0]), list(pdom[1])]
                    domains[child] = cdom
                    changed = True
                    continue
                if _intersect_into(cdom, pdom, ndim):
                    changed = True
            if not changed:
                break

    # emptied domain => contradiction
    for rep, dom in domains.items():
        if _domain_empty(dom, ndim):
            consistent = False
            contradiction = contradiction or (
                f"region {rep!r} domain is empty after propagation (no room for a box)"
            )

    bounds_report = tuple(
        RegionBounds(
            region_id=rep,
            lo=tuple(domains[rep][0]),
            hi=tuple(domains[rep][1]),
        )
        for rep in rep_ids
        if rep in domains
    )

    # ---- build reduced spec (representatives only, ids rewritten everywhere) ----
    rep_region = {r.id: r for r in spec.regions if r.id in set(rep_ids)}
    reduced_regions = tuple(rep_region[rid] for rid in rep_ids if rid in rep_region)

    reduced_witnesses = tuple(
        replace(
            w,
            incident_regions=tuple(dict.fromkeys(canon(r) for r in w.incident_regions)),
        )
        for w in spec.witnesses
    )
    reduced_overlays = tuple(
        replace(o, members=frozenset(canon(m) for m in o.members))
        for o in spec.overlays
    )

    reduced_spec = replace(
        spec,
        regions=reduced_regions,
        constraints=tuple(reduced_constraints),
        witnesses=reduced_witnesses,
        overlays=reduced_overlays,
    )

    report = PresolveReport(
        consistent=consistent,
        contradiction=contradiction,
        substitutions=substitutions,
        eliminated_regions=eliminated,
        dropped_constraints=tuple(dropped),
        bounds=bounds_report,
        propagation_passes=passes,
    )
    return reduced_spec, report


__all__ = ["presolve", "PresolveReport", "RegionBounds"]
