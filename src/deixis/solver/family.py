"""M6: solution-family generation and comparison over the immutable RelSpec.

A *family* is what you get when one un-grounded relational spec ``S`` is realized
under several different :class:`~deixis.core.types.GroundingDecision` scenarios: the
invariant/delegated skeleton is shared, only the grounded freedom differs. This module
lets you

* :func:`solve_family` — realize ``S`` once per scenario (reusing the fail-closed
  two-stage :mod:`deixis.pipeline`), tagging every :class:`Realization` with its
  ``scenario_id`` and a provenance ledger of *which decision grounded which constraint*;
* :func:`compare_realizations` — diff two realizations **relative to a pre-registered
  observation set O** (:class:`ObservationQuery`), so the honest claim "the invariant
  spec is *preserved*, only the grounded part *changed*" is checkable and not shrunk
  post-hoc. Everything is read back from the exact-rational geometry via
  :mod:`deixis.verify.reverse` (RCC-8 relation / contact dimension / overlap occupancy);
* :func:`canonical_key` — a quotient key that collapses solutions equal up to the
  axis-aligned symmetry group (translation, per-axis reflection, axis permutation) and
  positive scaling, via *coordinate compression* (order-type) of the box set, so
  symmetric duplicates are not counted twice;
* :func:`decision_provenance` — reverse-trace ``constraint id -> GroundingDecision id``
  (who decided what) straight off a family-produced realization.

Exactness: coordinate handling is pure ``Fraction`` rank arithmetic (coordinate
compression compares exact endpoints); RCC read-back is integer bit arithmetic. No float
enters any decision.
"""
from __future__ import annotations

import itertools
import json
from typing import Optional, Sequence

from dataclasses import replace

from deixis.core import rcc8
from deixis.core.ids import Provenance
from deixis.core.types import (
    GroundingDecision,
    ObservationQuery,
    Realization,
    RelSpec,
)
from deixis.verify.reverse import extract_contact_dimension, extract_relations
from deixis import pipeline

__all__ = [
    "solve_family",
    "compare_realizations",
    "canonical_key",
    "decision_provenance",
]

# Any relation whose interiors intersect with positive measure (the pair *occupies* a
# shared overlap atom). DC = disjoint, EC = boundary-only touch => NOT occupied.
_OVERLAP_MASK = rcc8.mask("PO", "TPP", "NTPP", "TPPi", "NTPPi", "EQ")
_EC_BIT = rcc8.bit("EC")

# Provenance ``origin`` written by solve_family; decision_provenance reads it back.
_PROV_ORIGIN = "family"

# Sentinel recorded when an observation item exists in only one of the two realizations.
_ABSENT = "<absent>"

# The observation kinds this module knows how to read back from realized geometry.
_KNOWN_KINDS = frozenset(
    {"rcc8", "contact_dimension", "atom_occupancy", "cell_identity"}
)


# ---------------------------------------------------------------- solve_family
def solve_family(
    spec: RelSpec,
    scenarios: Sequence[Sequence[GroundingDecision]],
    domain: str = "aabb_2d",
    bounds: tuple = (0, 100),
    *,
    min_size=None,
) -> list[Realization]:
    """Realize ``spec`` once per grounding scenario; one :class:`Realization` each.

    Each scenario is a sequence of :class:`GroundingDecision`. All scenarios branch off
    the *same immutable* base ``spec`` (grounding is a pure function returning a fresh
    spec — see :mod:`deixis.grounding.grounding_op`), so no branch sees order-dependent
    mutation. The realization for scenario ``i`` carries ``scenario_id == "s{i}"`` and a
    :class:`~deixis.core.ids.Provenance` whose ``note`` is a JSON map
    ``{constraint_id: decision_id}`` recording which decision grounded which constraint
    (consumed by :func:`decision_provenance`). Region-targeting decisions are recorded in
    ``inputs`` but not in the constraint map (they ground no relation).
    """
    out: list[Realization] = []
    for i, sc in enumerate(scenarios):
        sc = tuple(sc)
        real = pipeline.run(
            spec,
            groundings=sc,
            domain=domain,
            bounds=bounds,
            scenario_id=f"s{i}",
            min_size=min_size,
        )
        # constraint id -> decision id, only for decisions that target a constraint.
        ledger: dict[str, str] = {}
        for d in sc:
            if spec.constraint(d.target) is not None:
                ledger[d.target] = d.id
        prov = Provenance(
            origin=_PROV_ORIGIN,
            actor="designer",
            activity="solve_family",
            inputs=tuple(d.id for d in sc),
            note=json.dumps(ledger, sort_keys=True),
        )
        out.append(replace(real, prov=prov))
    return out


# ---------------------------------------------------------------- observation read-back
def _relations_canon(real: Realization) -> dict[tuple[str, str], int]:
    """Definite RCC-8 base relation of every region pair, keyed by the *sorted* pair.

    :func:`extract_relations` keys by box order; we re-orient onto the canonical
    ``(min_id, max_id)`` direction (converse when the box order is reversed) so two
    realizations are compared on identical keys regardless of box ordering."""
    out: dict[tuple[str, str], int] = {}
    for (i, j), m in extract_relations(real).items():
        if i <= j:
            out[(i, j)] = m
        else:
            out[(j, i)] = rcc8.converse(m)
    return out


def _observe(real: Realization, kinds: frozenset[str]) -> dict[tuple, object]:
    """Project a realization onto the observation set O as ``{item_key: value}``.

    Item keys are tuples tagged by observation kind so the four kinds never collide:
    ``("rcc8", A, B)`` -> base relation name, ``("contact_dimension", A, B)`` -> int
    (only for EC pairs), ``("atom_occupancy", A, B)`` -> bool (interiors overlap),
    ``("cell_identity", region)`` -> True (region has a box). Only kinds present in O
    are emitted, honouring the pre-registered-O discipline."""
    obs: dict[tuple, object] = {}
    want_rcc = "rcc8" in kinds
    want_dim = "contact_dimension" in kinds
    want_occ = "atom_occupancy" in kinds

    if want_rcc or want_dim or want_occ:
        rels = _relations_canon(real)
        for (i, j), m in rels.items():
            if want_rcc:
                names = rcc8.names(m)
                obs[("rcc8", i, j)] = names[0] if names else "EMPTY"
            if want_occ:
                obs[("atom_occupancy", i, j)] = bool(m & _OVERLAP_MASK)
            if want_dim and m == _EC_BIT:
                # EC guarantees a boundary contact -> extract_contact_dimension is total.
                obs[("contact_dimension", i, j)] = int(
                    extract_contact_dimension(real, i, j)
                )

    if "cell_identity" in kinds:
        for b in real.boxes:
            obs[("cell_identity", b.region_id)] = True

    return obs


def compare_realizations(
    r1: Realization,
    r2: Realization,
    observation: Optional[ObservationQuery] = None,
) -> dict:
    """Diff two realizations *relative to a pre-registered observation set O*.

    Returns ``{"changed": {...}, "preserved": {...}}``. Each observation item (an RCC-8
    relation, a contact dimension, an overlap occupancy, or a cell/region identity — see
    :func:`_observe`) that both realizations agree on lands in ``preserved`` (mapped to
    its shared value); every item they disagree on — including one present in only one
    realization — lands in ``changed`` mapped to ``(value_in_r1, value_in_r2)`` (with the
    :data:`_ABSENT` sentinel for a missing side).

    Preservation is only meaningful against a fixed O (guard against shrinking O
    post-hoc); ``observation`` defaults to the minimal O of :class:`ObservationQuery`
    (rcc8 + contact_dimension + cell_identity + atom_occupancy).
    """
    if observation is None:
        observation = ObservationQuery()
    kinds = observation.kinds
    # Fail-closed on an un-registered observation kind: silently ignoring it would let a
    # preservation claim be satisfied vacuously (the guarantee is only as strong as O).
    unknown = kinds - _KNOWN_KINDS
    if unknown:
        raise ValueError(
            f"unknown observation kind(s) {sorted(unknown)}; "
            f"this module reads back only {sorted(_KNOWN_KINDS)}"
        )

    o1 = _observe(r1, kinds)
    o2 = _observe(r2, kinds)
    # Realizability status is always compared: it stops two non-realized (boxes=()) results
    # from vacuously "preserving" everything, and flags a realized-vs-unrealized pair. The
    # geometry observation items above are only trustworthy when both are REALIZED_IN_D.
    o1[("realization_status",)] = r1.status.value
    o2[("realization_status",)] = r2.status.value

    changed: dict = {}
    preserved: dict = {}
    for key in set(o1) | set(o2):
        v1 = o1.get(key, _ABSENT)
        v2 = o2.get(key, _ABSENT)
        if v1 == v2:
            preserved[key] = v1
        else:
            changed[key] = (v1, v2)
    return {"changed": changed, "preserved": preserved}


# ---------------------------------------------------------------- canonical key
def _compress(boxes, ndim: int):
    """Coordinate-compress a box set: replace each exact endpoint by its per-axis rank.

    Returns ``(compressed, sizes)`` where ``compressed`` maps ``region_id -> (lo_ranks,
    hi_ranks)`` (ranks are 0-based positions of that endpoint among the sorted *distinct*
    endpoint values on the axis) and ``sizes[d]`` is the number of distinct values on
    axis ``d``. Compression is exact (Fraction comparisons) and invariant to translation
    and to any positive, order-preserving rescaling of the coordinates."""
    # Well-formedness: a shared arity and unique region ids (mirrors reverse.py's
    # contract). A duplicate id would be silently overwritten below and two distinct
    # realizations could then collide on one key -- fail-closed instead.
    seen: set[str] = set()
    for b in boxes:
        if len(b.lo) != ndim or len(b.hi) != ndim:
            raise ValueError(
                f"box {b.region_id!r} has arity {len(b.lo)}/{len(b.hi)}, expected {ndim}"
            )
        if b.region_id in seen:
            raise ValueError(f"duplicate region id in realization: {b.region_id!r}")
        seen.add(b.region_id)

    ranks = []
    sizes = []
    for d in range(ndim):
        vals = sorted({b.lo[d] for b in boxes} | {b.hi[d] for b in boxes})
        ranks.append({v: i for i, v in enumerate(vals)})
        sizes.append(len(vals))
    compressed: dict[str, tuple] = {}
    for b in boxes:
        lo = tuple(ranks[d][b.lo[d]] for d in range(ndim))
        hi = tuple(ranks[d][b.hi[d]] for d in range(ndim))
        compressed[b.region_id] = (lo, hi)
    return compressed, sizes


def canonical_key(realization: Realization):
    """Canonical key quotienting solutions by axis-aligned symmetry + rescale/translation.

    Two realizations receive the *same* key iff their box sets have the same combinatorial
    **order-type** up to the axis-aligned symmetry group — translation, per-axis
    reflection, and axis permutation. Coordinate compression keeps only the *rank* of each
    endpoint, so the equivalence is intentionally coarser than metric congruence: it also
    folds in any per-axis order-preserving rescaling (equal ranks => same key regardless of
    the gaps between them). So a solution and its mirror / translate / axis-swap / rescale
    collapse to one key and are not double-counted, while a genuinely different arrangement
    (a different order-type — e.g. touching vs a gap) keeps a distinct key. Region identity
    is preserved (regions are *not* permuted): only geometry is abstracted.

    Implementation: coordinate-compress the boxes to per-axis ranks, then take the
    lexicographically minimal serialization over every reflection (rank ``r`` on axis
    ``d`` -> ``size[d]-1-r``, swapping lo/hi) and every axis permutation. Returns a
    hashable, comparable tuple; the empty realization maps to ``()``.
    """
    boxes = realization.boxes
    if not boxes:
        return ()
    ndim = len(boxes[0].lo)
    compressed, sizes = _compress(boxes, ndim)

    best = None
    for perm in itertools.permutations(range(ndim)):
        for flips in itertools.product((False, True), repeat=ndim):
            items = []
            for rid, (lo, hi) in compressed.items():
                nlo = []
                nhi = []
                for oldd in perm:
                    k = sizes[oldd]
                    l, h = lo[oldd], hi[oldd]
                    if flips[oldd]:
                        l, h = k - 1 - h, k - 1 - l
                    nlo.append(l)
                    nhi.append(h)
                items.append((rid, tuple(nlo), tuple(nhi)))
            ser = tuple(sorted(items))
            if best is None or ser < best:
                best = ser
    return best


# ---------------------------------------------------------------- provenance trace
def decision_provenance(realization: Realization) -> dict[str, str]:
    """Reverse-trace ``constraint id -> GroundingDecision id`` off a family realization.

    Reads the ledger :func:`solve_family` stamped into the realization's provenance
    ``note`` (a JSON ``{constraint_id: decision_id}`` map): who grounded what. A
    realization not produced by :func:`solve_family` (no family provenance, or an
    unparseable/absent note) yields an empty map rather than raising — the honest "no
    recorded grounding decisions" answer."""
    prov = realization.prov
    if prov is None or prov.origin != _PROV_ORIGIN or not prov.note:
        return {}
    try:
        data = json.loads(prov.note)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    # A well-formed ledger maps id-string -> id-string; anything else (arrays, nested
    # objects, null) is a corrupt note -> report nothing rather than fabricate ids.
    out: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
        else:
            return {}
    return out
