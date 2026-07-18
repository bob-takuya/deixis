"""M-field (a), intent layer — thresholding + **connected-component splitting**.

This module sits *on top of* :func:`deixis.field.scalar.threshold_ground` and adds
one move it deliberately does not make: after thresholding a field into a level
set, it splits that level set into its **lattice-connected components** and offers
each component as a separate *analysis-region candidate*.

Why a separate file, and why "candidate" rather than "room" or "convex space":

    field --threshold_ground(tau)--> level set L_tau (one cell set, maybe several
                                     disconnected blobs)
    L_tau --threshold_regions------> [component_0, component_1, ...]  (+ one
                                     GroundingDecision per component)

Each returned component is exactly a maximal set of cells reachable from one
another through **grid (von Neumann / face) adjacency** — cells differing by 1 on
exactly one axis. That is a purely *topological* fact about the sampled level set.
It is emphatically **not** a claim that a component is a "convex space" in the
space-syntax sense, nor an axial/convex map, nor even a room. Connectivity of the
threshold blob is a necessary hint, not a sufficient definition: a genuine convex
space is an additional, stronger decision (convexity, sightlines, use) that this
module does not make. So a component is offered as an *analysis-region candidate*
— raw material to hand to a space-syntax pass (an ``access.py`` graph builder, not
yet present) — and every candidate is externalized as its own
:class:`~deixis.core.types.GroundingDecision`, because carving one level set into
several named regions is itself a decision about who gets to say "these cells are
one space and those are another".

This is the demo5 bridge: demo5 shows one field thresholded into different rooms
under different ``tau``; here, at a *fixed* ``tau``, a single "field"/room can
still fall apart into several disconnected candidate regions (e.g. two alcoves off
a corridor that clear the threshold but are not connected through it). Each such
piece is a candidate the downstream spatial-syntax analysis may accept or reject.

Honest limits (do not overclaim):
- Adjacency is **face adjacency** (von Neumann: L1 distance 1), not diagonal
  (Moore) adjacency. Two cells touching only at a corner are *not* connected here.
  This is a stated modelling choice; a different adjacency gives different
  components. It is fixed and exact — no thresholded float geometry is involved.
- A "component" is a set of *sampled cells*, a discretization — the same honest
  caveat as :func:`threshold_ground`: it equals a continuous connected region only
  under a sampling/interpolation model this module does not fix.
- "analysis-region candidate" names an *intent*, not a verified property. Nothing
  here checks convexity, visibility, or usability. Naming a candidate does not
  make it a convex space.
- This module **does not edit** :mod:`deixis.field.scalar`; it only calls its
  public :func:`~deixis.field.scalar.threshold_ground`. All values remain exact
  ``fractions.Fraction``; frozen dataclasses; float-free.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import List, Tuple

import networkx as nx

from ..core.ids import Provenance
from ..core.types import GroundingDecision
from .scalar import Cell, ScalarField, threshold_ground

__all__ = [
    "threshold_regions",
    "connected_components",
]


def _component_signature(comp: frozenset) -> Tuple[str, str]:
    """Stable, exact signature of a component's *actual cells* (order-independent).

    Two components differing in even one cell get different signatures, so a
    per-component decision id built from this cannot collide across fields that
    share ``(region_id, tau, geometry)`` but threshold to different cell sets. The
    full sorted cell list is exact (int indices); the short digest is only an id
    suffix — the human-readable cell list is carried verbatim in the decision.
    """
    body = ";".join(",".join(str(i) for i in cell) for cell in sorted(comp))
    digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    return f"cells=[{body}]", digest


def _lattice_neighbors(cell: Cell) -> Tuple[Cell, ...]:
    """Face (von Neumann) neighbors of ``cell``: shift by +/-1 on one axis.

    Negative indices are allowed through here (they simply will not be members of
    a level set drawn from a non-negative grid); membership is what gates edges.
    """
    out: List[Cell] = []
    for axis in range(len(cell)):
        for step in (-1, 1):
            nb = list(cell)
            nb[axis] = nb[axis] + step
            out.append(tuple(nb))
    return tuple(out)


def connected_components(cells: frozenset) -> List[frozenset]:
    """Split a set of grid cells into lattice-connected components (face adjacency).

    Two cells are adjacent iff they differ by exactly 1 on exactly one axis (von
    Neumann / L1-distance-1 adjacency; corner-only contact does **not** connect).
    Returns the components as a list of ``frozenset``s, ordered deterministically
    by each component's minimum cell so the output is stable and reproducible.

    All cells must share the same rank; a mixed-rank set is a caller error.
    """
    if not cells:
        return []
    ranks = {len(c) for c in cells}
    if len(ranks) != 1:
        raise ValueError(f"all cells must share one rank; got ranks {sorted(ranks)}")

    members = set(cells)
    graph = nx.Graph()
    graph.add_nodes_from(members)
    for cell in members:
        for nb in _lattice_neighbors(cell):
            # Add each edge once (nb > cell) and only within the level set, so
            # isolated cells remain their own singleton component.
            if nb in members and nb > cell:
                graph.add_edge(cell, nb)

    comps = [frozenset(comp) for comp in nx.connected_components(graph)]
    comps.sort(key=lambda comp: min(comp))
    return comps


def threshold_regions(
    field: ScalarField,
    tau: Fraction,
    region_id: str,
    intent: str = "analysis_candidate",
    actor: str = "designer",
    rationale: str = "",
) -> Tuple[List[frozenset], List[GroundingDecision]]:
    """Threshold a field, then split the level set into connected candidate regions.

    Delegates the thresholding to :func:`deixis.field.scalar.threshold_ground`
    (this module edits nothing in ``scalar.py``), then partitions the resulting
    level set ``L_tau`` into its **lattice-connected components** (face adjacency)
    and offers each component as an *analysis-region candidate* — raw material for a
    downstream space-syntax pass, **not** a verified convex space.

    Returns ``(components, decisions)`` where:
      * ``components`` is a list of ``frozenset`` cell sets, one per connected
        component of ``L_tau``, ordered deterministically (by minimum cell). An
        empty level set yields ``[]``.
      * ``decisions`` is a list of one :class:`~deixis.core.types.GroundingDecision`
        **per component** (same length / order as ``components``): carving a single
        level set into several separately-named candidate regions is itself a
        decision, so each candidate carries its own externalized ledger entry.

    ``intent`` labels *what the components are being offered as* (default
    ``"analysis_candidate"``). It is recorded on every decision. It names an intent,
    not a proven property: no convexity/visibility/usability is checked here.

    Honest note: like ``threshold_ground``, a component is a set of *sampled* cells
    (a discretization), and adjacency is face-only — corner-touching cells are
    separate components under this fixed, exact model.
    """
    # 1) Delegate the actual thresholding to the un-edited core.
    base_cells, base_decision = threshold_ground(
        field, tau, region_id, actor=actor, rationale=rationale
    )

    # 2) Split the level set into lattice-connected components (topological fact).
    components = connected_components(base_cells)

    # 3) One GroundingDecision per component — each named candidate is a decision.
    geom = f"shape={field.grid_shape};origin={field.origin};spacing={field.spacing}"
    n = len(components)
    decisions: List[GroundingDecision] = []
    for k, comp in enumerate(components):
        candidate_id = f"{region_id}#c{k}"
        cells_desc, cells_digest = _component_signature(comp)
        decisions.append(
            GroundingDecision(
                # digest of the actual cells disambiguates two fields that share
                # (region_id, tau, geometry) but threshold to different cell sets
                id=f"gd:threshold_region:{candidate_id}:tau={tau}:{geom}:{cells_digest}",
                target=candidate_id,
                fixed_value_or_domain=(
                    f"analysis-region candidate {candidate_id} = connected component "
                    f"{k} of L_tau(region={region_id}, tau={tau}) over {geom}; "
                    f"|component|={len(comp)}; {cells_desc}; intent={intent!r} "
                    f"(candidate, NOT a verified convex space)"
                ),
                actor=actor,
                rationale=rationale or (
                    f"threshold field at tau={tau} for region {region_id!r}, then split "
                    f"the level set into {n} lattice-connected component(s) (face "
                    f"adjacency); component {k} is offered as an analysis-region "
                    f"candidate (intent={intent!r}) for downstream space-syntax "
                    "analysis — connectivity is a hint, not a convex-space claim; "
                    "reversible: the field and the un-split level set are retained"
                ),
                reversibility="high",
                prov=Provenance(
                    origin="threshold-region-intent",
                    actor=actor,
                    activity="threshold_regions",
                    # trace back to the delegated threshold decision, not re-derived
                    inputs=(region_id, base_decision.id),
                    note=(
                        f"tau={tau}; {geom}; intent={intent}; component {k} of {n}; "
                        f"|component|={len(comp)} of |L_tau|={len(base_cells)}"
                    ),
                ),
            )
        )

    return components, decisions
