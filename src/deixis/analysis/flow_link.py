"""``flow_link`` — map a DEC flow 1-cochain onto a directed graph and read *choice*.

The M-field(b) layer (:mod:`deixis.field.flow`) describes space as a **flow field**:
a value on each directed edge of an oriented cell complex (circulation / movement /
airflow / sightlines). Space-syntax analysis, by contrast, reads *paths through a
graph* — its central quantity is **choice** (媒介中心性), the betweenness centrality
that measures how often an element lies on shortest routes between other elements.

This module is the honest bridge between the two: it re-orients each edge of the
complex by the **sign** of the flow on it, keeps the **magnitude** as an exact
rational edge weight, and hands the result to :mod:`networkx` so that betweenness /
choice can be computed. Sources and sinks (湧出 / 吸込) are read straight off the DEC
:func:`~deixis.field.flow.divergence`, so this layer stays consistent with field/flow
by construction — it never re-derives divergence with a different sign convention.

Honest limits (do NOT read past these):

  * The graph is a **derived reading**, not a new source of truth. It carries only
    what the flow 1-cochain already says; it adds no geometry and no capacity model.
  * Orientation encodes the *sign* of the flow and the edge weight encodes its
    *magnitude* (a non-negative :class:`~fractions.Fraction`). A zero-flow edge has
    no defined direction, so it is **omitted** from the graph rather than drawn with
    an arbitrary arrow.
  * :func:`flow_betweenness` is a thin wrapper over
    :func:`networkx.edge_betweenness_centrality`. Centrality values are the floats
    networkx returns (a ratio of shortest-path counts); every quantity *we* introduce
    (flow, weight, divergence) stays an exact :class:`~fractions.Fraction`.
  * ``weight`` defaults to ``None`` → **topological** choice (each retained edge is a
    unit step). Pass ``weight="weight"`` to let flow magnitude act as a path *cost*
    (Dijkstra over exact rationals). We do not claim one reading is canonical.

No floats are introduced by this module. Frozen field/flow objects are only read.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

import networkx as nx

from deixis.field.flow import CellComplex, Cochain, divergence

__all__ = [
    "flow_to_graph",
    "flow_betweenness",
    "sources_sinks",
]

_ZERO = Fraction(0)


def flow_to_graph(complex: CellComplex, c1: Cochain) -> nx.DiGraph:
    """Map a flow 1-cochain onto a directed graph oriented by the flow's sign.

    For each edge ``e = tail → head`` of ``complex`` with flow value ``w = c1(e)``:

      * ``w > 0`` → a directed edge ``tail → head`` (flow runs with the edge);
      * ``w < 0`` → a directed edge ``head → tail`` (flow runs *against* the edge);
      * ``w == 0`` → **omitted** (a zero-flow edge has no defined direction).

    Every arc carries two attributes, both exact:

      * ``weight``        — the non-negative *magnitude* of the flow along the arc's own
        direction (a :class:`~fractions.Fraction`), i.e. ``abs(w)``. When several complex
        edges resolve onto the **same** ordered pair (hence the same physical direction),
        their magnitudes are summed — that is the genuine throughput along the arc, and no
        flow is silently dropped (a :class:`~networkx.DiGraph` keeps one arc per pair). This
        is what :func:`flow_betweenness` can use as a path cost; it is never negative.
      * ``contributions`` — a tuple ``((edge_id, base_value), …)`` recording each
        originating complex edge and its *signed* 1-cochain value **on that edge's own
        base orientation**. Provenance is kept structurally (per edge), so a reversed
        edge's sign is preserved rather than folded into a single lossy scalar. The arc's
        direction already encodes the sign, so the magnitude and the signed provenance are
        stored separately and never contradict each other.

    All ``complex.vertices`` are added as nodes first, so isolated (flow-free) vertices
    still appear in the graph. Requires a 1-cochain.
    """
    if c1.degree != 1:
        raise ValueError(f"flow_to_graph expects a 1-cochain, got degree {c1.degree}")
    g = nx.DiGraph()
    g.add_nodes_from(complex.vertices)
    for e in complex.edges:
        w = c1.get(e.id)
        if w == _ZERO:
            continue
        if w > _ZERO:
            u, v = e.tail, e.head
        else:
            u, v = e.head, e.tail
        mag = abs(w)
        contribution = (e.id, w)
        # If two edges resolve onto the same ordered pair they share a physical
        # direction, so magnitudes ADD (throughput) and every signed contribution is
        # preserved separately — no cancellation, no lossy merge.
        if g.has_edge(u, v):
            data = g[u][v]
            data["weight"] = data["weight"] + mag
            data["contributions"] = data["contributions"] + (contribution,)
        else:
            g.add_edge(u, v, weight=mag, contributions=(contribution,))
    return g


def flow_betweenness(
    G: nx.DiGraph,
    *,
    weight: Optional[str] = None,
    normalized: bool = True,
) -> dict:
    """**Choice** (媒介中心性) of each directed arc: edge betweenness centrality.

    A thin, explicit wrapper over :func:`networkx.edge_betweenness_centrality`. The
    result maps each arc ``(u, v)`` to how often it lies on shortest paths between
    vertex pairs — the graph reading of space-syntax *choice*: arcs at branch points /
    junctions, through which many routes must pass, score high.

    * ``weight=None`` (default) → **topological** choice: every retained arc is one
      step, so betweenness counts routes structurally.
    * ``weight="weight"`` → flow **magnitude** is used as an edge *cost* (Dijkstra over
      exact :class:`~fractions.Fraction` weights); routes prefer smaller-cost arcs.

    Returns the networkx dict as-is. The centrality values are floats (a ratio of
    path counts) produced *by networkx*; this module introduces no floats of its own.
    """
    return nx.edge_betweenness_centrality(G, normalized=normalized, weight=weight)


def sources_sinks(
    complex: CellComplex, c1: Cochain
) -> tuple[dict[str, Fraction], dict[str, Fraction]]:
    """Read the 湧出 / 吸込 vertices of a flow straight off the DEC divergence.

    Uses :func:`deixis.field.flow.divergence` (same sign convention, no re-derivation),
    so this stays consistent with field/flow by construction:

      * ``divergence(v) > 0`` → **source** (湧出, net out-flow);
      * ``divergence(v) < 0`` → **sink**   (吸込, net in-flow);
      * ``divergence(v) == 0`` → internal / balanced (in neither map).

    Returns ``(sources, sinks)``, each a dict ``{vertex_id: divergence}`` with the
    exact signed :class:`~fractions.Fraction` divergence. Sink values are negative
    (their sign is preserved, not flipped). Requires a 1-cochain.
    """
    if c1.degree != 1:
        raise ValueError(f"sources_sinks expects a 1-cochain, got degree {c1.degree}")
    div = divergence(c1, complex)
    sources: dict[str, Fraction] = {}
    sinks: dict[str, Fraction] = {}
    for v in complex.vertices:
        d = div.get(v)
        if d > _ZERO:
            sources[v] = d
        elif d < _ZERO:
            sinks[v] = d
    return sources, sinks
