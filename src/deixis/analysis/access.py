"""``access`` — space-syntax *access graph* and centrality over the Relational IR.

Space syntax (Hillier & Hanson, *The Social Logic of Space*, 1984) reads a plan as a
**graph of permeability**: nodes are convex spaces / rooms, and an edge means one can
*move* from one to the other. Its central measures are read off this graph:

  * **step depth** — the topological distance (number of steps) from a chosen root;
  * **integration** — how shallow a space is *on average* from everywhere else, put on
    a graph-size-independent scale (Relative Asymmetry → Real Relative Asymmetry);
  * **choice** (媒介中心性) — how often a space lies on shortest routes between others
    (betweenness centrality).

This module derives that graph from the qualitative Relational IR and computes the
literature measures. Two honest limits are wired into the code, not just the prose:

  **1. RCC contact is NOT access.**  RCC-8 ``EC`` (externally connected / touching) and
  ``PO`` (partial overlap) tell you two regions *share boundary or interior* — a party
  wall is ``EC``; so is a doorway. Neither guarantees you can *pass* between them. So
  :func:`to_access_graph` lays down a contact edge for every ``EC``/``PO`` pair but marks
  it ``passable=False`` **by default**. Passage is a separate, externalised **grounding
  decision** (an opening / doorway): only contacts whose id is listed in ``passable``
  become traversable. Fail-closed: no declared opening ⇒ no access.

  **2. Integration is NOT raw closeness.**  We do not return
  :func:`networkx.closeness_centrality`. We compute Mean Depth exactly, form Relative
  Asymmetry ``RA = 2(MD-1)/(n-2)`` and Real Relative Asymmetry ``RRA = RA / D_n`` with the
  Hillier–Hanson diamond value ``D_n`` implemented explicitly (see :func:`d_value`).

Exactness: step depth is an ``int``; Mean Depth and RA stay exact
:class:`~fractions.Fraction`. The *only* floats produced are RRA (because ``D_n`` needs a
base-2 logarithm) and choice (a networkx ratio) — both come from the literature's
normalisation, never from a coordinate we invent, and ``D_n`` is exposed so the
normalising constant is inspectable rather than hidden.
"""
from __future__ import annotations

from fractions import Fraction
from math import log2
from typing import Optional

import networkx as nx

from deixis.core import rcc8
from deixis.core.types import RelSpec

__all__ = [
    "CONTACT_MASK",
    "to_access_graph",
    "step_depth",
    "justified_graph",
    "mean_depth",
    "relativised_asymmetry",
    "d_value",
    "integration",
    "choice",
]

# The two RCC-8 base relations that put two regions in *contact* (adjacency candidates):
# EC (externally connected / touching) and PO (partial overlap). A relation whose mask
# intersects this set means "contact is possible"; a single-bit EC or PO means contact
# is *certain*.
CONTACT_MASK: int = rcc8.mask("EC", "PO")


# --------------------------------------------------------------------- passability
def _is_passable(data: dict) -> bool:
    """Whether an edge may be traversed.

    Edges built by :func:`to_access_graph` always carry an explicit ``passable`` flag
    (a grounding decision). A plain :class:`networkx.Graph` handed in from elsewhere
    carries no such flag; there we default to ``True`` (treat every drawn edge as a real
    connection), so the depth/centrality helpers also work on ordinary graphs.
    """
    return bool(data.get("passable", True))


def _passable_view(G: nx.Graph) -> nx.Graph:
    """Undirected sub-graph keeping every node but only the *passable* edges."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if _is_passable(data):
            H.add_edge(u, v, **data)
    return H


# --------------------------------------------------------------------- graph build
def to_access_graph(spec: RelSpec, passable: Optional[set[str]] = None) -> nx.Graph:
    """Derive the space-syntax **access graph** from an IR's RCC-8 contact relations.

    Nodes are the region ids of ``spec``. For every :class:`RelationConstraint` whose
    ``rcc8_mask`` intersects :data:`CONTACT_MASK` (``EC``/``PO``), an undirected contact
    edge is drawn between ``src`` and ``dst``. Each edge carries:

      * ``rcc8``            — the constraint's mask (int) it came from;
      * ``contact``         — tuple of contact base names present (subset of ``EC``/``PO``);
      * ``contact_certain`` — ``True`` iff the relation is a *single* base ``EC`` or ``PO``
        (definite contact); ``False`` if contact is only one disjunct of an uncertain mask;
      * ``passable``        — whether the edge may be traversed (see below);
      * ``grounding_decided`` — whether ``passable`` was turned on by a grounding decision.

    **Passage is a grounding decision, not an RCC fact.**  ``EC``/``PO`` mean the regions
    *touch*; a solid party wall touches exactly as ``EC`` as a doorway does. RCC-8 cannot
    tell them apart, so a contact edge is **not** passable on its own. ``passable`` is the
    set of constraint ids that an externalised opening / doorway decision has grounded as
    traversable; only those edges get ``passable=True`` (and ``grounding_decided=True``).
    ``passable=None`` (the fail-closed default) leaves *every* contact non-passable —
    contact without a declared opening yields no access. To recover the naive
    "contact = access" reading, pass the ids of all contact constraints explicitly.

    If several constraints resolve onto the same unordered pair (e.g. a relation and its
    converse), the edge is merged: contact names are unioned, ``passable`` is OR-ed
    (any grounded opening makes the pair passable), and ``contact_certain`` stays ``True``
    only while every contributing relation is a definite single ``EC``/``PO``.
    """
    passable_ids = passable or set()
    G = nx.Graph()
    for r in spec.regions:
        G.add_node(r.id)

    for c in spec.constraints:
        contact_bits = c.rcc8_mask & CONTACT_MASK
        if contact_bits == 0:
            continue  # no EC/PO possible → not a contact, not an adjacency candidate
        # ensure endpoints exist as nodes even if a region was not declared
        for rid in (c.src, c.dst):
            if rid not in G:
                G.add_node(rid)
        names = tuple(rcc8.names(contact_bits))
        certain = rcc8.is_base(c.rcc8_mask) and (c.rcc8_mask & ~CONTACT_MASK) == 0
        is_pass = c.id in passable_ids

        if G.has_edge(c.src, c.dst):
            data = G[c.src][c.dst]
            merged = tuple(sorted(set(data["contact"]) | set(names),
                                  key=rcc8.BASE.index))
            data["contact"] = merged
            data["rcc8"] = data["rcc8"] | c.rcc8_mask
            data["contact_certain"] = bool(data["contact_certain"] and certain)
            data["passable"] = bool(data["passable"] or is_pass)
            data["grounding_decided"] = bool(data["grounding_decided"] or is_pass)
        else:
            G.add_edge(
                c.src, c.dst,
                rcc8=c.rcc8_mask,
                contact=names,
                contact_certain=certain,
                passable=is_pass,
                grounding_decided=is_pass,
            )
    return G


# --------------------------------------------------------------------- depth
def step_depth(G: nx.Graph, root: str) -> dict[str, int]:
    """Topological **step depth** of every node reachable from ``root``.

    Breadth-first distance (unweighted number of steps) over *passable* edges only, so a
    contact walled off with no opening does not shorten anything. ``root`` maps to ``0``.
    Unreachable nodes are **omitted** (rather than given ``inf``): the space-syntax depth
    measures are only defined within a connected access component.
    """
    if root not in G:
        raise ValueError(f"root {root!r} is not a node of the graph")
    depth: dict[str, int] = {root: 0}
    frontier = [root]
    while frontier:
        nxt: list[str] = []
        for u in frontier:
            for v in G.adj[u]:
                if v in depth:
                    continue
                if not _is_passable(G.edges[u, v]):
                    continue
                depth[v] = depth[u] + 1
                nxt.append(v)
        frontier = nxt
    return depth


def justified_graph(G: nx.Graph, root: str) -> nx.Graph:
    """The **justified graph** (j-graph) of ``G`` rooted at ``root``.

    In space syntax a justified graph is the access graph *re-drawn* with the root at the
    bottom and every other node lifted to its step-depth level — it keeps all the
    (passable) edges, it is not a spanning tree. This returns the sub-graph induced by the
    nodes reachable from ``root`` over passable edges, with each node annotated
    ``depth`` = its step depth and the graph annotated ``root``. Non-passable contact edges
    are dropped (they are not part of the permeability graph).
    """
    depth = step_depth(G, root)
    reach = set(depth)
    H = nx.Graph()
    for node in reach:
        attrs = dict(G.nodes[node])
        attrs["depth"] = depth[node]
        H.add_node(node, **attrs)
    for u, v, data in G.edges(data=True):
        if u in reach and v in reach and _is_passable(data):
            H.add_edge(u, v, **data)
    H.graph["root"] = root
    return H


# --------------------------------------------------------------------- centrality
def _depths_and_n(G: nx.Graph, root: str) -> tuple[dict[str, int], int]:
    depth = step_depth(G, root)
    return depth, len(depth)


def mean_depth(G: nx.Graph, root: str) -> Optional[Fraction]:
    """Mean Depth of ``root``: mean step depth to the other reachable nodes.

    ``MD = (Σ depth) / (n - 1)`` where ``n`` is the size of ``root``'s reachable component.
    Returns an **exact** :class:`~fractions.Fraction`, or ``None`` when ``root`` reaches no
    other node (``n < 2``), where Mean Depth is undefined.
    """
    depth, n = _depths_and_n(G, root)
    if n < 2:
        return None
    total = sum(depth.values())  # root contributes 0
    return Fraction(total, n - 1)


def relativised_asymmetry(G: nx.Graph) -> dict[str, Optional[Fraction]]:
    """**Relative Asymmetry** ``RA`` per node (Hillier & Hanson 1984).

    ``RA = 2 (MD - 1) / (n - 2)`` where ``MD`` is Mean Depth and ``n`` the reachable
    component size. RA rescales Mean Depth to ``[0, 1]`` independently of graph size:
    ``0`` = maximally integrating (a hub everyone is one step from), ``1`` = maximally
    segregating (the end of a chain). Stays an **exact** :class:`~fractions.Fraction`;
    ``None`` when ``n < 3`` (RA is undefined for fewer than three spaces).
    """
    out: dict[str, Optional[Fraction]] = {}
    for node in G.nodes:
        depth, n = _depths_and_n(G, node)
        if n < 3:
            out[node] = None
            continue
        md = Fraction(sum(depth.values()), n - 1)
        out[node] = Fraction(2) * (md - 1) / (n - 2)
    return out


def d_value(n: int) -> float:
    """Hillier–Hanson **diamond value** ``D_n`` used to normalise RA into RRA.

    ``D_n`` is the RA of the root of a diamond-shaped justified graph of ``n`` nodes — the
    canonical yardstick that makes Real Relative Asymmetry comparable across graph sizes
    (Hillier & Hanson 1984; Teklenburg et al. 1993)::

        D_n = 2 { n [ log2((n + 2) / 3) - 1 ] + 1 } / [ (n - 1)(n - 2) ]

    This is a **float** because of the base-2 logarithm — the one place the literature's
    normalisation forces us off exact rationals. It is exposed as its own function so the
    normalising constant is inspectable, not buried inside RRA. Requires ``n >= 3``.
    """
    if n < 3:
        raise ValueError(f"d_value requires n >= 3, got {n}")
    return 2.0 * (n * (log2((n + 2) / 3.0) - 1.0) + 1.0) / ((n - 1) * (n - 2))


def integration(G: nx.Graph) -> dict[str, Optional[float]]:
    """**Integration** as Real Relative Asymmetry ``RRA = RA / D_n`` per node.

    The space-syntax integration reading, built through the literature chain
    Mean Depth → Relative Asymmetry → Real Relative Asymmetry (never raw closeness). Lower
    RRA = *more integrated* (shallower from everywhere); higher = more segregated. Some
    texts report the integration value ``i = 1 / RRA`` instead; that is a reciprocal of
    this and is left to the caller. ``None`` for nodes whose reachable component has
    ``n < 3`` (RA/D_n undefined). Result is **float** solely because ``D_n`` (see
    :func:`d_value`) needs a logarithm; the RA numerator is computed exactly first.
    """
    ra = relativised_asymmetry(G)
    out: dict[str, Optional[float]] = {}
    for node in G.nodes:
        ra_node = ra[node]
        if ra_node is None:
            out[node] = None
            continue
        _, n = _depths_and_n(G, node)
        out[node] = float(ra_node) / d_value(n)
    return out


def choice(G: nx.Graph, *, normalized: bool = True) -> dict[str, float]:
    """**Choice** (媒介中心性) per node: betweenness centrality over passable edges.

    A thin wrapper over :func:`networkx.betweenness_centrality`, computed on the
    *passable* sub-graph so that walled-off contacts carry no routes. Choice measures how
    often a space lies on shortest paths between other spaces — spaces at junctions that
    many routes must cross score high. The values are the floats networkx returns (a ratio
    of shortest-path counts); this module introduces no floats of its own here.
    """
    return nx.betweenness_centrality(_passable_view(G), normalized=normalized)
