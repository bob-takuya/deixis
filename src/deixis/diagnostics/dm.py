"""Dulmage-Mendelsohn (DM) structural decomposition — a research-only diagnostic.

What this is
------------
A *coarse* Dulmage-Mendelsohn decomposition of the (equation x variable) bipartite
graph induced by the **equality part of stage B** (the geometric realization stage).
It sorts the equality subsystem into three structural blocks via a maximum matching:

    - **overdetermined**   : more equations than variables can absorb (structurally
                             surplus equations — reachable from an unmatched equation);
    - **well-determined**  : an exactly-square, perfectly-matchable core;
    - **underdetermined**  : more variables than equations pin down (free structural
                             degrees of freedom — reachable from an unmatched variable).

Scope: **stage-B equalities only.** The only RCC-8 base relation that lowers to
*unconditional* per-coordinate equalities on AABB bounds is ``EQ`` (``A.lo[d]==B.lo[d]``
and ``A.hi[d]==B.hi[d]`` on every axis; cf. :mod:`deixis.solver.geom_solver`). EC/TPP
boundary coincidences are **disjunctive** (``Or(A.hi==B.lo, B.hi==A.lo)``, a *choice* of
disjunct) and are therefore *deliberately excluded* — they are not fixed equations.
A disjunctive mask that merely *contains* ``EQ`` (e.g. ``EQ|PO``) is likewise excluded:
it does not commit to the equality. Inequalities (``<``, ``<=``) are out of scope: DM is a
square-system structural method and inequalities are not equations.

HONESTY / anti-overclaiming (these limits are load-bearing, not disclaimers)
----------------------------------------------------------------------------
1. **Structural only.** DM sees the *incidence pattern* (which equation touches which
   variable), never the numeric coefficients. It reports *structural* over/under-
   determination. It **cannot** tell a semantically redundant equation from a numerically
   contradictory one — both are invisible to it. In particular a consistent cycle of
   equalities (``A=B, B=C, C=A``) is structurally *square* and is reported as
   well-determined, NOT overdetermined: DM refuses to flag it even though it is
   algebraically rank-deficient.
2. **It does NOT name a redundant constraint to remove.** An overdetermined block means
   "these equations+variables are structurally coupled with a surplus equation somewhere
   in the block"; *which* equation is surplus is matching-dependent and not unique, and
   removing it may or may not be semantically valid. No blame is attributed.
3. **These are lock *candidates*, not locked cells.** A well-determined / overdetermined
   block only *suggests where* commitment already constrains geometry; it does not decide
   or perform any grounding (lock). Grounding to absolute constants is out of scope, so
   equalities are treated purely *relatively* — a fully-consistent component still reads as
   underdetermined by its absolute-position DOF (this is expected, not a bug).
4. **Not novel.** DM decomposition is a classical, well-known structural-analysis device
   (Dulmage & Mendelsohn 1958) used across sparse-matrix / equation-system tooling. Its
   presence here is a standard measuring device applied to the equality subsystem, and it
   is claimed as nothing more.

The bipartite graph, the maximum matching, and the alternating-reachability partition are
all built with :mod:`networkx` (``networkx.bipartite.maximum_matching``).
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx
from networkx.algorithms import bipartite as nx_bipartite

from ..core import rcc8
from ..core.types import RelSpec

# ndim by selected AABB domain (mirrors deixis.solver.geom_solver, kept local on purpose
# so this diagnostic never couples to the solver's internals).
_NDIM: dict[str, int] = {"aabb_2d": 2, "aabb_3d": 3}

# The single RCC-8 base relation whose stage-B encoding is unconditional equalities.
_EQ_BIT: int = rcc8.bit("EQ")

_VAR = 0   # bipartite side: coordinate variable
_EQN = 1   # bipartite side: scalar equality equation


# ---------------------------------------------------------------- graph construction

def constraint_variable_graph(spec: RelSpec, domain: str = "aabb_2d") -> nx.Graph:
    """Bipartite (equation x variable) graph of the stage-B **equality** subsystem.

    Variables are the AABB coordinate scalars ``(region, side, axis)`` rendered as node
    ids ``"var:<region>|<lo|hi>|<axis>"``. Each *definite* ``EQ`` constraint contributes,
    per axis ``d`` and side ``s in {lo, hi}``, one equality-equation node connecting the two
    scalars it equates. Only variables actually touched by an equation appear (no isolated
    nodes). Non-``EQ`` and disjunctive (``EQ|...``) constraints contribute nothing — see the
    module docstring for why only unconditional equalities are modeled.

    Duplicated equality constraints (same pair, distinct constraint entries) yield distinct
    parallel equation nodes on purpose: that surplus is exactly what a DM decomposition is
    meant to expose *structurally* (without claiming which duplicate is the redundant one).

    Parameters
    ----------
    spec : RelSpec
        Source of ``regions`` and ``constraints``.
    domain : str
        ``'aabb_2d'`` or ``'aabb_3d'`` — selects the number of coordinate axes.

    Returns
    -------
    networkx.Graph
        Undirected bipartite graph; every node carries a ``bipartite`` attribute
        (``0`` = variable, ``1`` = equation).
    """
    if domain not in _NDIM:
        raise ValueError(f"unsupported domain {domain!r}; use one of {sorted(_NDIM)}")
    ndim = _NDIM[domain]

    graph = nx.Graph()
    for idx, c in enumerate(spec.constraints):
        # Only a definite (single-base) EQ commits to unconditional coordinate equalities.
        if c.rcc8_mask != _EQ_BIT:
            continue
        if c.src == c.dst:
            # A EQ A is a trivial self-relation: no non-degenerate equation.
            continue
        cid = c.id or f"{c.src}->{c.dst}"
        for side in ("lo", "hi"):
            for d in range(ndim):
                v1 = f"var:{c.src}|{side}|{d}"
                v2 = f"var:{c.dst}|{side}|{d}"
                if v1 == v2:
                    continue
                eqn = f"eq:{idx}:{cid}#{side}{d}"
                graph.add_node(v1, bipartite=_VAR)
                graph.add_node(v2, bipartite=_VAR)
                graph.add_node(eqn, bipartite=_EQN)
                graph.add_edge(eqn, v1)
                graph.add_edge(eqn, v2)
    return graph


# ---------------------------------------------------------------- DM decomposition

def _reach(graph: nx.Graph, match: dict, starts: Iterable, *, var_nonmatching: bool) -> set:
    """Alternating-path reachable set from ``starts`` (the DM tail construction).

    ``var_nonmatching=True`` (underdetermined tail from unmatched variables): leave a
    variable along a *non-matching* edge and an equation along its *matching* edge.
    ``var_nonmatching=False`` (overdetermined tail from unmatched equations): the mirror
    image — variables leave along matching edges, equations along non-matching edges.
    """
    visited = set(starts)
    stack = list(visited)
    while stack:
        n = stack.pop()
        part = graph.nodes[n]["bipartite"]
        for nb in graph.neighbors(n):
            is_match = match.get(n) == nb
            if part == _VAR:
                take = (not is_match) if var_nonmatching else is_match
            else:
                take = is_match if var_nonmatching else (not is_match)
            if take and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return visited


def dm_decompose(bipartite: nx.Graph) -> dict:
    """Coarse Dulmage-Mendelsohn decomposition of an (equation x variable) graph.

    Computes a maximum matching with ``networkx.bipartite.maximum_matching`` and partitions
    every node into exactly one of the over/well/under-determined blocks by
    alternating-path reachability from the unmatched nodes (the classical DM construction).

    Returns
    -------
    dict
        ``{"matching": <both-direction match dict>, "variables": [...], "equations": [...],
        "overdetermined": {"variables": [...], "equations": [...]},
        "welldetermined": {...}, "underdetermined": {...}}``.
        The three blocks partition all nodes; ``over`` and ``under`` are provably disjoint.

    Notes
    -----
    The blocks are *structural*. See the module docstring: an overdetermined block does not
    identify a removable/redundant constraint, and a well-determined block may still be
    numerically singular (DM cannot see coefficients).
    """
    var_nodes = {n for n, dd in bipartite.nodes(data=True) if dd.get("bipartite") == _VAR}
    eqn_nodes = {n for n, dd in bipartite.nodes(data=True) if dd.get("bipartite") == _EQN}

    if bipartite.number_of_edges() == 0:
        match: dict = {}
    else:
        # top_nodes makes the matching well-defined even when the graph is disconnected.
        match = nx_bipartite.maximum_matching(bipartite, top_nodes=var_nodes)

    matched = set(match.keys())
    unmatched_v = var_nodes - matched
    unmatched_e = eqn_nodes - matched

    under = _reach(bipartite, match, unmatched_v, var_nonmatching=True)
    over = _reach(bipartite, match, unmatched_e, var_nonmatching=False)
    well = (var_nodes | eqn_nodes) - under - over

    def _split(nodes: set) -> dict:
        return {
            "variables": sorted(nodes & var_nodes),
            "equations": sorted(nodes & eqn_nodes),
        }

    return {
        "matching": dict(match),
        "variables": sorted(var_nodes),
        "equations": sorted(eqn_nodes),
        "overdetermined": _split(over),
        "welldetermined": _split(well),
        "underdetermined": _split(under),
    }


# ---------------------------------------------------------------- top-level diagnose

def diagnose(spec: RelSpec, domain: str = "aabb_2d") -> dict:
    """Structural over/under-determination report for a spec's stage-B equality subsystem.

    Returns
    -------
    dict
        ``{"overconstrained": bool, "underconstrained": bool, "blocks": <dm_decompose dict>}``.
        ``overconstrained`` is true iff some equation is structurally surplus (an
        overdetermined block exists); ``underconstrained`` iff some coordinate variable is
        structurally free (an underdetermined block exists). Both can be true at once (a
        spec can be locally over- and elsewhere under-determined).

    This is a *description*, not a verdict: a true ``overconstrained`` does NOT mean the spec
    is inconsistent, and it does NOT point at which constraint to drop (see module docstring).
    """
    graph = constraint_variable_graph(spec, domain)
    blocks = dm_decompose(graph)
    return {
        "overconstrained": bool(blocks["overdetermined"]["equations"]),
        "underconstrained": bool(blocks["underdetermined"]["variables"]),
        "blocks": blocks,
    }


__all__ = ["constraint_variable_graph", "dm_decompose", "diagnose"]
