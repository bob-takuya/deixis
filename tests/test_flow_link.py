"""Tests for :mod:`deixis.analysis.flow_link` — flow-cochain → directed graph → choice.

Covers the module's honest claims:
  * a conservative source→sink flow becomes a directed graph oriented by flow sign;
  * negative flow reverses the arc direction and keeps a non-negative magnitude weight;
  * zero-flow edges are omitted (no arbitrary arrow);
  * sources / sinks agree with :func:`deixis.field.flow.divergence` (field/flow consistency);
  * edge betweenness (choice) is higher on a shared bottleneck / junction arc;
  * flow magnitudes stay exact rationals (no floats introduced by this layer).
"""
from fractions import Fraction

import networkx as nx
import pytest

from deixis.field.flow import (
    CellComplex,
    Cochain,
    Edge,
    Face,
    divergence,
    flux_conservation,
    make_grid,
)
from deixis.analysis.flow_link import (
    flow_betweenness,
    flow_to_graph,
    sources_sinks,
)


# ---------------------------------------------------------------- fixtures
def _grid():
    return make_grid(2, 2)


def _unit_L_flow():
    """Unit flow along v:0:0 -> v:1:0 -> v:2:0 -> v:2:1 -> v:2:2 (conservative)."""
    path = ["eh:0:0", "eh:1:0", "ev:2:0", "ev:2:1"]
    return Cochain(degree=1, values={e: Fraction(1) for e in path})


# ---------------------------------------------------------------- graph shape
def test_flow_to_graph_orients_positive_flow_with_edge():
    g = _grid()
    c1 = _unit_L_flow()
    G = flow_to_graph(g, c1)
    assert isinstance(G, nx.DiGraph)
    # every complex vertex is present as a node
    for v in g.vertices:
        assert v in G
    # positive unit flow -> arc along the edge direction (tail -> head)
    assert G.has_edge("v:0:0", "v:1:0")
    assert G["v:0:0"]["v:1:0"]["weight"] == Fraction(1)
    # provenance preserves the signed base value of the originating edge
    assert G["v:0:0"]["v:1:0"]["contributions"] == (("eh:0:0", Fraction(1)),)


def test_flow_to_graph_only_flowing_edges_become_arcs():
    g = _grid()
    c1 = _unit_L_flow()
    G = flow_to_graph(g, c1)
    # exactly the 4 path edges carry flow -> exactly 4 arcs
    assert G.number_of_edges() == 4
    expected = {
        ("v:0:0", "v:1:0"),
        ("v:1:0", "v:2:0"),
        ("v:2:0", "v:2:1"),
        ("v:2:1", "v:2:2"),
    }
    assert set(G.edges()) == expected


def test_zero_flow_edge_is_omitted():
    g = _grid()
    # explicit zero on an edge must not create an arc
    c1 = Cochain(degree=1, values={"eh:0:0": Fraction(0), "eh:1:0": Fraction(1)})
    G = flow_to_graph(g, c1)
    assert not G.has_edge("v:0:0", "v:1:0")
    assert G.has_edge("v:1:0", "v:2:0")
    assert G.number_of_edges() == 1


def test_negative_flow_reverses_direction_and_weight_is_magnitude():
    g = _grid()
    # flow -3 on eh:0:0 (tail v:0:0, head v:1:0) => runs head -> tail
    c1 = Cochain(degree=1, values={"eh:0:0": Fraction(-3)})
    G = flow_to_graph(g, c1)
    assert G.has_edge("v:1:0", "v:0:0")          # reversed
    assert not G.has_edge("v:0:0", "v:1:0")
    d = G["v:1:0"]["v:0:0"]
    assert d["weight"] == Fraction(3)            # non-negative magnitude
    assert d["weight"] > 0
    # the arc direction encodes the sign; provenance still keeps the signed base value
    assert d["contributions"] == (("eh:0:0", Fraction(-3)),)


def test_fractional_weights_stay_exact_fractions():
    g = _grid()
    c1 = Cochain(degree=1, values={"eh:0:0": Fraction(2, 7)})
    G = flow_to_graph(g, c1)
    w = G["v:0:0"]["v:1:0"]["weight"]
    assert isinstance(w, Fraction)
    assert w == Fraction(2, 7)
    assert not isinstance(w, float)


def test_flow_to_graph_rejects_non_1_cochain():
    g = _grid()
    c0 = Cochain(degree=0, values={"v:0:0": Fraction(1)})
    with pytest.raises(ValueError):
        flow_to_graph(g, c0)


def test_parallel_flows_onto_same_pair_accumulate():
    # two directed edges collapsing to the same ordered pair must sum, not drop.
    cx = CellComplex(
        vertices=("a", "b"),
        edges=(Edge("e1", "a", "b"), Edge("e2", "a", "b")),
    )
    c1 = Cochain(degree=1, values={"e1": Fraction(1), "e2": Fraction(2)})
    G = flow_to_graph(cx, c1)
    assert G.number_of_edges() == 1
    assert G["a"]["b"]["weight"] == Fraction(3)
    assert G["a"]["b"]["contributions"] == (("e1", Fraction(1)), ("e2", Fraction(2)))


def test_reverse_oriented_parallel_flows_do_not_cancel():
    # e1 (base a->b) carries +1  => physical flow a->b, magnitude 1.
    # e2 (base b->a) carries -2  => reversed => physical flow a->b, magnitude 2.
    # Both resolve onto the SAME ordered pair (a,b): magnitudes must ADD to 3 (the true
    # throughput), and the signed base values must both survive (no lossy cancellation).
    cx = CellComplex(
        vertices=("a", "b"),
        edges=(Edge("e1", "a", "b"), Edge("e2", "b", "a")),
    )
    c1 = Cochain(degree=1, values={"e1": Fraction(1), "e2": Fraction(-2)})
    G = flow_to_graph(cx, c1)
    assert G.number_of_edges() == 1
    assert G.has_edge("a", "b")
    assert G["a"]["b"]["weight"] == Fraction(3)          # 1 + 2, not -1
    assert G["a"]["b"]["contributions"] == (("e1", Fraction(1)), ("e2", Fraction(-2)))


# ---------------------------------------------------------------- sources / sinks
def test_sources_sinks_match_divergence():
    g = _grid()
    c1 = _unit_L_flow()
    sources, sinks = sources_sinks(g, c1)
    # the L-path is a unit source at its start and unit sink at its end
    assert sources == {"v:0:0": Fraction(1)}
    assert sinks == {"v:2:2": Fraction(-1)}
    # and this is *exactly* what field/flow.divergence says (consistency by construction)
    div = divergence(c1, g)
    for v, d in {**sources, **sinks}.items():
        assert div.get(v) == d
    # internal vertices balance (conservative flow) => in neither map
    inner = "v:1:0"
    assert inner not in sources and inner not in sinks
    assert div.get(inner) == Fraction(0)


def test_sources_sinks_consistent_with_flux_conservation():
    g = _grid()
    c1 = _unit_L_flow()
    sources, sinks = sources_sinks(g, c1)
    # declaring exactly the discovered sources+sinks as the prescribed divergence
    # makes the flow conservative everywhere (no violations) — field/flow agrees.
    prescribed = Cochain(degree=0, values={**sources, **sinks})
    assert flux_conservation(c1, g, source=prescribed) == []


def test_sources_sinks_rejects_non_1_cochain():
    g = _grid()
    c0 = Cochain(degree=0, values={"v:0:0": Fraction(1)})
    with pytest.raises(ValueError):
        sources_sinks(g, c0)


# ---------------------------------------------------------------- betweenness / choice
def test_flow_betweenness_returns_centrality_for_each_arc():
    g = _grid()
    c1 = _unit_L_flow()
    G = flow_to_graph(g, c1)
    bc = flow_betweenness(G)
    assert set(bc.keys()) == set(G.edges())


def test_betweenness_high_on_shared_junction_arc():
    # Two sources (a, b) both feed a bottleneck edge m->n, which then fans to two
    # sinks (c, d). The bottleneck arc lies on every source->sink route => top choice.
    #        a          c
    #          \      /
    #        m --------- n        (wait: build explicitly below)
    #          /      \
    #        b          d
    cx = CellComplex(
        vertices=("a", "b", "m", "n", "c", "d"),
        edges=(
            Edge("e_am", "a", "m"),
            Edge("e_bm", "b", "m"),
            Edge("e_mn", "m", "n"),   # the bottleneck / junction
            Edge("e_nc", "n", "c"),
            Edge("e_nd", "n", "d"),
        ),
    )
    c1 = Cochain(
        degree=1,
        values={"e_am": 1, "e_bm": 1, "e_mn": 2, "e_nc": 1, "e_nd": 1},
    )
    G = flow_to_graph(cx, c1)
    bc = flow_betweenness(G)
    bottleneck = bc[("m", "n")]
    # the bottleneck must strictly dominate every other arc's choice value
    others = [v for k, v in bc.items() if k != ("m", "n")]
    assert all(bottleneck > o for o in others)


def test_flow_betweenness_weighted_uses_fraction_costs():
    # weighted variant must run over Fraction weights without touching floats.
    cx = CellComplex(
        vertices=("a", "m", "n", "c"),
        edges=(
            Edge("e_am", "a", "m"),
            Edge("e_mn", "m", "n"),
            Edge("e_nc", "n", "c"),
        ),
    )
    c1 = Cochain(degree=1, values={"e_am": Fraction(1, 2), "e_mn": Fraction(3), "e_nc": Fraction(1)})
    G = flow_to_graph(cx, c1)
    bc = flow_betweenness(G, weight="weight")
    assert set(bc.keys()) == set(G.edges())
    # middle arc lies on the a->c route -> positive choice
    assert bc[("m", "n")] > 0
