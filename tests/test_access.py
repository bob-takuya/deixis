"""Tests for :mod:`deixis.analysis.access` — space-syntax access graph + centrality.

Covers the module's honest claims:
  * a contact adjacency graph is derived from RCC-8 EC/PO (DC etc. are not contacts);
  * passage is a *grounding decision*, not an RCC fact — an EC contact is non-passable
    by default and only becomes traversable when its constraint id is grounded passable;
  * step depth / justified graph are computed over passable edges only (walls block);
  * Mean Depth and Relative Asymmetry follow the literature formulas exactly and stay
    exact Fractions; RRA = RA / D_n uses the explicit Hillier–Hanson diamond value;
  * choice is networkx betweenness over the passable sub-graph.
"""
from __future__ import annotations

from fractions import Fraction
from math import log2

import networkx as nx
import pytest

from deixis.core import rcc8
from deixis.core.types import Region, RelationConstraint, RelSpec
from deixis.analysis.access import (
    CONTACT_MASK,
    choice,
    d_value,
    integration,
    justified_graph,
    mean_depth,
    relativised_asymmetry,
    step_depth,
    to_access_graph,
)


# ---------------------------------------------------------------- builders
def _c(src, dst, *names, cid=None):
    return RelationConstraint(
        src=src, dst=dst, rcc8_mask=rcc8.mask(*names), id=cid or f"{src}-{dst}",
    )


def _spec(regions, constraints):
    return RelSpec(
        regions=tuple(Region(id=r) for r in regions),
        constraints=tuple(constraints),
    )


def _line(n, *, passable=None):
    """Path graph over regions r0..r{n-1} joined by EC contacts (optionally passable)."""
    regions = [f"r{i}" for i in range(n)]
    cons = [_c(f"r{i}", f"r{i+1}", "EC") for i in range(n - 1)]
    spec = _spec(regions, cons)
    return to_access_graph(spec, passable=passable)


def _all_contact_ids(spec):
    return {c.id for c in spec.constraints
            if c.rcc8_mask & CONTACT_MASK}


# ---------------------------------------------------------------- graph derivation
def test_contact_edge_from_ec_and_po_only():
    spec = _spec(
        ["a", "b", "c", "d"],
        [_c("a", "b", "EC"), _c("b", "c", "PO"), _c("c", "d", "DC")],
    )
    G = to_access_graph(spec)
    assert isinstance(G, nx.Graph)
    assert set(G.nodes) == {"a", "b", "c", "d"}
    assert G.has_edge("a", "b")           # EC → contact
    assert G.has_edge("b", "c")           # PO → contact
    assert not G.has_edge("c", "d")       # DC → not a contact
    assert G["a"]["b"]["contact"] == ("EC",)
    assert G["b"]["c"]["contact"] == ("PO",)


def test_contact_certain_vs_uncertain():
    spec = _spec(
        ["a", "b", "c"],
        [_c("a", "b", "EC"), _c("b", "c", "EC", "DC")],
    )
    G = to_access_graph(spec)
    # single-base EC → definite contact
    assert G["a"]["b"]["contact_certain"] is True
    # {EC, DC} → contact only possible, not certain (edge still drawn)
    assert G.has_edge("b", "c")
    assert G["b"]["c"]["contact_certain"] is False
    assert G["b"]["c"]["contact"] == ("EC",)   # only the contact disjunct is reported


# ---------------------------------------------------------------- passage = grounding
def test_contact_is_not_passable_by_default():
    # RCC EC does NOT guarantee passage: with no grounding, the contact is a wall.
    spec = _spec(["a", "b"], [_c("a", "b", "EC")])
    G = to_access_graph(spec)            # passable=None → fail-closed
    assert G.has_edge("a", "b")
    assert G["a"]["b"]["passable"] is False
    assert G["a"]["b"]["grounding_decided"] is False


def test_passable_edge_is_grounding_decided():
    spec = _spec(["a", "b", "c"],
                 [_c("a", "b", "EC"), _c("b", "c", "EC")])
    # only a–b is grounded as an opening (doorway)
    G = to_access_graph(spec, passable={"a-b"})
    assert G["a"]["b"]["passable"] is True
    assert G["a"]["b"]["grounding_decided"] is True     # passage came from grounding
    # b–c is EC (touches) but has no opening → still not passable
    assert G["b"]["c"]["passable"] is False
    assert G["b"]["c"]["grounding_decided"] is False


def test_converse_constraints_merge_and_or_passable():
    spec = _spec(["a", "b"],
                 [_c("a", "b", "EC", cid="fwd"), _c("b", "a", "EC", cid="rev")])
    G = to_access_graph(spec, passable={"rev"})
    assert G.number_of_edges() == 1
    # OR of the two contributing constraints: one grounded opening suffices
    assert G["a"]["b"]["passable"] is True


# ---------------------------------------------------------------- step depth
def test_step_depth_bfs_over_passable_edges():
    G = _line(4, passable={"r0-r1", "r1-r2", "r2-r3"})
    d = step_depth(G, "r0")
    assert d == {"r0": 0, "r1": 1, "r2": 2, "r3": 3}


def test_non_passable_edge_blocks_traversal():
    # r0-r1 open, r1-r2 walled (no grounding), r2-r3 open
    G = _line(4, passable={"r0-r1", "r2-r3"})
    d = step_depth(G, "r0")
    assert d == {"r0": 0, "r1": 1}      # r2, r3 unreachable (wall between r1,r2)
    assert "r2" not in d and "r3" not in d


def test_step_depth_rejects_missing_root():
    G = _line(2, passable={"r0-r1"})
    with pytest.raises(ValueError):
        step_depth(G, "nope")


# ---------------------------------------------------------------- justified graph
def test_justified_graph_annotates_depth_and_root():
    G = _line(4, passable={"r0-r1", "r1-r2", "r2-r3"})
    J = justified_graph(G, "r1")
    assert J.graph["root"] == "r1"
    assert J.nodes["r1"]["depth"] == 0
    assert J.nodes["r0"]["depth"] == 1
    assert J.nodes["r2"]["depth"] == 1
    assert J.nodes["r3"]["depth"] == 2
    # keeps all passable edges among reachable nodes (not a spanning tree here it is a path)
    assert {frozenset(e) for e in J.edges} == {
        frozenset(("r0", "r1")), frozenset(("r1", "r2")), frozenset(("r2", "r3")),
    }


def test_justified_graph_excludes_unreachable_nodes():
    G = _line(4, passable={"r0-r1"})     # only r0-r1 open
    J = justified_graph(G, "r0")
    assert set(J.nodes) == {"r0", "r1"}


# ---------------------------------------------------------------- mean depth / RA
def test_mean_depth_exact_fraction():
    G = _line(4, passable={"r0-r1", "r1-r2", "r2-r3"})
    md = mean_depth(G, "r0")             # depths 1,2,3 over 3 others
    assert md == Fraction(6, 3) == Fraction(2)
    assert isinstance(md, Fraction)
    assert not isinstance(md, float)
    # middle node r1: depths 1(r0),1(r2),2(r3) = 4 / 3
    assert mean_depth(G, "r1") == Fraction(4, 3)


def test_mean_depth_none_for_isolated_root():
    G = _line(2, passable=set())         # no openings at all
    assert mean_depth(G, "r0") is None   # reaches no other node


def test_relative_asymmetry_formula_exact():
    # P4: for the end node MD=2, n=4 → RA = 2(2-1)/(4-2) = 1 (exact).
    G = _line(4, passable={"r0-r1", "r1-r2", "r2-r3"})
    ra = relativised_asymmetry(G)
    assert ra["r0"] == Fraction(1)
    assert isinstance(ra["r0"], Fraction)
    # middle node: MD=4/3 → RA = 2(4/3 - 1)/(4-2) = 2*(1/3)/2 = 1/3
    assert ra["r1"] == Fraction(1, 3)


def test_relative_asymmetry_none_below_three_nodes():
    G = _line(2, passable={"r0-r1"})     # n = 2 < 3
    ra = relativised_asymmetry(G)
    assert ra["r0"] is None and ra["r1"] is None


# ---------------------------------------------------------------- D_n / RRA / integration
def test_d_value_matches_hillier_hanson_formula():
    for n in (3, 4, 5, 10):
        expected = 2.0 * (n * (log2((n + 2) / 3.0) - 1.0) + 1.0) / ((n - 1) * (n - 2))
        assert d_value(n) == pytest.approx(expected)
    # D_4 = 1/3 (the closed-form value used to sanity-check RRA below)
    assert d_value(4) == pytest.approx(1.0 / 3.0)


def test_d_value_requires_three_nodes():
    with pytest.raises(ValueError):
        d_value(2)


def test_integration_is_ra_over_d_value():
    G = _line(4, passable={"r0-r1", "r1-r2", "r2-r3"})
    integ = integration(G)
    ra = relativised_asymmetry(G)
    # RRA = RA / D_4 ; end node RA=1, D_4=1/3 → RRA = 3.0
    assert integ["r0"] == pytest.approx(float(ra["r0"]) / d_value(4))
    assert integ["r0"] == pytest.approx(3.0)
    # NOT raw closeness: networkx closeness of an end node of P4 is 3/6 = 0.5, ≠ 3.0
    close = nx.closeness_centrality(nx.path_graph(["r0", "r1", "r2", "r3"]))
    assert integ["r0"] != pytest.approx(close["r0"])


def test_integration_none_below_three_nodes():
    G = _line(2, passable={"r0-r1"})
    assert integration(G)["r0"] is None


# ---------------------------------------------------------------- choice = betweenness
def test_choice_equals_networkx_betweenness_on_passable_view():
    G = _line(4, passable={"r0-r1", "r1-r2", "r2-r3"})
    ch = choice(G)
    ref = nx.betweenness_centrality(nx.path_graph(["r0", "r1", "r2", "r3"]))
    for node in G.nodes:
        assert ch[node] == pytest.approx(ref[node])


def test_choice_ignores_walled_contacts():
    # r1-r2 walled: no route crosses it, so r1/r2 are not on any through-route.
    G = _line(4, passable={"r0-r1", "r2-r3"})
    ch = choice(G)
    # graph splits into {r0,r1} and {r2,r3}: every betweenness is 0 (no node between others)
    assert all(v == pytest.approx(0.0) for v in ch.values())
