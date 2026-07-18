"""Tests for the research-only Dulmage-Mendelsohn structural diagnostic.

These assert the tool detects *structural* over/under-determination on the stage-B
equality subsystem and, crucially, that it stays honest: it does NOT flag a semantically
redundant (but structurally square) equality cycle as overconstrained, and it never
attributes blame / names a redundant constraint.
"""
from __future__ import annotations

import networkx as nx

from deixis.core import rcc8
from deixis.core.types import Region, RelationConstraint, RelSpec, Status
from deixis.diagnostics import dm


# ---------------------------------------------------------------- helpers

def _eq(src: str, dst: str, cid: str = "") -> RelationConstraint:
    return RelationConstraint(src=src, dst=dst, rcc8_mask=rcc8.bit("EQ"),
                              status=Status.DELEGATED, id=cid)


def _spec(regions, constraints) -> RelSpec:
    return RelSpec(regions=tuple(Region(r) for r in regions),
                   constraints=tuple(constraints))


# ---------------------------------------------------------------- graph construction

def test_graph_is_bipartite_with_expected_shape():
    spec = _spec(["A", "B"], [_eq("A", "B", "e0")])
    g = dm.constraint_variable_graph(spec, domain="aabb_2d")
    assert nx.is_bipartite(g)
    var = [n for n, d in g.nodes(data=True) if d["bipartite"] == 0]
    eqn = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
    # 2 regions x 2 sides x 2 axes = 8 coordinate variables; one equation per (side,axis).
    assert len(var) == 8
    assert len(eqn) == 4
    # every equation touches exactly two distinct variables
    for e in eqn:
        assert g.degree[e] == 2


def test_only_definite_eq_is_modeled():
    # DC, PO, and a disjunctive EQ|PO must all contribute nothing.
    dc = RelationConstraint(src="A", dst="B", rcc8_mask=rcc8.bit("DC"), id="dc")
    po = RelationConstraint(src="A", dst="B", rcc8_mask=rcc8.bit("PO"), id="po")
    eqpo = RelationConstraint(src="A", dst="B",
                              rcc8_mask=rcc8.mask("EQ", "PO"), id="eqpo")
    spec = _spec(["A", "B"], [dc, po, eqpo])
    g = dm.constraint_variable_graph(spec)
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0


def test_self_relation_is_skipped():
    spec = _spec(["A"], [_eq("A", "A", "self")])
    g = dm.constraint_variable_graph(spec)
    assert g.number_of_nodes() == 0


def test_domain_3d_has_three_axes():
    spec = _spec(["A", "B"], [_eq("A", "B", "e0")])
    g = dm.constraint_variable_graph(spec, domain="aabb_3d")
    eqn = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
    assert len(eqn) == 6  # 2 sides x 3 axes


# ---------------------------------------------------------------- underdetermined

def test_single_equality_is_underdetermined():
    # One EQ pins each coordinate pair by one relation but leaves the absolute DOF free:
    # structurally underdetermined, never overdetermined.
    spec = _spec(["A", "B"], [_eq("A", "B", "e0")])
    d = dm.diagnose(spec)
    assert d["underconstrained"] is True
    assert d["overconstrained"] is False
    assert d["blocks"]["underdetermined"]["variables"]
    assert d["blocks"]["overdetermined"]["equations"] == []


def test_open_chain_is_underdetermined():
    # A=B=C (a path, no closure): 3 vars, 2 eqs per scalar -> one free var.
    spec = _spec(["A", "B", "C"], [_eq("A", "B", "e0"), _eq("B", "C", "e1")])
    d = dm.diagnose(spec)
    assert d["underconstrained"] is True
    assert d["overconstrained"] is False


# ---------------------------------------------------------------- overdetermined

def test_triplicated_equality_is_overdetermined():
    # Three EQ constraints on the same pair -> 3 parallel equations on 2 vars per scalar,
    # so one equation cannot be matched: a structural surplus (overdetermined).
    spec = _spec(["A", "B"],
                 [_eq("A", "B", "e0"), _eq("A", "B", "e1"), _eq("A", "B", "e2")])
    d = dm.diagnose(spec)
    assert d["overconstrained"] is True
    over = d["blocks"]["overdetermined"]
    assert over["equations"]  # surplus equation(s) present
    # ...but the tool must NOT name *which* of the three is the redundant one.
    # It only reports the coupled block; the surplus is not uniquely attributed.
    assert len(over["equations"]) >= 1


# ---------------------------------------------------------------- honesty: structural only

def test_consistent_cycle_is_not_flagged_overconstrained():
    # A=B, B=C, C=A is algebraically rank-deficient (redundant) BUT structurally square:
    # 3 vars, 3 eqs forming a 6-cycle -> perfectly matchable -> well-determined.
    # DM must NOT flag it as overconstrained (it cannot see the numeric redundancy).
    spec = _spec(["A", "B", "C"],
                 [_eq("A", "B", "e0"), _eq("B", "C", "e1"), _eq("C", "A", "e2")])
    d = dm.diagnose(spec)
    assert d["overconstrained"] is False
    assert d["underconstrained"] is False
    assert d["blocks"]["welldetermined"]["variables"]
    assert d["blocks"]["welldetermined"]["equations"]


def test_report_makes_no_redundancy_or_contradiction_claim():
    # The diagnose contract exposes only structural keys — no "redundant", "remove",
    # "contradiction", "inconsistent", or "lock" verdicts.
    spec = _spec(["A", "B"],
                 [_eq("A", "B", "e0"), _eq("A", "B", "e1"), _eq("A", "B", "e2")])
    d = dm.diagnose(spec)
    assert set(d.keys()) == {"overconstrained", "underconstrained", "blocks"}
    forbidden = {"redundant", "remove", "contradiction", "inconsistent", "lock", "blame"}
    assert forbidden.isdisjoint(d["blocks"].keys())


# ---------------------------------------------------------------- decomposition invariants

def test_blocks_partition_all_nodes_and_over_under_disjoint():
    spec = _spec(["A", "B", "C", "D"],
                 [_eq("A", "B", "e0"), _eq("A", "B", "e1"),  # overdetermined region
                  _eq("C", "D", "e2")])                       # underdetermined region
    g = dm.constraint_variable_graph(spec)
    dec = dm.dm_decompose(g)
    all_nodes = set(dec["variables"]) | set(dec["equations"])
    over = set(dec["overdetermined"]["variables"]) | set(dec["overdetermined"]["equations"])
    well = set(dec["welldetermined"]["variables"]) | set(dec["welldetermined"]["equations"])
    under = set(dec["underdetermined"]["variables"]) | set(dec["underdetermined"]["equations"])
    # partition
    assert over | well | under == all_nodes
    assert over.isdisjoint(under)
    assert over.isdisjoint(well)
    assert well.isdisjoint(under)


def test_empty_spec_is_neither_over_nor_underconstrained():
    spec = _spec([], [])
    d = dm.diagnose(spec)
    assert d["overconstrained"] is False
    assert d["underconstrained"] is False
    assert d["blocks"]["variables"] == []
    assert d["blocks"]["equations"] == []


def test_matching_is_a_valid_matching():
    spec = _spec(["A", "B"], [_eq("A", "B", "e0")])
    g = dm.constraint_variable_graph(spec)
    dec = dm.dm_decompose(g)
    match = dec["matching"]
    # symmetric and each node used at most once
    for a, b in match.items():
        assert match[b] == a
        assert g.has_edge(a, b)
