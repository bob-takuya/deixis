"""Tests for RCC-8 path-consistency and tractable-fragment reasoning.

Covers: the algebraic-closure fixpoint (triangle refinement, converse maintenance),
sound inconsistency detection, the three maximal tractable subclasses ^H8/C8/Q8
(cardinalities regression-checked against Renz & Nebel 1999), fragment-based
decidability, the honest 'unknown' verdict, and deletion-minimal unsat cores.
All arithmetic is exact integer bit-mask work (no floats)."""
import pytest

from deixis.core.rcc8 import (
    BASE, UNIVERSAL, EMPTY, bit, mask, converse, compose, names,
)
from deixis.solver.pathconsistency import (
    RCCNetwork,
    H8, C8, Q8, FRAGMENTS,
    in_np8, in_H8, in_C8, in_Q8,
    fragment_of_mask, fragment_membership, decides_satisfiability,
    path_consistency, unsat_core,
)

DC = bit("DC"); EC = bit("EC"); PO = bit("PO")
TPP = bit("TPP"); NTPP = bit("NTPP"); TPPi = bit("TPPi"); NTPPi = bit("NTPPi"); EQ = bit("EQ")
ALL_BASE = [bit(n) for n in BASE]


# ============================================================ fragment definitions
def test_fragment_cardinalities_match_renz_nebel():
    """|NP8|=76, |^H8|=148, |C8|=158, |Q8|=160 (Renz & Nebel 1999; Renz 1999)."""
    np8 = [m for m in range(256) if in_np8(m)]
    h8 = [m for m in range(256) if in_H8(m)]
    c8 = [m for m in range(256) if in_C8(m)]
    q8 = [m for m in range(256) if in_Q8(m)]
    assert len(np8) == 76
    assert len(h8) == 148
    assert len(c8) == 158
    assert len(q8) == 160


def test_H8_union_C8_is_complement_of_NP8():
    complement = {m for m in range(256) if not in_np8(m)}
    h8 = {m for m in range(256) if in_H8(m)}
    c8 = {m for m in range(256) if in_C8(m)}
    q8 = {m for m in range(256) if in_Q8(m)}
    assert h8 | c8 == complement
    assert len(complement) == 180
    # every Q8 relation is covered by H8 or C8 (Renz 1999)
    assert q8 <= (h8 | c8)


def test_base_universal_empty_in_all_fragments():
    for b in ALL_BASE:
        assert in_H8(b) and in_C8(b) and in_Q8(b)
        assert not in_np8(b)
    for m in (UNIVERSAL, EMPTY):
        assert in_H8(m) and in_C8(m) and in_Q8(m)


def test_fragments_and_np8_partition_disjointly():
    # NP8 is exactly the relations in NO tractable subclass.
    for m in range(256):
        in_any = in_H8(m) or in_C8(m) or in_Q8(m)
        assert in_np8(m) == (not in_any)
        assert set(fragment_of_mask(m)) == {
            n for n, p in ((H8, in_H8), (C8, in_C8), (Q8, in_Q8)) if p(m)
        }


def test_known_np8_relation_is_intractable():
    # {TPP,TPPi}: PO absent, has a (N)TPP and a (N)TPPi -> NP8, in no fragment.
    r = mask("TPP", "TPPi")
    assert in_np8(r)
    assert fragment_of_mask(r) == frozenset()
    # one of the four explicit NP8 relations
    assert in_np8(mask("EC", "NTPP", "EQ"))


def test_mask_range_validation():
    for fn in (in_np8, in_H8, in_C8, in_Q8):
        with pytest.raises(ValueError):
            fn(256)
        with pytest.raises(ValueError):
            fn(-1)


# ============================================================ network construction
def test_converse_completion_and_defaults():
    net = RCCNetwork.from_edges(["a", "b", "c"], {("a", "b"): TPP})
    assert net.relation("a", "b") == TPP
    assert net.relation("b", "a") == converse(TPP) == TPPi
    assert net.relation("a", "a") == EQ            # diagonal identity
    assert net.relation("a", "c") == UNIVERSAL     # unconstrained
    assert set(net.variables) == {"a", "b", "c"}


def test_opposite_edges_reconciled_by_intersection():
    # (a,b)=NTPP and (b,a)=NTPPi (converse) are consistent -> NTPP survives.
    net = RCCNetwork.from_edges(["a", "b"], [("a", "b", mask("NTPP", "TPP")),
                                             ("b", "a", NTPPi)])
    # converse(NTPPi)=NTPP, intersect with {TPP,NTPP} -> NTPP
    assert net.relation("a", "b") == NTPP


def test_self_edge_must_be_exactly_eq():
    with pytest.raises(ValueError):
        RCCNetwork.from_edges(["a"], {("a", "a"): DC})
    # a contradictory EMPTY self-edge must NOT be silently dropped
    with pytest.raises(ValueError):
        RCCNetwork.from_edges(["a"], {("a", "a"): EMPTY})
    # EQ self-edge is tolerated (identity is implicit)
    net = RCCNetwork.from_edges(["a"], {("a", "a"): EQ})
    assert net.relation("a", "a") == EQ


# ============================================================ path consistency
def test_triangle_transitivity_refinement():
    # a NTPP b, b NTPP c  =>  a NTPP c  (compose(NTPP,NTPP)=NTPP).
    net = RCCNetwork.from_edges(["a", "b", "c"],
                                {("a", "b"): NTPP, ("b", "c"): NTPP})
    status, closed = path_consistency(net)
    assert status == "consistent"
    assert closed.relation("a", "c") == NTPP
    assert closed.relation("c", "a") == NTPPi


def test_triangle_composition_matches_table():
    # a DC b, b NTPP c  =>  a-c refined to compose(DC,NTPP).
    net = RCCNetwork.from_edges(["a", "b", "c"],
                                {("a", "b"): DC, ("b", "c"): NTPP})
    status, closed = path_consistency(net)
    assert status == "consistent"
    assert closed.relation("a", "c") == compose(DC, NTPP)
    assert closed.relation("a", "c") == mask("DC", "EC", "PO", "TPP", "NTPP")


def test_unit_eq_network_consistent():
    net = RCCNetwork.from_edges(["a", "b"], {("a", "b"): EQ})
    status, closed = path_consistency(net)
    assert status == "consistent"
    assert closed.relation("a", "b") == EQ
    assert decides_satisfiability(net)


def test_inconsistent_network_detected():
    # a EQ b, b EQ c, a DC c  =>  EQ forces a=c, contradicting DC.
    net = RCCNetwork.from_edges(["a", "b", "c"],
                                {("a", "b"): EQ, ("b", "c"): EQ, ("a", "c"): DC})
    status, closed = path_consistency(net)
    assert status == "inconsistent"
    # some pair collapsed to EMPTY
    assert any(closed.relation(i, j) == EMPTY
               for i in closed.variables for j in closed.variables if i != j)


def test_already_empty_edge_is_inconsistent():
    net = RCCNetwork.from_edges(["a", "b"], {("a", "b"): EMPTY})
    status, _ = path_consistency(net)
    assert status == "inconsistent"


def test_closure_is_idempotent():
    net = RCCNetwork.from_edges(["a", "b", "c"],
                                {("a", "b"): NTPP, ("b", "c"): NTPP})
    _, closed = path_consistency(net)
    status2, closed2 = path_consistency(closed)
    assert status2 == "consistent"
    for i in closed.variables:
        for j in closed.variables:
            assert closed.relation(i, j) == closed2.relation(i, j)


# ============================================================ fragment membership / decidability
def test_fragment_membership_base_network_is_H8():
    net = RCCNetwork.from_edges(["a", "b", "c"],
                                {("a", "b"): DC, ("b", "c"): NTPP, ("a", "c"): PO})
    assert fragment_membership(net) == H8
    assert decides_satisfiability(net)


def test_np8_network_outside_all_fragments_is_unknown():
    # single edge with an NP8 relation, path-consistent but undecidable here.
    net = RCCNetwork.from_edges(["a", "b"], {("a", "b"): mask("TPP", "TPPi")})
    assert fragment_membership(net) is None
    assert not decides_satisfiability(net)
    status, _ = path_consistency(net)
    assert status == "unknown"


def test_np8_edge_but_actually_inconsistent_reports_inconsistent():
    # inconsistency is sound even when an NP8 (non-fragment) edge is present.
    npr = mask("TPP", "TPPi")  # NP8, in no tractable fragment
    net = RCCNetwork.from_edges(["a", "b", "c"],
                                {("a", "b"): EQ, ("b", "c"): EQ, ("a", "c"): DC,
                                 ("b", "c"): npr})
    # note: ("b","c") given twice above collapses by intersection; build explicitly instead
    net = RCCNetwork.from_edges(
        ["a", "b", "c", "d"],
        [("a", "b", EQ), ("b", "c", EQ), ("a", "c", DC), ("c", "d", npr)],
    )
    assert fragment_membership(net) is None      # npr keeps it out of every fragment
    status, _ = path_consistency(net)
    assert status == "inconsistent"              # but the a,b,c contradiction still fires


def test_membership_prefers_H8_then_C8_then_Q8():
    # relation in C8/Q8 but not H8: {EQ, NTPP} (EQ,NTPP present, TPP absent) -> removed from H8.
    r = mask("EQ", "NTPP")
    assert not in_H8(r)
    assert in_C8(r) or in_Q8(r)  # covered by H8 U C8, and not H8 => in C8
    assert in_C8(r)
    net = RCCNetwork.from_edges(["a", "b"], {("a", "b"): r})
    assert fragment_membership(net) == C8


# ============================================================ unsat core
def test_unsat_core_all_edges_needed():
    net = RCCNetwork.from_edges(["a", "b", "c"],
                                {("a", "b"): EQ, ("b", "c"): EQ, ("a", "c"): DC})
    core = unsat_core(net)
    assert core == {("a", "b"), ("b", "c"), ("a", "c")}
    # every core edge is genuinely needed: dropping it restores consistency
    for drop in list(core):
        remaining = [(i, j, net.relation(i, j)) for (i, j) in core if (i, j) != drop]
        sub = RCCNetwork.from_edges(net.variables, remaining)
        status, _ = path_consistency(sub)
        assert status != "inconsistent"


def test_unsat_core_excludes_irrelevant_edges():
    # d is attached by an irrelevant DC edge; the contradiction is among a,b,c.
    net = RCCNetwork.from_edges(["a", "b", "c", "d"],
                                {("a", "b"): EQ, ("b", "c"): EQ, ("a", "c"): DC,
                                 ("a", "d"): DC})
    core = unsat_core(net)
    assert ("a", "d") not in core
    assert core == {("a", "b"), ("b", "c"), ("a", "c")}


def test_unsat_core_empty_when_consistent():
    net = RCCNetwork.from_edges(["a", "b", "c"],
                                {("a", "b"): NTPP, ("b", "c"): NTPP})
    assert unsat_core(net) == set()


def test_unsat_core_is_irreducible():
    net = RCCNetwork.from_edges(["a", "b", "c", "d"],
                                {("a", "b"): EQ, ("b", "c"): EQ, ("a", "c"): DC,
                                 ("a", "d"): DC, ("b", "d"): PO})
    core = unsat_core(net)
    # the core itself is inconsistent
    core_net = RCCNetwork.from_edges(net.variables,
                                     [(i, j, net.relation(i, j)) for (i, j) in core])
    assert path_consistency(core_net)[0] == "inconsistent"
    # and minimal: removing any edge makes it path-consistent (non-inconsistent)
    for drop in list(core):
        sub = RCCNetwork.from_edges(
            net.variables,
            [(i, j, net.relation(i, j)) for (i, j) in core if (i, j) != drop],
        )
        assert path_consistency(sub)[0] != "inconsistent"
