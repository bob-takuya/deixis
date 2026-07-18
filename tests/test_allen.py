"""Tests for the Allen interval algebra (deixis.solver.allen).

Verifies: the 13 base relations, name<->mask helpers, converse involution and its
self/pair structure, that eq is the composition identity, converse-composition
contains eq, several known composition-table entries (cross-checked against Allen
1983 / Alspaugh Table 4a), and that a contradictory network is refuted by
path-consistency. Completeness is NOT asserted (path-consistency is only a sound
refutation procedure on the full algebra)."""
from __future__ import annotations

import itertools

import pytest

from deixis.solver import allen
from deixis.solver.allen import (
    BASE,
    UNIVERSAL,
    EMPTY,
    bit,
    mask,
    names,
    converse,
    compose,
    base_compose,
    is_base,
    AllenNetwork,
    path_consistency,
)


# ---------------------------------------------------------------- basics
def test_thirteen_base_relations():
    assert len(BASE) == 13
    assert len(set(BASE)) == 13
    assert set(BASE) == {
        "b", "bi", "m", "mi", "o", "oi", "s", "si", "d", "di", "f", "fi", "eq",
    }


def test_universal_and_empty():
    assert UNIVERSAL == (1 << 13) - 1
    assert EMPTY == 0
    assert mask() == EMPTY
    assert mask(*BASE) == UNIVERSAL
    assert names(UNIVERSAL) == list(BASE)
    assert names(EMPTY) == []


def test_bit_mask_names_roundtrip():
    for i, name in enumerate(BASE):
        assert bit(name) == 1 << i
        assert names(bit(name)) == [name]
    m = mask("b", "eq", "di")
    assert set(names(m)) == {"b", "eq", "di"}
    # names preserves canonical BASE order
    assert names(m) == [n for n in BASE if n in {"b", "eq", "di"}]


def test_bit_unknown_raises():
    with pytest.raises(ValueError):
        bit("nope")


def test_is_base():
    for name in BASE:
        assert is_base(bit(name))
    assert not is_base(EMPTY)
    assert not is_base(mask("b", "m"))
    assert not is_base(UNIVERSAL)


# ---------------------------------------------------------------- converse
def test_converse_is_involution():
    for m in range(UNIVERSAL + 1):
        assert converse(converse(m)) == m


def test_converse_pairs_and_selfconverse():
    assert converse(bit("eq")) == bit("eq")
    pairs = [("b", "bi"), ("m", "mi"), ("o", "oi"),
             ("s", "si"), ("d", "di"), ("f", "fi")]
    for a, b in pairs:
        assert converse(bit(a)) == bit(b)
        assert converse(bit(b)) == bit(a)


def test_converse_distributes_over_union():
    assert converse(mask("b", "m", "d")) == mask("bi", "mi", "di")


# ---------------------------------------------------------------- eq is identity
def test_eq_is_composition_identity():
    eq = bit("eq")
    for name in BASE:
        b = bit(name)
        assert compose(eq, b) == b
        assert compose(b, eq) == b
    # holds for arbitrary masks too
    for m in (EMPTY, UNIVERSAL, mask("b", "oi"), mask("s", "si", "f")):
        assert compose(eq, m) == m
        assert compose(m, eq) == m


def test_empty_composition_absorbs():
    assert compose(EMPTY, UNIVERSAL) == EMPTY
    assert compose(UNIVERSAL, EMPTY) == EMPTY
    assert compose(bit("b"), EMPTY) == EMPTY


# ---------------------------------------------------------------- converse-composition ⊇ eq
def test_compose_relation_with_converse_contains_eq():
    eq = bit("eq")
    for name in BASE:
        b = bit(name)
        assert compose(b, converse(b)) & eq, (
            f"compose({name}, converse({name})) must contain eq"
        )


# ---------------------------------------------------------------- known table entries
def test_known_composition_entries():
    b, bi = bit("b"), bit("bi")
    m, mi = bit("m"), bit("mi")
    o, oi = bit("o"), bit("oi")
    s, si = bit("s"), bit("si")
    d, di = bit("d"), bit("di")
    f, fi = bit("f"), bit("fi")

    # before ∘ before = before (transitive)
    assert base_compose("b", "b") == b
    # before ∘ after = universal (no information)
    assert base_compose("b", "bi") == UNIVERSAL
    # meets ∘ meets = before
    assert base_compose("m", "m") == b
    # meets ∘ met-by = {fi, eq, f}  (X.end == Y.start == Z.end -> shared end)
    assert base_compose("m", "mi") == mask("fi", "eq", "f")
    # during ∘ during = during (transitive)
    assert base_compose("d", "d") == d
    # contains ∘ contains = contains (transitive)
    assert base_compose("di", "di") == di
    # starts ∘ started-by = {s, eq, si}
    assert base_compose("s", "si") == mask("s", "eq", "si")
    # overlaps ∘ overlapped-by = "concur"
    assert base_compose("o", "oi") == mask(
        "o", "fi", "di", "s", "eq", "si", "d", "f", "oi"
    )
    # before ∘ during = {b, m, o, s, d}
    assert base_compose("b", "d") == mask("b", "m", "o", "s", "d")
    # after ∘ before = universal
    assert base_compose("bi", "b") == UNIVERSAL
    # finishes ∘ finished-by = {fi, eq, f}
    assert base_compose("f", "fi") == mask("fi", "eq", "f")


def test_compose_symmetry_law():
    # Fundamental relation-algebra law: converse(a ∘ b) == converse(b) ∘ converse(a).
    for a, b in itertools.product(BASE, repeat=2):
        ab = base_compose(a, b)
        rhs = compose(converse(bit(b)), converse(bit(a)))
        assert converse(ab) == rhs, f"symmetry law fails for {a} ∘ {b}"


def test_compose_union_is_union_of_base_compositions():
    a = mask("b", "m")
    bb = mask("d", "eq")
    expected = 0
    for x in ("b", "m"):
        for y in ("d", "eq"):
            expected |= base_compose(x, y)
    assert compose(a, bb) == expected


# ---------------------------------------------------------------- network / path consistency
def test_network_from_edges_and_relation():
    net = AllenNetwork.from_edges(
        ["A", "B", "C"],
        [("A", "B", bit("b")), ("B", "C", bit("b"))],
    )
    assert net.relation("A", "B") == bit("b")
    # converse derived automatically
    assert net.relation("B", "A") == bit("bi")
    # unconstrained pair -> universal
    assert net.relation("A", "C") == UNIVERSAL
    # diagonal -> eq
    assert net.relation("A", "A") == bit("eq")


def test_path_consistency_infers_before_chain():
    # A before B, B before C  =>  closure tightens A?C to before.
    net = AllenNetwork.from_edges(
        ["A", "B", "C"],
        [("A", "B", bit("b")), ("B", "C", bit("b"))],
    )
    status, closed = path_consistency(net)
    assert status == "path_consistent"
    assert closed.relation("A", "C") == bit("b")


def test_path_consistency_detects_contradiction():
    # A before B, B before C, but C before A -> unsatisfiable cycle.
    net = AllenNetwork.from_edges(
        ["A", "B", "C"],
        [
            ("A", "B", bit("b")),
            ("B", "C", bit("b")),
            ("C", "A", bit("b")),
        ],
    )
    status, _ = path_consistency(net)
    assert status == "inconsistent"


def test_path_consistency_direct_empty_edge_is_inconsistent():
    net = AllenNetwork.from_edges(
        ["A", "B"],
        [("A", "B", EMPTY)],
    )
    status, _ = path_consistency(net)
    assert status == "inconsistent"


def test_from_edges_bad_self_edge_raises():
    with pytest.raises(ValueError):
        AllenNetwork.from_edges(["A"], [("A", "A", bit("b"))])


def test_from_edges_reconciles_opposite_directions():
    # (A,B)=b and (B,A)=b  ->  b & converse(b) = b & bi = EMPTY
    net = AllenNetwork.from_edges(
        ["A", "B"],
        [("A", "B", bit("b")), ("B", "A", bit("b"))],
    )
    assert net.relation("A", "B") == EMPTY


def test_all_base_pairs_have_nonempty_composition():
    # Every base∘base entry in Allen's table is non-empty (jointly exhaustive).
    for a, b in itertools.product(BASE, repeat=2):
        assert base_compose(a, b) != EMPTY
