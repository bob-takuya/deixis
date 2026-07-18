"""Tests for the RCC-8 relation algebra (deixis.core.rcc8).

The composition table is asserted against the standard theorem-prover-verified
Randell-Cui-Cohn table via algebraic laws (converse involution, EQ identity,
Peircean converse law, r∘r^-1 contains EQ) plus explicit spot-checks of known
entries. All arithmetic is exact integer bit-mask work (no floats)."""
from itertools import product

import pytest

from deixis.core import rcc8
from deixis.core.rcc8 import (
    BASE, UNIVERSAL, EMPTY,
    bit, mask, names, converse, compose, base_compose, is_base,
)

ALL_BASE_MASKS = [bit(n) for n in BASE]


# ---------------------------------------------------------------- name <-> mask
def test_base_order_and_indices():
    assert BASE == ("DC", "EC", "PO", "TPP", "NTPP", "TPPi", "NTPPi", "EQ")
    assert bit("DC") == 1
    assert bit("EQ") == 1 << 7
    assert UNIVERSAL == 0xFF
    assert EMPTY == 0


def test_mask_and_names_roundtrip():
    for subset_bits in range(256):
        assert mask(*names(subset_bits)) == subset_bits
    assert mask() == EMPTY
    assert mask("DC", "EC") == 0b11
    assert names(0) == []
    assert names(UNIVERSAL) == list(BASE)


def test_bit_unknown_name_raises():
    with pytest.raises(ValueError):
        bit("XX")
    with pytest.raises(ValueError):
        names(256)


# ---------------------------------------------------------------- converse
def test_converse_fixed_points_and_swaps():
    for n in ("DC", "EC", "PO", "EQ"):
        assert converse(bit(n)) == bit(n)
    assert converse(bit("TPP")) == bit("TPPi")
    assert converse(bit("TPPi")) == bit("TPP")
    assert converse(bit("NTPP")) == bit("NTPPi")
    assert converse(bit("NTPPi")) == bit("NTPP")


def test_converse_is_involution():
    for m in range(256):
        assert converse(converse(m)) == m


def test_converse_distributes_over_union():
    assert converse(mask("TPP", "NTPP")) == mask("TPPi", "NTPPi")
    assert converse(UNIVERSAL) == UNIVERSAL
    assert converse(EMPTY) == EMPTY


# ---------------------------------------------------------------- EQ identity
def test_eq_is_composition_identity():
    eq = bit("EQ")
    for m in ALL_BASE_MASKS:
        assert compose(m, eq) == m
        assert compose(eq, m) == m
    # also on disjunctions
    for m in (mask("DC", "PO"), mask("TPP", "NTPP", "EQ"), UNIVERSAL):
        assert compose(m, eq) == m
        assert compose(eq, m) == m


# ---------------------------------------------------------------- r ∘ r^-1 ⊇ EQ
def test_compose_relation_with_converse_contains_eq():
    eq = bit("EQ")
    for n in BASE:
        r = bit(n)
        assert compose(r, converse(r)) & eq, f"{n} ∘ {n}^-1 must contain EQ"


# ---------------------------------------------------------------- EMPTY behaviour
def test_empty_composition_is_empty():
    for m in ALL_BASE_MASKS + [UNIVERSAL]:
        assert compose(EMPTY, m) == EMPTY
        assert compose(m, EMPTY) == EMPTY


# ---------------------------------------------------------------- Peircean law
def test_peircean_converse_law():
    # converse(a ∘ b) == converse(b) ∘ converse(a) for all base pairs.
    for a, b in product(BASE, repeat=2):
        lhs = converse(compose(bit(a), bit(b)))
        rhs = compose(converse(bit(b)), converse(bit(a)))
        assert lhs == rhs, f"Peirce law fails for {a} ∘ {b}"


# ---------------------------------------------------------------- distributivity
def test_compose_distributes_over_union():
    # compose is union over base pairs -> distributes over disjunction masks.
    for _ in range(1):
        a1, a2 = mask("DC", "TPP"), mask("EC", "NTPPi")
        b = mask("PO", "TPPi")
        assert compose(a1 | a2, b) == compose(a1, b) | compose(a2, b)
        assert compose(a1, a2 | b) == compose(a1, a2) | compose(a1, b)


# ---------------------------------------------------------------- monotonicity
def test_compose_monotone_and_universal_absorbs():
    # a ⊆ a' ==> compose(a,b) ⊆ compose(a',b)
    a = bit("TPP")
    a_big = mask("TPP", "NTPP")
    b = bit("PO")
    assert compose(a, b) | compose(a_big, b) == compose(a_big, b)
    # composing anything non-empty on both sides with the universal produces
    # something inside the universal (trivially) — check a few reach UNIVERSAL.
    assert compose(bit("PO"), bit("PO")) == UNIVERSAL
    assert compose(bit("NTPP"), bit("NTPPi")) == UNIVERSAL
    assert compose(bit("DC"), bit("DC")) == UNIVERSAL


# ---------------------------------------------------------------- table symmetry
def test_composition_table_diagonal_symmetry_of_selfconverse():
    # For self-converse a and b, compose(a,b) must be self-converse (closed under converse),
    # by the Peircean law converse(a∘b)=converse(b)∘converse(a)=b∘a ... check the
    # simpler consequence: a∘a is self-converse for self-converse a.
    for n in ("DC", "EC", "PO", "EQ"):
        r = bit(n)
        assert converse(compose(r, r)) == compose(r, r)


# ---------------------------------------------------------------- explicit spot-checks
# Known entries from the standard RCC-8 composition table (Randell-Cui-Cohn 1992).
EXPECTED = {
    ("DC", "DC"): tuple(BASE),
    ("DC", "TPPi"): ("DC",),
    ("DC", "NTPPi"): ("DC",),
    ("DC", "EC"): ("DC", "EC", "PO", "TPP", "NTPP"),
    ("EC", "EC"): ("DC", "EC", "PO", "TPP", "TPPi", "EQ"),
    ("EC", "NTPP"): ("PO", "TPP", "NTPP"),
    ("EC", "NTPPi"): ("DC",),
    ("EC", "TPPi"): ("DC", "EC"),
    ("PO", "PO"): tuple(BASE),
    ("PO", "TPP"): ("PO", "TPP", "NTPP"),
    ("TPP", "TPP"): ("TPP", "NTPP"),
    ("TPP", "NTPP"): ("NTPP",),
    ("TPP", "TPPi"): ("DC", "EC", "PO", "TPP", "TPPi", "EQ"),
    ("TPP", "NTPPi"): ("DC", "EC", "PO", "TPPi", "NTPPi"),
    ("NTPP", "NTPP"): ("NTPP",),
    ("NTPP", "NTPPi"): tuple(BASE),
    ("NTPP", "EC"): ("DC",),
    ("TPPi", "TPP"): ("PO", "TPP", "TPPi", "EQ"),
    ("TPPi", "TPPi"): ("TPPi", "NTPPi"),
    ("TPPi", "NTPPi"): ("NTPPi",),
    ("TPPi", "EC"): ("EC", "PO", "TPPi", "NTPPi"),
    ("NTPPi", "NTPP"): ("PO", "TPP", "NTPP", "TPPi", "NTPPi", "EQ"),
    ("NTPPi", "NTPPi"): ("NTPPi",),
    ("NTPPi", "EC"): ("PO", "TPPi", "NTPPi"),
    ("NTPPi", "TPP"): ("PO", "TPPi", "NTPPi"),
}


@pytest.mark.parametrize("pair,expected", list(EXPECTED.items()))
def test_composition_table_spot_checks(pair, expected):
    a, b = pair
    assert compose(bit(a), bit(b)) == mask(*expected)
    assert base_compose(a, b) == mask(*expected)
    # every tabulated entry is a valid mask within range
    assert 0 < compose(bit(a), bit(b)) <= UNIVERSAL


# ---------------------------------------------------------------- full-table sanity
def test_every_base_composition_nonempty_and_in_range():
    # RCC-8 base relations are JEPD over a connected space: no base composition is empty.
    for a, b in product(BASE, repeat=2):
        m = compose(bit(a), bit(b))
        assert 0 < m <= UNIVERSAL, f"{a} ∘ {b} out of range"


def test_is_base_helper():
    for n in BASE:
        assert is_base(bit(n))
    assert not is_base(EMPTY)
    assert not is_base(mask("DC", "EC"))
    assert not is_base(UNIVERSAL)
