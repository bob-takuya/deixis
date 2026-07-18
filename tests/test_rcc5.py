"""Tests for the RCC-5 relation algebra and the RCC-8 -> RCC-5 quotient (deixis.core.rcc5).

The RCC-5 composition table is cross-checked two ways: (1) algebraic laws (converse
involution, EQ identity, Peircean converse law, r o r^-1 contains EQ) and (2) an *exact*
match against the quotient of the theorem-prover-verified RCC-8 table under
``rcc8_to_rcc5`` --- the property that makes the quotient a sound coarsening. All
arithmetic is exact integer bit-mask work (no floats)."""
from itertools import product

import pytest

from deixis.core import rcc5, rcc8
from deixis.core.rcc5 import (
    BASE, UNIVERSAL, EMPTY,
    bit, mask, names, converse, compose, base_compose, is_base,
    rcc8_to_rcc5,
)

ALL_BASE_MASKS = [bit(n) for n in BASE]

# preimage of each RCC-5 base under the quotient (fixed spec map)
_PREIMAGE = {
    "DR": ("DC", "EC"),
    "PO": ("PO",),
    "PP": ("TPP", "NTPP"),
    "PPi": ("TPPi", "NTPPi"),
    "EQ": ("EQ",),
}


# ---------------------------------------------------------------- name <-> mask
def test_base_order_and_indices():
    assert BASE == ("DR", "PO", "PP", "PPi", "EQ")
    assert bit("DR") == 1
    assert bit("EQ") == 1 << 4
    assert UNIVERSAL == 0x1F
    assert EMPTY == 0


def test_mask_and_names_roundtrip():
    for subset_bits in range(32):
        assert mask(*names(subset_bits)) == subset_bits
    assert mask() == EMPTY
    assert mask("DR", "PO") == 0b11
    assert names(0) == []
    assert names(UNIVERSAL) == list(BASE)


def test_bit_unknown_name_raises():
    with pytest.raises(ValueError):
        bit("XX")


def test_mask_range_checks():
    for fn in (names, converse):
        with pytest.raises(ValueError):
            fn(-1)
        with pytest.raises(ValueError):
            fn(UNIVERSAL + 1)
    with pytest.raises(ValueError):
        compose(UNIVERSAL + 1, 0)


def test_is_base():
    assert all(is_base(b) for b in ALL_BASE_MASKS)
    assert not is_base(EMPTY)
    assert not is_base(mask("DR", "PO"))
    assert not is_base(UNIVERSAL)


# ---------------------------------------------------------------- converse
def test_converse_fixed_and_swapped():
    for n in ("DR", "PO", "EQ"):
        assert converse(bit(n)) == bit(n)             # self-converse
    assert converse(bit("PP")) == bit("PPi")
    assert converse(bit("PPi")) == bit("PP")


def test_converse_involution():
    for m in range(32):
        assert converse(converse(m)) == m


def test_converse_distributes_over_union():
    for a, b in product(ALL_BASE_MASKS, repeat=2):
        assert converse(a | b) == converse(a) | converse(b)


# ---------------------------------------------------------------- composition
def test_eq_is_identity():
    eq = bit("EQ")
    for b in ALL_BASE_MASKS:
        assert compose(eq, b) == b
        assert compose(b, eq) == b


def test_compose_with_empty_is_empty():
    for b in ALL_BASE_MASKS + [UNIVERSAL]:
        assert compose(EMPTY, b) == EMPTY
        assert compose(b, EMPTY) == EMPTY


def test_self_composition_contains_eq():
    # r o r^-1 must always admit EQ (two regions can coincide).
    for b in ALL_BASE_MASKS:
        assert compose(b, converse(b)) & bit("EQ")


def test_peircean_converse_law():
    # converse(a o b) == converse(b) o converse(a)
    for a, b in product(ALL_BASE_MASKS, repeat=2):
        assert converse(compose(a, b)) == compose(converse(b), converse(a))


def test_composition_is_weak_union_of_bases():
    # compose(mask) == union of base compositions.
    for a, b in product(range(32), repeat=2):
        expect = 0
        for i in range(5):
            if not (a & (1 << i)):
                continue
            for j in range(5):
                if b & (1 << j):
                    expect |= base_compose(BASE[i], BASE[j])
        assert compose(a, b) == expect


def test_known_composition_entries():
    # spot-checks of the standard RCC-5 table.
    assert compose(bit("DR"), bit("DR")) == UNIVERSAL
    assert compose(bit("PO"), bit("PO")) == UNIVERSAL
    assert compose(bit("PP"), bit("PPi")) == UNIVERSAL
    assert names(compose(bit("DR"), bit("PP"))) == ["DR", "PO", "PP"]
    assert names(compose(bit("PP"), bit("DR"))) == ["DR"]
    assert names(compose(bit("PPi"), bit("PP"))) == ["PO", "PP", "PPi", "EQ"]
    assert names(compose(bit("PO"), bit("PP"))) == ["PO", "PP"]
    assert names(compose(bit("PPi"), bit("PO"))) == ["PO", "PPi"]


# ------------------------------------------------ RCC-8 -> RCC-5 quotient map
def test_quotient_maps_each_rcc8_base():
    expect = {
        "DC": "DR", "EC": "DR", "PO": "PO",
        "TPP": "PP", "NTPP": "PP", "TPPi": "PPi", "NTPPi": "PPi", "EQ": "EQ",
    }
    for n8, n5 in expect.items():
        assert names(rcc8_to_rcc5(rcc8.bit(n8))) == [n5]


def test_quotient_folds_disjunctions():
    # DC|EC (both -> DR) collapses to a single DR.
    assert names(rcc8_to_rcc5(rcc8.mask("DC", "EC"))) == ["DR"]
    # TPP|NTPP -> PP.
    assert names(rcc8_to_rcc5(rcc8.mask("TPP", "NTPP"))) == ["PP"]
    # a mix spanning several classes.
    assert names(rcc8_to_rcc5(rcc8.mask("DC", "PO", "NTPPi", "EQ"))) == ["DR", "PO", "PPi", "EQ"]
    # universal RCC-8 -> universal RCC-5.
    assert rcc8_to_rcc5(rcc8.UNIVERSAL) == UNIVERSAL


def test_quotient_empty_is_empty():
    # Sound coarsening at the mask level: a contradictory RCC-8 relation stays empty.
    assert rcc8_to_rcc5(rcc8.EMPTY) == EMPTY


def test_quotient_only_empty_maps_to_empty():
    # No non-empty RCC-8 mask ever collapses to EMPTY (so RCC-5 emptiness <=> RCC-8 emptiness).
    for m in range(1, 256):
        assert rcc8_to_rcc5(m) != EMPTY


def test_quotient_range_check():
    with pytest.raises(ValueError):
        rcc8_to_rcc5(256)
    with pytest.raises(ValueError):
        rcc8_to_rcc5(-1)


# ---------------------------- homomorphism / sound-coarsening property --------
# Independent transcription of the published RCC-5 weak composition table
# (Cohn, Bennett, Gooday & Gotts 1997, Table 2), NOT derived from this repo's rcc8.
# Rows = first relation (x,y), columns = second (y,z). "*" = universal.
_PUBLISHED_RCC5 = {
    "DR":  {"DR": "*",            "PO": "DR PO PP",   "PP": "DR PO PP", "PPi": "DR",             "EQ": "DR"},
    "PO":  {"DR": "DR PO PPi",    "PO": "*",          "PP": "PO PP",    "PPi": "DR PO PPi",      "EQ": "PO"},
    "PP":  {"DR": "DR",           "PO": "DR PO PP",   "PP": "PP",       "PPi": "*",              "EQ": "PP"},
    "PPi": {"DR": "DR PO PPi",    "PO": "PO PPi",     "PP": "PO PP PPi EQ", "PPi": "PPi",        "EQ": "PPi"},
    "EQ":  {"DR": "DR",           "PO": "PO",         "PP": "PP",       "PPi": "PPi",            "EQ": "EQ"},
}


def test_composition_matches_published_table():
    # Cross-check every entry against the independently transcribed published table.
    for a5 in BASE:
        for b5 in BASE:
            cell = _PUBLISHED_RCC5[a5][b5]
            expect = UNIVERSAL if cell == "*" else mask(*cell.split())
            assert base_compose(a5, b5) == expect, (a5, b5)


def test_rcc5_table_is_tightest_aggregate_of_rcc8():
    # The RCC-5 table also equals the union, over all RCC-8 base pairs in each
    # preimage, of the quotiented RCC-8 composition -- so it is the tightest sound
    # coarsening of the theorem-prover-verified RCC-8 table.
    for a5, b5 in product(BASE, repeat=2):
        agg = 0
        for a8 in _PREIMAGE[a5]:
            for b8 in _PREIMAGE[b5]:
                agg |= rcc8_to_rcc5(rcc8.base_compose(a8, b8))
        assert base_compose(a5, b5) == agg, (a5, b5, names(base_compose(a5, b5)), names(agg))


def test_composition_is_sound_coarsening_on_bases():
    # Per RCC-8 base pair the quotiented composition is a SUBSET of the coarse
    # composition (soundness); equality does NOT hold in general.
    q = {n8: names(rcc8_to_rcc5(rcc8.bit(n8)))[0] for n8 in rcc8.BASE}
    strict_seen = False
    for a8, b8 in product(rcc8.BASE, repeat=2):
        lhs = rcc8_to_rcc5(rcc8.base_compose(a8, b8))
        rhs = base_compose(q[a8], q[b8])
        assert lhs & rhs == lhs, (a8, b8, names(lhs), names(rhs))   # lhs subset of rhs
        if lhs != rhs:
            strict_seen = True
    assert strict_seen   # the coarsening genuinely loses information somewhere


def test_composition_is_sound_coarsening_on_all_masks():
    # Full soundness on masks: rcc8_to_rcc5(compose8(A,B)) is a subset of
    # compose5(q(A), q(B)) for all A, B --- so RCC-5 UNSAT implies RCC-8 UNSAT.
    for A, B in product(range(256), repeat=2):
        lhs = rcc8_to_rcc5(rcc8.compose(A, B))
        rhs = compose(rcc8_to_rcc5(A), rcc8_to_rcc5(B))
        assert lhs & rhs == lhs                       # lhs subset of rhs (sound over-approx)
        if rhs == EMPTY:                              # coarse contradiction ...
            assert lhs == EMPTY                       # ... implies fine contradiction


def test_quotient_commutes_with_converse():
    for m in range(256):
        assert rcc8_to_rcc5(rcc8.converse(m)) == converse(rcc8_to_rcc5(m))
