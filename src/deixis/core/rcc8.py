"""RCC-8 relation algebra (qualitative spatial reasoning).

The 8 jointly-exhaustive, pairwise-disjoint base relations of the Region Connection
Calculus (Randell, Cui & Cohn 1992):

    index  name    meaning
      0     DC      disconnected
      1     EC      externally connected (touch, boundaries share, interiors disjoint)
      2     PO      partially overlap (proper overlap)
      3     TPP     tangential proper part          (src inside dst, touching boundary)
      4     NTPP    non-tangential proper part      (src strictly inside dst)
      5     TPPi    converse of TPP                 (dst is a TPP of src)
      6     NTPPi   converse of NTPP
      7     EQ      equal

A *relation* between two regions in the presence of incomplete knowledge is a
DISJUNCTION of base relations, encoded here as an 8-bit integer mask
(bit i set  <=>  base relation ``BASE[i]`` is possible). UNIVERSAL = 0xFF (all 8),
EMPTY = 0 (the contradictory/impossible relation).

The composition table below is the standard, theorem-prover-verified RCC-8 weak
composition table (Randell-Cui-Cohn 1992; reproduced in Renz & Nebel, and on the
Region Connection Calculus Wikipedia article). ``compose(a, b)`` returns, for
masks a (relating x,y) and b (relating y,z), the tightest mask relating x,z that
is entailed --- the union over all base pairs of their tabulated composition.
"""
from __future__ import annotations

from typing import Iterable

# -- base relations, fixed index order -----------------------------------------
BASE: tuple[str, ...] = ("DC", "EC", "PO", "TPP", "NTPP", "TPPi", "NTPPi", "EQ")
_INDEX: dict[str, int] = {name: i for i, name in enumerate(BASE)}

UNIVERSAL: int = 0xFF          # all 8 base relations possible
EMPTY: int = 0                 # no base relation possible (impossible / contradiction)


# -- name <-> mask helpers -----------------------------------------------------
def bit(name: str) -> int:
    """Single-base-relation mask for ``name`` (e.g. bit('DC') == 1)."""
    try:
        return 1 << _INDEX[name]
    except KeyError:
        raise ValueError(f"unknown RCC-8 base relation: {name!r}") from None


def mask(*names: str) -> int:
    """Disjunction mask of the given base-relation names. mask() == EMPTY."""
    m = 0
    for n in names:
        m |= bit(n)
    return m


def names(m: int) -> list[str]:
    """Base-relation names present in mask ``m``, in canonical BASE order."""
    if m < 0 or m > UNIVERSAL:
        raise ValueError(f"mask out of range 0..255: {m}")
    return [name for i, name in enumerate(BASE) if m & (1 << i)]


# -- converse ------------------------------------------------------------------
# DC, EC, PO, EQ are self-converse; TPP<->TPPi, NTPP<->NTPPi.
_CONVERSE_NAME: dict[str, str] = {
    "DC": "DC", "EC": "EC", "PO": "PO", "EQ": "EQ",
    "TPP": "TPPi", "TPPi": "TPP",
    "NTPP": "NTPPi", "NTPPi": "NTPP",
}
# precomputed single-bit -> converse-bit mask
_CONVERSE_BIT: tuple[int, ...] = tuple(bit(_CONVERSE_NAME[name]) for name in BASE)


def converse(m: int) -> int:
    """Converse of relation mask ``m``: converse(x r y) is the mask for (y r^-1 x)."""
    if m < 0 or m > UNIVERSAL:
        raise ValueError(f"mask out of range 0..255: {m}")
    out = 0
    for i in range(8):
        if m & (1 << i):
            out |= _CONVERSE_BIT[i]
    return out


# -- composition table ---------------------------------------------------------
# _COMP[a][b] = base relations entailed by composing base a (x,y) with base b (y,z).
# Standard RCC-8 weak composition table (Randell-Cui-Cohn 1992). "*" is UNIVERSAL.
_ALL = BASE  # convenience alias for the universal entry

_COMP_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    "DC": {
        "DC":    _ALL,
        "EC":    ("DC", "EC", "PO", "TPP", "NTPP"),
        "PO":    ("DC", "EC", "PO", "TPP", "NTPP"),
        "TPP":   ("DC", "EC", "PO", "TPP", "NTPP"),
        "NTPP":  ("DC", "EC", "PO", "TPP", "NTPP"),
        "TPPi":  ("DC",),
        "NTPPi": ("DC",),
        "EQ":    ("DC",),
    },
    "EC": {
        "DC":    ("DC", "EC", "PO", "TPPi", "NTPPi"),
        "EC":    ("DC", "EC", "PO", "TPP", "TPPi", "EQ"),
        "PO":    ("DC", "EC", "PO", "TPP", "NTPP"),
        "TPP":   ("EC", "PO", "TPP", "NTPP"),
        "NTPP":  ("PO", "TPP", "NTPP"),
        "TPPi":  ("DC", "EC"),
        "NTPPi": ("DC",),
        "EQ":    ("EC",),
    },
    "PO": {
        "DC":    ("DC", "EC", "PO", "TPPi", "NTPPi"),
        "EC":    ("DC", "EC", "PO", "TPPi", "NTPPi"),
        "PO":    _ALL,
        "TPP":   ("PO", "TPP", "NTPP"),
        "NTPP":  ("PO", "TPP", "NTPP"),
        "TPPi":  ("DC", "EC", "PO", "TPPi", "NTPPi"),
        "NTPPi": ("DC", "EC", "PO", "TPPi", "NTPPi"),
        "EQ":    ("PO",),
    },
    "TPP": {
        "DC":    ("DC",),
        "EC":    ("DC", "EC"),
        "PO":    ("DC", "EC", "PO", "TPP", "NTPP"),
        "TPP":   ("TPP", "NTPP"),
        "NTPP":  ("NTPP",),
        "TPPi":  ("DC", "EC", "PO", "TPP", "TPPi", "EQ"),
        "NTPPi": ("DC", "EC", "PO", "TPPi", "NTPPi"),
        "EQ":    ("TPP",),
    },
    "NTPP": {
        "DC":    ("DC",),
        "EC":    ("DC",),
        "PO":    ("DC", "EC", "PO", "TPP", "NTPP"),
        "TPP":   ("NTPP",),
        "NTPP":  ("NTPP",),
        "TPPi":  ("DC", "EC", "PO", "TPP", "NTPP"),
        "NTPPi": _ALL,
        "EQ":    ("NTPP",),
    },
    "TPPi": {
        "DC":    ("DC", "EC", "PO", "TPPi", "NTPPi"),
        "EC":    ("EC", "PO", "TPPi", "NTPPi"),
        "PO":    ("PO", "TPPi", "NTPPi"),
        "TPP":   ("PO", "TPP", "TPPi", "EQ"),
        "NTPP":  ("PO", "TPP", "NTPP"),
        "TPPi":  ("TPPi", "NTPPi"),
        "NTPPi": ("NTPPi",),
        "EQ":    ("TPPi",),
    },
    "NTPPi": {
        "DC":    ("DC", "EC", "PO", "TPPi", "NTPPi"),
        "EC":    ("PO", "TPPi", "NTPPi"),
        "PO":    ("PO", "TPPi", "NTPPi"),
        "TPP":   ("PO", "TPPi", "NTPPi"),
        "NTPP":  ("PO", "TPP", "NTPP", "TPPi", "NTPPi", "EQ"),
        "TPPi":  ("NTPPi",),
        "NTPPi": ("NTPPi",),
        "EQ":    ("NTPPi",),
    },
    "EQ": {
        "DC":    ("DC",),
        "EC":    ("EC",),
        "PO":    ("PO",),
        "TPP":   ("TPP",),
        "NTPP":  ("NTPP",),
        "TPPi":  ("TPPi",),
        "NTPPi": ("NTPPi",),
        "EQ":    ("EQ",),
    },
}

# precompute the base-vs-base composition as masks, indexed by base index.
_COMP_BIT: tuple[tuple[int, ...], ...] = tuple(
    tuple(mask(*_COMP_NAMES[a][b]) for b in BASE) for a in BASE
)


def base_compose(a_name: str, b_name: str) -> int:
    """Composition mask of two *single* base relations (exposed for tests/inspection)."""
    return _COMP_BIT[_INDEX[a_name]][_INDEX[b_name]]


def compose(a_mask: int, b_mask: int) -> int:
    """Compose relation masks: given (x a_mask y) and (y b_mask z), return mask (x ? z).

    Weak composition = union over every base pair (a in a_mask, b in b_mask) of the
    tabulated base composition. Composing with EMPTY yields EMPTY (no consistent y)."""
    if a_mask < 0 or a_mask > UNIVERSAL or b_mask < 0 or b_mask > UNIVERSAL:
        raise ValueError(f"mask out of range 0..255: {a_mask}, {b_mask}")
    out = 0
    for i in range(8):
        if not (a_mask & (1 << i)):
            continue
        row = _COMP_BIT[i]
        for j in range(8):
            if b_mask & (1 << j):
                out |= row[j]
    return out


def intersect(*masks: int) -> int:
    """Bitwise-AND (conjunction) of relation masks."""
    out = UNIVERSAL
    for m in masks:
        out &= m
    return out


def union(*masks: int) -> int:
    """Bitwise-OR (disjunction) of relation masks."""
    out = 0
    for m in masks:
        out |= m
    return out


def is_base(m: int) -> bool:
    """True iff ``m`` is a single (definite) base relation."""
    return m != 0 and (m & (m - 1)) == 0


__all__ = [
    "BASE", "UNIVERSAL", "EMPTY",
    "bit", "mask", "names",
    "converse", "compose", "base_compose",
    "intersect", "union", "is_base",
]
