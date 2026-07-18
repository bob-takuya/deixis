"""RCC-5 relation algebra + the RCC-8 -> RCC-5 quotient (coarse spatial reasoning).

The 5 jointly-exhaustive, pairwise-disjoint base relations of RCC-5, the coarse
sibling of RCC-8 that forgets the boundary-contact distinctions (it identifies
"disconnected" with "externally connected", and "tangential" with "non-tangential"
part-hood):

    index  name   meaning
      0     DR     discrete             (interiors disjoint; RCC-8 DC or EC)
      1     PO     partial overlap      (proper overlap)
      2     PP     proper part          (src strictly inside dst; RCC-8 TPP or NTPP)
      3     PPi    inverse proper part  (converse of PP)
      4     EQ     equal

A relation under incomplete knowledge is a DISJUNCTION of base relations, encoded
as a 5-bit integer mask (bit i set <=> BASE[i] possible). UNIVERSAL = 0x1F (all 5),
EMPTY = 0 (the contradictory/impossible relation).

The composition table is the standard RCC-5 weak composition table (Bennett 1994;
Cohn, Bennett, Gooday & Gotts 1997 "Qualitative Spatial Representation and Reasoning
with the Region Connection Calculus", Table 2). It is verified in the tests to be the
*tightest sound* coarsening of the theorem-prover-verified RCC-8 table under the map
``rcc8_to_rcc5`` below (DC,EC -> DR / PO -> PO / TPP,NTPP -> PP / TPPi,NTPPi -> PPi /
EQ -> EQ): each entry is the union, over all RCC-8 base pairs in the preimage, of the
quotiented RCC-8 composition. This gives the soundness containment

    rcc8_to_rcc5(rcc8.compose(A, B))  is-subset-of  compose(rcc8_to_rcc5(A), rcc8_to_rcc5(B))

for all RCC-8 masks A, B. (Equality does NOT hold in general --- e.g. DC o EC quotients
to {DR,PO,PP} whereas DR o DR is universal --- because distinct RCC-8 bases share one
RCC-5 image, so the coarse table must over-approximate.) Consequently the RCC-5 layer
is a *sound* coarsening: an inconsistency exposed in RCC-5 is a real RCC-8 inconsistency
(the coarse layer never invents contradictions). The converse does NOT hold --- RCC-5
is strictly weaker, so an RCC-8 contradiction may survive coarsening as RCC-5-consistent.
RCC-5 is therefore used only as a cheap early rejection filter ahead of RCC-8 refinement;
this module makes no completeness claim.

rcc8 is imported solely to *interpret* an incoming 8-bit mask; the rcc8 module is not
modified and this module owns no rcc8 state.
"""
from __future__ import annotations

from . import rcc8

# -- base relations, fixed index order -----------------------------------------
BASE: tuple[str, ...] = ("DR", "PO", "PP", "PPi", "EQ")
_INDEX: dict[str, int] = {name: i for i, name in enumerate(BASE)}

UNIVERSAL: int = 0x1F          # all 5 base relations possible
EMPTY: int = 0                 # no base relation possible (impossible / contradiction)


# -- name <-> mask helpers -----------------------------------------------------
def bit(name: str) -> int:
    """Single-base-relation mask for ``name`` (e.g. bit('DR') == 1)."""
    try:
        return 1 << _INDEX[name]
    except KeyError:
        raise ValueError(f"unknown RCC-5 base relation: {name!r}") from None


def mask(*names: str) -> int:
    """Disjunction mask of the given base-relation names. mask() == EMPTY."""
    m = 0
    for n in names:
        m |= bit(n)
    return m


def names(m: int) -> list[str]:
    """Base-relation names present in mask ``m``, in canonical BASE order."""
    if m < 0 or m > UNIVERSAL:
        raise ValueError(f"mask out of range 0..31: {m}")
    return [name for i, name in enumerate(BASE) if m & (1 << i)]


# -- converse ------------------------------------------------------------------
# DR, PO, EQ are self-converse; PP <-> PPi.
_CONVERSE_NAME: dict[str, str] = {
    "DR": "DR", "PO": "PO", "EQ": "EQ",
    "PP": "PPi", "PPi": "PP",
}
_CONVERSE_BIT: tuple[int, ...] = tuple(bit(_CONVERSE_NAME[name]) for name in BASE)


def converse(m: int) -> int:
    """Converse of relation mask ``m``: converse(x r y) is the mask for (y r^-1 x)."""
    if m < 0 or m > UNIVERSAL:
        raise ValueError(f"mask out of range 0..31: {m}")
    out = 0
    for i in range(5):
        if m & (1 << i):
            out |= _CONVERSE_BIT[i]
    return out


# -- composition table ---------------------------------------------------------
# _COMP_NAMES[a][b] = base relations entailed by composing base a (x,y) with base b (y,z).
# Standard RCC-5 weak composition table (Cohn/Bennett/Gooday/Gotts 1997). "*" is UNIVERSAL.
_ALL = BASE  # convenience alias for the universal entry

_COMP_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    "DR": {
        "DR":  _ALL,
        "PO":  ("DR", "PO", "PP"),
        "PP":  ("DR", "PO", "PP"),
        "PPi": ("DR",),
        "EQ":  ("DR",),
    },
    "PO": {
        "DR":  ("DR", "PO", "PPi"),
        "PO":  _ALL,
        "PP":  ("PO", "PP"),
        "PPi": ("DR", "PO", "PPi"),
        "EQ":  ("PO",),
    },
    "PP": {
        "DR":  ("DR",),
        "PO":  ("DR", "PO", "PP"),
        "PP":  ("PP",),
        "PPi": _ALL,
        "EQ":  ("PP",),
    },
    "PPi": {
        "DR":  ("DR", "PO", "PPi"),
        "PO":  ("PO", "PPi"),
        "PP":  ("PO", "PP", "PPi", "EQ"),
        "PPi": ("PPi",),
        "EQ":  ("PPi",),
    },
    "EQ": {
        "DR":  ("DR",),
        "PO":  ("PO",),
        "PP":  ("PP",),
        "PPi": ("PPi",),
        "EQ":  ("EQ",),
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
    tabulated base composition. Composing with EMPTY yields EMPTY."""
    if a_mask < 0 or a_mask > UNIVERSAL or b_mask < 0 or b_mask > UNIVERSAL:
        raise ValueError(f"mask out of range 0..31: {a_mask}, {b_mask}")
    out = 0
    for i in range(5):
        if not (a_mask & (1 << i)):
            continue
        row = _COMP_BIT[i]
        for j in range(5):
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


# -- RCC-8 -> RCC-5 quotient ---------------------------------------------------
# Many-to-one map on base relations: DC,EC -> DR ; PO -> PO ; TPP,NTPP -> PP ;
# TPPi,NTPPi -> PPi ; EQ -> EQ. Built as a per-rcc8-base target-bit table so that
# rcc8_to_rcc5 folds a whole disjunction mask by OR-ing the images.
_RCC8_TO_RCC5_NAME: dict[str, str] = {
    "DC": "DR", "EC": "DR",
    "PO": "PO",
    "TPP": "PP", "NTPP": "PP",
    "TPPi": "PPi", "NTPPi": "PPi",
    "EQ": "EQ",
}
# indexed by rcc8 base index (0..7) -> rcc5 target bit
_RCC8_TO_RCC5_BIT: tuple[int, ...] = tuple(
    bit(_RCC8_TO_RCC5_NAME[name]) for name in rcc8.BASE
)


def rcc8_to_rcc5(rcc8_mask: int) -> int:
    """Fold an 8-bit RCC-8 disjunction mask onto its 5-bit RCC-5 image (quotient map).

    Many-to-one: DC,EC -> DR / PO -> PO / TPP,NTPP -> PP / TPPi,NTPPi -> PPi / EQ -> EQ.
    EMPTY maps to EMPTY, so a contradictory RCC-8 relation stays contradictory --- the
    coarsening is *sound* (RCC-5 inconsistency implies RCC-8 inconsistency). Because the
    map merges base relations it is lossy in the other direction: an RCC-5-consistent
    image does NOT certify RCC-8 consistency. Do not read this as a completeness claim."""
    if rcc8_mask < 0 or rcc8_mask > rcc8.UNIVERSAL:
        raise ValueError(f"rcc8 mask out of range 0..255: {rcc8_mask}")
    out = 0
    for i in range(8):
        if rcc8_mask & (1 << i):
            out |= _RCC8_TO_RCC5_BIT[i]
    return out


__all__ = [
    "BASE", "UNIVERSAL", "EMPTY",
    "bit", "mask", "names",
    "converse", "compose", "base_compose",
    "intersect", "union", "is_base",
    "rcc8_to_rcc5",
]
