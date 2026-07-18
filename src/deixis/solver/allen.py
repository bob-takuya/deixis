"""Allen's interval algebra (qualitative temporal reasoning) — symbolic layer only.

This module is the *time* counterpart of :mod:`deixis.core.rcc8`. It is a pure
symbolic/relation-algebra module: **no geometry imports**, all reasoning is exact
integer bit arithmetic (no floats anywhere), objects are frozen/immutable.

The thirteen jointly-exhaustive, pairwise-disjoint base relations between two
1-D intervals X, Y (Allen 1983), in fixed index order:

    index  name   meaning (of ``X rel Y``)            converse
      0     b     X before Y      (X ends before Y starts)     bi
      1     bi    X after Y                                     b
      2     m     X meets Y       (X.end == Y.start)            mi
      3     mi    X met-by Y                                    m
      4     o     X overlaps Y    (X starts first, ends inside) oi
      5     oi    X overlapped-by Y                             o
      6     s     X starts Y      (same start, X shorter)       si
      7     si    X started-by Y                                s
      8     d     X during Y      (X strictly inside Y)         di
      9     di    X contains Y                                  d
     10     f     X finishes Y    (same end, X shorter)         fi
     11     fi    X finished-by Y                               f
     12     eq    X equals Y                                    eq

A relation under incomplete knowledge is a DISJUNCTION of base relations, encoded
as a 13-bit integer mask (bit i set  <=>  base relation ``BASE[i]`` is possible).
``UNIVERSAL`` = all 13 (0x1FFF), ``EMPTY`` = 0 (contradiction / impossible).

The 13x13 composition table and the converse table below were transcribed from
Allen's standard transitivity table (Allen, "Maintaining Knowledge about Temporal
Intervals", CACM 1983), cross-checked against T. Alspaugh's published reproduction
(Table 4a). ``compose(a, b)`` returns, for masks ``a`` (relating X,Y) and ``b``
(relating Y,Z), the union over every base pair of their tabulated composition — the
tightest mask relating X,Z that the algebra entails.

Completeness note (honest, fail-closed)
---------------------------------------
Deciding satisfiability of a general Allen network is NP-complete. Path-consistency
(:func:`path_consistency`) is a *sound* refutation procedure — it never reports a
genuinely satisfiable network as inconsistent — but it is **NOT** a complete
satisfiability test on the full algebra. A path-consistent (non-empty) closure is
therefore reported as ``"path_consistent"``, which does **not** claim satisfiability.
The tractable subclass on which path-consistency *is* complete (ORD-Horn; Nebel &
Bürckert 1995) is intentionally left for future work and NOT claimed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

# -- base relations, fixed index order -----------------------------------------
BASE: tuple[str, ...] = (
    "b", "bi", "m", "mi", "o", "oi", "s", "si", "d", "di", "f", "fi", "eq",
)
_INDEX: dict[str, int] = {name: i for i, name in enumerate(BASE)}

_N = len(BASE)                 # 13
UNIVERSAL: int = (1 << _N) - 1  # all 13 base relations possible (0x1FFF)
EMPTY: int = 0                  # no base relation possible (impossible / contradiction)


# -- name <-> mask helpers -----------------------------------------------------
def bit(name: str) -> int:
    """Single-base-relation mask for ``name`` (e.g. bit('b') == 1)."""
    try:
        return 1 << _INDEX[name]
    except KeyError:
        raise ValueError(f"unknown Allen base relation: {name!r}") from None


def mask(*names: str) -> int:
    """Disjunction mask of the given base-relation names. mask() == EMPTY."""
    m = 0
    for n in names:
        m |= bit(n)
    return m


def names(m: int) -> list[str]:
    """Base-relation names present in mask ``m``, in canonical BASE order."""
    if m < 0 or m > UNIVERSAL:
        raise ValueError(f"mask out of range 0..{UNIVERSAL}: {m}")
    return [name for i, name in enumerate(BASE) if m & (1 << i)]


# -- converse ------------------------------------------------------------------
# eq is self-converse; every other relation pairs with its "i" partner.
_CONVERSE_NAME: dict[str, str] = {
    "b": "bi", "bi": "b",
    "m": "mi", "mi": "m",
    "o": "oi", "oi": "o",
    "s": "si", "si": "s",
    "d": "di", "di": "d",
    "f": "fi", "fi": "f",
    "eq": "eq",
}
_CONVERSE_BIT: tuple[int, ...] = tuple(bit(_CONVERSE_NAME[name]) for name in BASE)


def converse(m: int) -> int:
    """Converse of relation mask ``m``: converse(X r Y) is the mask for (Y r^-1 X)."""
    if m < 0 or m > UNIVERSAL:
        raise ValueError(f"mask out of range 0..{UNIVERSAL}: {m}")
    out = 0
    for i in range(_N):
        if m & (1 << i):
            out |= _CONVERSE_BIT[i]
    return out


# -- composition table ---------------------------------------------------------
# Recurring result-sets (Allen 1983 / Alspaugh Table 4a), expressed in our names.
_FULL = BASE
_PMOSD = ("b", "m", "o", "s", "d")
_OSD = ("o", "s", "d")
_PMO = ("b", "m", "o")
_PMOFD = ("b", "m", "o", "fi", "di")
_OFD = ("o", "fi", "di")
_CONCUR = ("o", "fi", "di", "s", "eq", "si", "d", "f", "oi")
_DSO = ("di", "si", "oi")
_DSOMP = ("di", "si", "oi", "mi", "bi")
_SES = ("s", "eq", "si")
_DFO = ("d", "f", "oi")
_DFOMP = ("d", "f", "oi", "mi", "bi")
_OMP = ("oi", "mi", "bi")
_FEF = ("fi", "eq", "f")

# _COMP_NAMES[a][b] = base relations entailed by composing base a (X,Y) with base b (Y,Z).
_COMP_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    "b": {
        "b": ("b",), "m": ("b",), "o": ("b",), "fi": ("b",), "di": ("b",),
        "s": ("b",), "eq": ("b",), "si": ("b",),
        "d": _PMOSD, "f": _PMOSD, "oi": _PMOSD, "mi": _PMOSD, "bi": _FULL,
    },
    "m": {
        "b": ("b",), "m": ("b",), "o": ("b",), "fi": ("b",), "di": ("b",),
        "s": ("m",), "eq": ("m",), "si": ("m",),
        "d": _OSD, "f": _OSD, "oi": _OSD, "mi": _FEF, "bi": _DSOMP,
    },
    "o": {
        "b": ("b",), "m": ("b",), "o": _PMO, "fi": _PMO, "di": _PMOFD,
        "s": ("o",), "eq": ("o",), "si": _OFD,
        "d": _OSD, "f": _OSD, "oi": _CONCUR, "mi": _DSO, "bi": _DSOMP,
    },
    "fi": {
        "b": ("b",), "m": ("m",), "o": ("o",), "fi": ("fi",), "di": ("di",),
        "s": ("o",), "eq": ("fi",), "si": ("di",),
        "d": _OSD, "f": _FEF, "oi": _DSO, "mi": _DSO, "bi": _DSOMP,
    },
    "di": {
        "b": _PMOFD, "m": _OFD, "o": _OFD, "fi": ("di",), "di": ("di",),
        "s": _OFD, "eq": ("di",), "si": ("di",),
        "d": _CONCUR, "f": _DSO, "oi": _DSO, "mi": _DSO, "bi": _DSOMP,
    },
    "s": {
        "b": ("b",), "m": ("b",), "o": _PMO, "fi": _PMO, "di": _PMOFD,
        "s": ("s",), "eq": ("s",), "si": _SES,
        "d": ("d",), "f": ("d",), "oi": _DFO, "mi": ("mi",), "bi": ("bi",),
    },
    "eq": {
        "b": ("b",), "m": ("m",), "o": ("o",), "fi": ("fi",), "di": ("di",),
        "s": ("s",), "eq": ("eq",), "si": ("si",),
        "d": ("d",), "f": ("f",), "oi": ("oi",), "mi": ("mi",), "bi": ("bi",),
    },
    "si": {
        "b": _PMOFD, "m": _OFD, "o": _OFD, "fi": ("di",), "di": ("di",),
        "s": _SES, "eq": ("si",), "si": ("si",),
        "d": _DFO, "f": ("oi",), "oi": ("oi",), "mi": ("mi",), "bi": ("bi",),
    },
    "d": {
        "b": ("b",), "m": ("b",), "o": _PMOSD, "fi": _PMOSD, "di": _FULL,
        "s": ("d",), "eq": ("d",), "si": _DFOMP,
        "d": ("d",), "f": ("d",), "oi": _DFOMP, "mi": ("bi",), "bi": ("bi",),
    },
    "f": {
        "b": ("b",), "m": ("m",), "o": _OSD, "fi": _FEF, "di": _DSOMP,
        "s": ("d",), "eq": ("f",), "si": _OMP,
        "d": ("d",), "f": ("f",), "oi": _OMP, "mi": ("bi",), "bi": ("bi",),
    },
    "oi": {
        "b": _PMOFD, "m": _OFD, "o": _CONCUR, "fi": _DSO, "di": _DSOMP,
        "s": _DFO, "eq": ("oi",), "si": _OMP,
        "d": _DFO, "f": ("oi",), "oi": _OMP, "mi": ("bi",), "bi": ("bi",),
    },
    "mi": {
        "b": _PMOFD, "m": _SES, "o": _DFO, "fi": ("mi",), "di": ("bi",),
        "s": _DFO, "eq": ("mi",), "si": ("bi",),
        "d": _DFO, "f": ("mi",), "oi": ("bi",), "mi": ("bi",), "bi": ("bi",),
    },
    "bi": {
        "b": _FULL, "m": _DFOMP, "o": _DFOMP, "fi": ("bi",), "di": ("bi",),
        "s": _DFOMP, "eq": ("bi",), "si": ("bi",),
        "d": _DFOMP, "f": ("bi",), "oi": ("bi",), "mi": ("bi",), "bi": ("bi",),
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
    """Compose relation masks: given (X a_mask Y) and (Y b_mask Z), return mask (X ? Z).

    Composition = union over every base pair (a in a_mask, b in b_mask) of the tabulated
    base composition. Composing with EMPTY yields EMPTY (no consistent middle interval)."""
    if a_mask < 0 or a_mask > UNIVERSAL or b_mask < 0 or b_mask > UNIVERSAL:
        raise ValueError(f"mask out of range 0..{UNIVERSAL}: {a_mask}, {b_mask}")
    out = 0
    for i in range(_N):
        if not (a_mask & (1 << i)):
            continue
        row = _COMP_BIT[i]
        for j in range(_N):
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


# ---------------------------------------------------------------- network container
def _check_mask(m: int) -> int:
    if not isinstance(m, int) or isinstance(m, bool):
        raise TypeError(f"relation mask must be a plain int, got {type(m)!r}")
    if m < 0 or m > UNIVERSAL:
        raise ValueError(f"relation mask out of range 0..{UNIVERSAL}: {m}")
    return m


_EQ = bit("eq")


def _canon(i: str, j: str) -> tuple[str, str]:
    """Canonical unordered-pair key (sorted), used for de-duplicating edges."""
    return (i, j) if i <= j else (j, i)


@dataclass(frozen=True)
class AllenNetwork:
    """A qualitative Allen interval-algebra constraint network (frozen / immutable).

    ``variables`` is the sorted tuple of interval ids. ``arcs`` maps a **canonical**
    unordered pair ``(i,j)`` with ``i < j`` to the relation mask asserted for the
    ordered pair ``i -> j``. The converse ``j -> i`` mask is derived on demand via
    :func:`converse`; the diagonal ``i -> i`` is the identity ``eq``; any pair not
    listed is the universal relation (no information). Never mutate an instance;
    closure operations return a fresh network.
    """

    variables: tuple[str, ...]
    arcs: tuple[tuple[tuple[str, str], int], ...]  # ((i,j), mask) with i < j, sorted

    @classmethod
    def from_edges(
        cls,
        variables: Iterable[str],
        edges: Mapping[tuple[str, str], int] | Iterable[tuple[str, str, int]] = (),
    ) -> "AllenNetwork":
        """Build a network from a variable set and directed edges ``(i, j) -> mask``.

        ``edges`` may be a mapping ``{(i, j): mask}`` or an iterable of ``(i, j, mask)``.
        Opposite directions are reconciled by converse + intersection. Self edges must
        be exactly ``eq``; anything else raises.
        """
        var_set = set(variables)
        if isinstance(edges, Mapping):
            triples = [(i, j, m) for (i, j), m in edges.items()]
        else:
            triples = list(edges)

        acc: dict[tuple[str, str], int] = {}
        for i, j, m in triples:
            _check_mask(m)
            var_set.add(i)
            var_set.add(j)
            if i == j:
                if m != _EQ:
                    raise ValueError(
                        f"self-edge ({i},{i}) must be exactly eq, got {names(m)}"
                    )
                continue
            key = _canon(i, j)
            m_canon = m if key == (i, j) else converse(m)
            if key in acc:
                acc[key] &= m_canon
            else:
                acc[key] = m_canon

        arcs = tuple(sorted(acc.items()))
        return cls(variables=tuple(sorted(var_set)), arcs=arcs)

    def _arc_dict(self) -> dict[tuple[str, str], int]:
        return dict(self.arcs)

    def relation(self, i: str, j: str) -> int:
        """Relation mask for the ordered pair ``i -> j`` (eq on the diagonal;
        UNIVERSAL if unconstrained). Converse pairs are derived automatically."""
        if i == j:
            return _EQ
        key = _canon(i, j)
        d = self._arc_dict()
        if key not in d:
            return UNIVERSAL
        m = d[key]
        return m if key == (i, j) else converse(m)

    def matrix(self) -> dict[tuple[str, str], int]:
        """Full ordered-pair relation matrix for every pair of variables."""
        out: dict[tuple[str, str], int] = {}
        for i in self.variables:
            for j in self.variables:
                out[(i, j)] = self.relation(i, j)
        return out

    def constraint_masks(self) -> tuple[int, ...]:
        """The explicitly asserted (canonical-direction) relation masks."""
        return tuple(m for _, m in self.arcs)

    def edge_keys(self) -> tuple[tuple[str, str], ...]:
        """Canonical ``(i, j)`` keys (``i < j``) of the explicitly asserted edges."""
        return tuple(k for k, _ in self.arcs)


# ---------------------------------------------------------------- path consistency
def _closure(matrix: dict[tuple[str, str], int], variables: tuple[str, ...]):
    """Run the algebraic-closure fixpoint in place on ``matrix`` (same pattern as the
    RCC-8 reasoner, re-implemented here so pathconsistency.py is untouched).

    Returns ``(consistent, matrix)``: ``consistent`` is False as soon as any pair's
    mask becomes EMPTY. Masks only ever shrink (intersection), so the fixpoint is
    reached in finitely many sweeps.
    """
    for i in variables:
        for j in variables:
            if i != j and matrix[(i, j)] == EMPTY:
                return False, matrix

    changed = True
    while changed:
        changed = False
        for i in variables:
            for k in variables:
                if k == i:
                    continue
                r_ik = matrix[(i, k)]
                if r_ik == UNIVERSAL:
                    continue
                for j in variables:
                    if j == i or j == k:
                        continue
                    comp = compose(r_ik, matrix[(k, j)])
                    old = matrix[(i, j)]
                    new = old & comp
                    if new != old:
                        matrix[(i, j)] = new
                        matrix[(j, i)] = converse(new)
                        changed = True
                        if new == EMPTY:
                            return False, matrix
    return True, matrix


def _matrix_to_network(matrix: dict[tuple[str, str], int], variables: tuple[str, ...]) -> AllenNetwork:
    edges = []
    for i in variables:
        for j in variables:
            if i < j:
                edges.append((i, j, matrix[(i, j)]))
    return AllenNetwork.from_edges(variables, edges)


def path_consistency(net: AllenNetwork) -> tuple[str, AllenNetwork]:
    """Compute the algebraic closure and return ``(status, closed_net)``.

    ``status`` is one of:

    * ``'inconsistent'``   -- some pair's relation collapsed to EMPTY. This is **sound**:
      the network is genuinely unsatisfiable (path-consistency never over-prunes).
    * ``'path_consistent'`` -- the closure reached a fixpoint with every relation
      non-empty. **This does NOT claim satisfiability.** Deciding Allen satisfiability
      is NP-complete; path-consistency is complete only on tractable subclasses
      (ORD-Horn), which are intentionally not checked here. So a path-consistent
      network may still be unsatisfiable — the honest verdict is "not refuted".

    ``closed_net`` is the refined network (a fresh, immutable :class:`AllenNetwork`).
    """
    matrix = net.matrix()
    consistent, matrix = _closure(matrix, net.variables)
    closed = _matrix_to_network(matrix, net.variables)
    if not consistent:
        return "inconsistent", closed
    return "path_consistent", closed


__all__ = [
    "BASE", "UNIVERSAL", "EMPTY",
    "bit", "mask", "names",
    "converse", "compose", "base_compose",
    "intersect", "union", "is_base",
    "AllenNetwork", "path_consistency",
]
