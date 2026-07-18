"""RCC-8 path-consistency (algebraic closure) and tractable-fragment reasoning.

This module provides the qualitative constraint reasoner over the RCC-8 relation
algebra (see :mod:`deixis.core.rcc8`). Its job in the IR is honest, fail-closed:

* :func:`path_consistency` enforces the algebraic-closure (path-consistency) fixpoint
  ``R[i][j] <= R[i][k] o R[k][j]`` over every triangle and reports one of three
  epistemic verdicts (never silently claims satisfiability it cannot back).
* :func:`fragment_membership` classifies whether a whole network lies inside one of
  the three *maximal tractable* subclasses of RCC-8.
* :func:`decides_satisfiability` says whether, for this network, path-consistency is
  a sound **and complete** decision procedure for satisfiability.
* :func:`unsat_core` returns a (deletion-)minimal set of edges responsible for an
  inconsistency.

Tractable fragments (Renz & Nebel 1999; Renz 1999, complete analysis)
--------------------------------------------------------------------
Deciding RCC-8 satisfiability (RSAT) is NP-hard in general, but path-consistency
*decides* it on any network whose relations all come from one of the three maximal
tractable subclasses that contain all base relations:

    ^H8  (148 relations)   Q8  (160 relations)   C8  (158 relations)

``NP8`` (76 relations) is the complementary set: any of them combined with the base
relations yields NP-completeness. Using masks ``R`` (subsets of the 8 base relations,
where ``x in R`` means base relation ``x`` is possible), the published definitions are::

    NP8 = { R | PO not in R and (NTPP in R or TPP in R)
                            and (NTPPi in R or TPPi in R) }
        u { {EC,NTPP,EQ}, {DC,EC,NTPP,EQ}, {EC,NTPPi,EQ}, {DC,EC,NTPPi,EQ} }

    ^H8 = (RCC8 \\ NP8) \\ { R | (EQ,NTPP  in R and TPP  not in R)
                              or (EQ,NTPPi in R and TPPi not in R) }
    C8  = (RCC8 \\ NP8) \\ { R | EC in R and PO not in R
                                 and R & {TPP,NTPP,TPPi,NTPPi,EQ} != {} }
    Q8  = (RCC8 \\ NP8) \\ { R | EQ in R and PO not in R
                                 and R & {TPP,NTPP,TPPi,NTPPi} != {} }

with ``^H8 u C8 = RCC8 \\ NP8`` and ``Q8 subset (^H8 u C8)``. The cardinalities above are
reproduced exactly by the predicates below (regression-checked in the test suite).

All relations are 8-bit integer masks; all reasoning is exact integer bit arithmetic
(no floats anywhere). Network objects are frozen/immutable; closure returns a NEW
network rather than mutating the input.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from deixis.core.rcc8 import (
    UNIVERSAL,
    EMPTY,
    bit,
    compose,
    converse,
    mask as _mask,
    names,
)

# ---------------------------------------------------------------- base-relation bits
_DC = bit("DC")
_EC = bit("EC")
_PO = bit("PO")
_TPP = bit("TPP")
_NTPP = bit("NTPP")
_TPPi = bit("TPPi")
_NTPPi = bit("NTPPi")
_EQ = bit("EQ")

# Fragment name constants (returned by fragment_membership).
H8 = "H8"    # ^H8  (Renz & Nebel's H-hat-8): the smallest but best-decomposing subclass
C8 = "C8"
Q8 = "Q8"
FRAGMENTS = (H8, C8, Q8)

# The four explicit NP8 relations not captured by the parametric NP8 condition.
_NP8_EXPLICIT = frozenset({
    _mask("EC", "NTPP", "EQ"),
    _mask("DC", "EC", "NTPP", "EQ"),
    _mask("EC", "NTPPi", "EQ"),
    _mask("DC", "EC", "NTPPi", "EQ"),
})


def _check_mask(m: int) -> int:
    if not isinstance(m, int) or isinstance(m, bool):
        raise TypeError(f"relation mask must be a plain int, got {type(m)!r}")
    if m < 0 or m > UNIVERSAL:
        raise ValueError(f"relation mask out of range 0..255: {m}")
    return m


# ---------------------------------------------------------------- fragment membership
def in_np8(m: int) -> bool:
    """True iff relation mask ``m`` is in NP8 (intractable when mixed with base rels)."""
    _check_mask(m)
    parametric = (
        not (m & _PO)
        and ((m & _NTPP) or (m & _TPP))
        and ((m & _NTPPi) or (m & _TPPi))
    )
    return bool(parametric) or m in _NP8_EXPLICIT


def in_H8(m: int) -> bool:
    """True iff ``m`` belongs to the maximal tractable subclass ^H8 (148 relations)."""
    _check_mask(m)
    if in_np8(m):
        return False
    removed = (
        ((m & _EQ) and (m & _NTPP) and not (m & _TPP))
        or ((m & _EQ) and (m & _NTPPi) and not (m & _TPPi))
    )
    return not removed


def in_C8(m: int) -> bool:
    """True iff ``m`` belongs to the maximal tractable subclass C8 (158 relations)."""
    _check_mask(m)
    if in_np8(m):
        return False
    removed = (
        (m & _EC)
        and not (m & _PO)
        and (m & (_TPP | _NTPP | _TPPi | _NTPPi | _EQ))
    )
    return not bool(removed)


def in_Q8(m: int) -> bool:
    """True iff ``m`` belongs to the maximal tractable subclass Q8 (160 relations)."""
    _check_mask(m)
    if in_np8(m):
        return False
    removed = (
        (m & _EQ)
        and not (m & _PO)
        and (m & (_TPP | _NTPP | _TPPi | _NTPPi))
    )
    return not bool(removed)


_FRAGMENT_PRED = {H8: in_H8, C8: in_C8, Q8: in_Q8}


def fragment_of_mask(m: int) -> frozenset[str]:
    """Set of tractable subclasses (subset of {H8,C8,Q8}) that contain relation ``m``."""
    return frozenset(name for name in FRAGMENTS if _FRAGMENT_PRED[name](m))


# ---------------------------------------------------------------- network container
def _canon(i: str, j: str) -> tuple[str, str]:
    """Canonical unordered-pair key (sorted), used for de-duplicating edges."""
    return (i, j) if i <= j else (j, i)


@dataclass(frozen=True)
class RCCNetwork:
    """A qualitative RCC-8 constraint network (frozen / immutable).

    ``variables`` is the sorted tuple of region ids. ``arcs`` maps a **canonical**
    unordered pair ``(i,j)`` with ``i < j`` to the relation mask asserted for the
    ordered pair ``i -> j``. The converse ``j -> i`` mask is *derived* on demand via
    :func:`deixis.core.rcc8.converse`; diagonal ``i -> i`` is the identity ``EQ``; any
    pair not listed is the universal relation (no information).

    Build with :meth:`from_edges`, which normalizes user input (adds symmetry via
    converse, folds duplicate/opposite edges by intersection, validates masks). Never
    mutate an instance; closure operations return a fresh network.
    """

    variables: tuple[str, ...]
    arcs: tuple[tuple[tuple[str, str], int], ...]  # ((i,j), mask) with i < j, sorted

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_edges(
        cls,
        variables: Iterable[str],
        edges: Mapping[tuple[str, str], int] | Iterable[tuple[str, str, int]] = (),
    ) -> "RCCNetwork":
        """Build a network from a variable set and directed edges ``(i, j) -> mask``.

        ``edges`` may be a mapping ``{(i, j): mask}`` or an iterable of ``(i, j, mask)``.
        Opposite directions are reconciled by converse + intersection, so giving both
        ``(a, b) -> m`` and ``(b, a) -> m'`` yields ``m & converse(m')`` on ``a -> b``.
        Self edges must be a sub-relation of EQ (EQ or EMPTY); anything else raises.
        """
        var_set = set(variables)
        # normalize edges into an iterable of (i, j, mask)
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
                # A region relates to itself only by EQ. Anything else -- including a
                # broader mask or the EMPTY relation (a "region != itself" contradiction)
                # -- is malformed and must not be silently dropped.
                if m != _EQ:
                    raise ValueError(
                        f"self-edge ({i},{i}) must be exactly EQ, got {names(m)}"
                    )
                # identity is implicit; ignore explicit EQ self-edges
                continue
            key = _canon(i, j)
            # store the mask in canonical i<j direction
            m_canon = m if key == (i, j) else converse(m)
            if key in acc:
                acc[key] &= m_canon
            else:
                acc[key] = m_canon

        arcs = tuple(sorted(acc.items()))
        return cls(variables=tuple(sorted(var_set)), arcs=arcs)

    # -- accessors ------------------------------------------------------------
    def _arc_dict(self) -> dict[tuple[str, str], int]:
        return dict(self.arcs)

    def relation(self, i: str, j: str) -> int:
        """Relation mask for the ordered pair ``i -> j`` (EQ on the diagonal;
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
    """Run the algebraic-closure fixpoint in place on ``matrix``.

    Returns ``(consistent, matrix)``: ``consistent`` is False as soon as any pair's
    mask becomes EMPTY. Masks only ever shrink (intersection), so the fixpoint is
    reached in finitely many sweeps.
    """
    # An already-empty edge is an immediate contradiction.
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
                    # composing through k via a universal first leg cannot refine.
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


def _matrix_to_network(matrix: dict[tuple[str, str], int], variables: tuple[str, ...]) -> RCCNetwork:
    edges = []
    for i in variables:
        for j in variables:
            if i < j:
                edges.append((i, j, matrix[(i, j)]))
    return RCCNetwork.from_edges(variables, edges)


def path_consistency(net: RCCNetwork) -> tuple[str, RCCNetwork]:
    """Compute the algebraic closure and return ``(status, closed_net)``.

    ``status`` is one of:

    * ``'inconsistent'`` -- some pair's relation collapsed to EMPTY. This is **sound**:
      the network is genuinely unsatisfiable (path-consistency never over-prunes).
    * ``'consistent'``   -- the closure reached a fixpoint with every relation
      non-empty **and** the network lies within a tractable fragment, so
      path-consistency is a complete decision procedure -> genuinely satisfiable.
    * ``'unknown'``      -- fixpoint reached with all relations non-empty, but the
      network uses relations outside every tractable fragment. Path-consistency is
      necessary-but-not-sufficient here, so satisfiability is honestly undecided
      (fail-closed).

    ``closed_net`` is the refined network (a fresh, immutable :class:`RCCNetwork`).
    """
    matrix = net.matrix()
    consistent, matrix = _closure(matrix, net.variables)
    closed = _matrix_to_network(matrix, net.variables)
    if not consistent:
        return "inconsistent", closed
    if decides_satisfiability(net):
        return "consistent", closed
    return "unknown", closed


# ---------------------------------------------------------------- fragment of a network
def fragment_membership(net: RCCNetwork) -> Optional[str]:
    """Return a tractable subclass name containing *every* asserted relation, else None.

    Checks the three maximal tractable subclasses in the order ``H8, C8, Q8`` and
    returns the first one that contains all of the network's constraint masks. Base
    and universal relations are in all three, so unconstrained pairs never disqualify
    a network. Returns ``None`` when no single subclass covers every relation (the
    network may still be satisfiable, but path-consistency will not decide it).
    """
    masks = net.constraint_masks()
    for name in FRAGMENTS:
        pred = _FRAGMENT_PRED[name]
        if all(pred(m) for m in masks):
            return name
    return None


def decides_satisfiability(net: RCCNetwork) -> bool:
    """True iff path-consistency is a sound+complete satisfiability test for ``net``.

    Holds exactly when the network lies within one of the maximal tractable subclasses
    (``fragment_membership`` is not None). Outside them, returns False (=> path
    consistency yields ``'unknown'``, never a false ``'consistent'``)."""
    return fragment_membership(net) is not None


# ---------------------------------------------------------------- unsat core
def unsat_core(net: RCCNetwork) -> set[tuple[str, str]]:
    """Return an (approximately) minimal set of edges responsible for inconsistency.

    Uses a deletion filter: starting from all asserted edges, try to relax each edge
    to the universal relation; if the network stays inconsistent without it, the edge
    is not needed and is dropped. The surviving edges form an **irreducible** (locally
    minimal) unsatisfiable core -- removing any one of them makes the remainder
    path-consistent. Edge keys are canonical ``(i, j)`` with ``i < j``.

    Returns the empty set if the network is *not* inconsistent (nothing to explain).
    """
    status, _ = path_consistency(net)
    if status != "inconsistent":
        return set()

    variables = net.variables
    base = net._arc_dict()
    core = dict(base)  # candidate edges kept in the core

    def is_inconsistent(edge_map: dict[tuple[str, str], int]) -> bool:
        edges = [(i, j, m) for (i, j), m in edge_map.items()]
        cand = RCCNetwork.from_edges(variables, edges)
        matrix = cand.matrix()
        consistent, _ = _closure(matrix, variables)
        return not consistent

    # Deletion filter: iterate over a stable ordering of the edges.
    for key in sorted(base.keys()):
        if key not in core:
            continue
        trial = dict(core)
        del trial[key]
        if is_inconsistent(trial):
            # edge not needed for the contradiction -> drop it
            core = trial

    return set(core.keys())


__all__ = [
    "RCCNetwork",
    "H8", "C8", "Q8", "FRAGMENTS",
    "in_np8", "in_H8", "in_C8", "in_Q8",
    "fragment_of_mask", "fragment_membership", "decides_satisfiability",
    "path_consistency", "unsat_core",
]
