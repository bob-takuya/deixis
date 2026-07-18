"""Two-stage, fail-closed realizability pipeline (symbolic RCC -> geometry-in-D).

The pipeline turns a :class:`~deixis.core.types.RelSpec` into a
:class:`~deixis.core.types.Realization` by climbing the fail-closed realizability
ladder one rung at a time and *never* claiming a rung it cannot back:

    Stage A  (symbolic)   -- deixis.solver.pathconsistency + deixis.incidence.witness
      * Build the RCC-8 constraint network and run algebraic closure
        (:func:`path_consistency`).
      * If any pair collapses to the EMPTY relation  ->  status = INCONSISTENT and we
        STOP: the network is genuinely unsatisfiable, so no geometry is attempted.
      * Also run the witnesses' local consistency: a contact witness entails ``EC``;
        if a stated constraint on that pair forbids ``EC`` that is a symbolic
        contradiction  ->  INCONSISTENT (still no geometry).
      * Otherwise the closure reached a non-empty fixpoint: the spec is at least
        path-consistent  ->  status = RCC_CONSISTENT (Stage A passed). The stronger
        'consistent' verdict (tractable-fragment, satisfiability *decided*) and the
        weaker 'unknown' verdict (outside every tractable fragment) both pass Stage A
        to the RCC_CONSISTENT rung -- 'unknown' honestly does not license anything
        higher on its own.

    Stage B  (geometric)  -- deixis.grounding.grounding_op + deixis.solver.geom_solver
      * Apply the grounding decisions (turning delegated relations into grounded ones,
        recording the decision ledger; a decision may additionally *refine* its target
        constraint's relation mask when it names an RCC-8 relation).
      * Realize with :func:`geom_solver.solve_boxes` (exact rational AABBs).
      * REVERSE-VERIFY: recompute, from the exact Fraction box coordinates, the RCC-8
        base relation actually holding between every constrained pair (and, for EC
        pairs carrying a witness, the realized contact dimension) and cross-check it
        against the spec. Only if every constraint is satisfied by the *readback* do we
        report status = REALIZED_IN_D.
      * If geometry is UNSAT/UNKNOWN in the selected domain, or the reverse check
        disagrees, status = UNKNOWN -- "not realized in D" is NOT "impossible"
        (realized-unsat-in-D != INCONSISTENT); the symbolic layer already proved the
        spec is not contradictory, so we must not slander it as INCONSISTENT.

Fail-closed contract: downstream consumers may only trust a Realization at the *top
rung it reached*. INCONSISTENT and UNKNOWN carry no boxes worth building.

Everything is exact: box coordinates are :class:`fractions.Fraction`, RCC reasoning is
integer bit arithmetic. No floats enter any decision.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Optional, Sequence

from dataclasses import replace

from deixis.core import rcc8
from deixis.core.types import (
    Box,
    GroundingDecision,
    Realizability,
    Realization,
    RelationConstraint,
    RelSpec,
    Status,
)
from deixis.solver.pathconsistency import RCCNetwork, path_consistency, unsat_core
from deixis.solver import geom_solver
from deixis.grounding.grounding_op import GroundingError, ground
from deixis.incidence.witness import check_local_consistency

__all__ = ["PipelineError", "run", "family"]

_DOMAIN_NDIM = {"aabb_2d": 2, "aabb_3d": 3}


class PipelineError(Exception):
    """Raised for pipeline-level misuse (unknown domain, malformed request)."""


# ---------------------------------------------------------------- constraint ids
def _cid(c: RelationConstraint) -> str:
    """Stable id for a constraint, matching geom_solver's own convention."""
    return c.id or f"{c.src}->{c.dst}"


# ---------------------------------------------------------------- Stage A: symbolic
def _build_network(spec: RelSpec) -> RCCNetwork:
    """RCC-8 constraint network of the spec's *relation constraints*.

    Witnesses are handled separately (their EC entailment is checked by
    :func:`check_local_consistency`), so this network reflects only the declared
    qualitative relations. Duplicate/opposite edges on a pair are folded by
    :meth:`RCCNetwork.from_edges` (converse + intersection)."""
    variables = [r.id for r in spec.regions]
    edges = [(c.src, c.dst, c.rcc8_mask) for c in spec.constraints]
    return RCCNetwork.from_edges(variables, edges)


# Witness local-consistency codes that mean the spec is *malformed* (not merely
# unsatisfiable): these are fail-closed -- the pipeline refuses to realize such a spec.
_STRUCTURAL_CODES = frozenset(
    {"too_few_regions", "duplicate_region", "bad_contact_dim", "unknown_region", "unknown_cell"}
)


def _canon(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _witness_contradictions(spec: RelSpec) -> list[tuple[str, str]]:
    """Region pairs where a contact witness (entails EC) contradicts a constraint.

    Uses the witness module's own local-consistency checker and keeps only the
    ``rcc_incompatible`` verdict -- the genuinely *symbolic* contradiction (a witness
    says the pair touches, a constraint forbids EC). Structural well-formedness codes
    are handled separately by :func:`_structural_violations` (fail-closed)."""
    pairs: list[tuple[str, str]] = []
    for v in check_local_consistency(spec):
        if v.code == "rcc_incompatible" and len(v.regions) == 2:
            a, b = v.regions
            pairs.append(_canon(a, b))
    return pairs


def _structural_violations(spec: RelSpec) -> list[str]:
    """Human-readable descriptions of malformed-witness violations (fail-closed)."""
    return [str(v) for v in check_local_consistency(spec) if v.code in _STRUCTURAL_CODES]


def _witness_dim_conflicts(spec: RelSpec) -> list[tuple[str, str]]:
    """Canonical pairs constrained by two witnesses that disagree on contact dimension.

    A pair cannot simultaneously be, e.g., an edge (dim 1) *and* a point (dim 0) contact;
    such a witness set is a symbolic contradiction independent of any geometry."""
    seen: dict[tuple[str, str], int] = {}
    conflicts: list[tuple[str, str]] = []
    for w in spec.witnesses:
        regs = tuple(dict.fromkeys(w.incident_regions))
        d = int(w.intended_contact_dimension)
        for a, b in combinations(regs, 2):
            key = _canon(a, b)
            if key in seen:
                if seen[key] != d and key not in conflicts:
                    conflicts.append(key)
            else:
                seen[key] = d
    return conflicts


def _augment_with_witness_contacts(spec: RelSpec) -> RelSpec:
    """Add synthetic ``EC`` constraints for witness pairs that have no stated constraint.

    A contact witness entails EC, but the geometric solver only *imposes* a relation when
    a :class:`RelationConstraint` names the pair -- so a witnessed contact with no
    constraint would otherwise be left unrealized (and unchecked). We materialize it as a
    delegated EC constraint; the solver then honours the witness's contact dimension and
    the reverse-verify step checks it. Pairs that already carry a constraint are left
    alone (an EC-incompatible one is caught symbolically in Stage A)."""
    existing = {frozenset((c.src, c.dst)) for c in spec.constraints}
    ec = rcc8.mask("EC")
    extra: list[RelationConstraint] = []
    seen: set[frozenset[str]] = set()
    for w in spec.witnesses:
        regs = tuple(dict.fromkeys(w.incident_regions))
        for a, b in combinations(regs, 2):
            key = frozenset((a, b))
            if key in existing or key in seen:
                continue
            seen.add(key)
            extra.append(
                RelationConstraint(
                    src=a, dst=b, rcc8_mask=ec, status=Status.DELEGATED,
                    id=f"witness:{w.id}:{a}-{b}",
                )
            )
    if not extra:
        return spec
    return spec.with_constraints(spec.constraints + tuple(extra))


def _constraints_on_pairs(spec: RelSpec, pairs: Iterable[tuple[str, str]]) -> tuple[str, ...]:
    """Constraint ids whose {src,dst} matches any of the given canonical pairs."""
    want = {frozenset(p) for p in pairs}
    out: list[str] = []
    for c in spec.constraints:
        if frozenset((c.src, c.dst)) in want:
            out.append(_cid(c))
    return tuple(out)


# ---------------------------------------------------------------- grounding
def _parse_relation_mask(text: str) -> Optional[int]:
    """Interpret a grounding's ``fixed_value_or_domain`` as an RCC-8 relation, or None.

    Accepts a disjunction of base-relation names in any of the common separators, e.g.
    ``"EC"``, ``"EC|DC"``, ``"EC, DC"``, ``"rel:EC"``. Returns the corresponding mask,
    or ``None`` when the string is not a relation spec (e.g. a coordinate grounding
    like ``"x==4.20"``), in which case the grounding only flips edit-authority and
    does not refine geometry."""
    if not text:
        return None
    tokens = []
    tok = ""
    for ch in text:
        if ch.isalpha():
            tok += ch
        else:
            if tok:
                tokens.append(tok)
            tok = ""
    if tok:
        tokens.append(tok)
    if not tokens:
        return None
    upper = [t.upper() for t in tokens if t.lower() != "rel"]
    if not upper or any(t not in rcc8.BASE for t in upper):
        return None
    return rcc8.mask(*upper)


def _refine_constraint_mask(spec: RelSpec, target_id: str, refine: int) -> RelSpec:
    """Intersect the target constraint's mask with ``refine`` (sound narrowing)."""
    cs = list(spec.constraints)
    for i, c in enumerate(cs):
        if c.id == target_id:
            cs[i] = replace(c, rcc8_mask=c.rcc8_mask & refine)
            return spec.with_constraints(tuple(cs))
    return spec


def _apply_groundings(
    spec: RelSpec, groundings: Sequence[GroundingDecision]
) -> RelSpec:
    """Apply grounding decisions to the spec, returning a NEW spec.

    For a decision targeting a DELEGATED relation constraint we use the M0
    :func:`grounding.grounding_op.ground` (fail-closed against INVARIANT / non-delegated
    targets and duplicate ledger ids) to flip status and record the ledger; when the
    decision's ``fixed_value_or_domain`` names an RCC-8 relation we additionally refine
    the constraint's mask (intersection -- never widens). Decisions targeting a region
    or cell are recorded in the ledger only (they do not change the geometry encoding).
    """
    s = spec
    for d in groundings:
        if s.constraint(d.target) is not None:
            s = ground(s, d.target, d)  # fail-closed status flip + ledger append
            refine = _parse_relation_mask(d.fixed_value_or_domain)
            if refine is not None:
                s = _refine_constraint_mask(s, d.target, refine)
        else:
            # Not a constraint: the only other groundable target the M0 RelSpec exposes is
            # a Region. Reject an unknown target (typo / dangling id) fail-closed rather
            # than silently recording a decision that grounds nothing.
            if s.region(d.target) is None:
                raise GroundingError(
                    f"grounding target {d.target!r} is neither a RelationConstraint "
                    f"nor a Region in this spec"
                )
            if any(g.id == d.id for g in s.groundings):
                raise GroundingError(
                    f"decision id {d.id!r} already present in ledger"
                )
            s = s.with_groundings(s.groundings + (d,))
    return s


# ---------------------------------------------------------------- reverse verify
def _rcc_base_of_boxes(a: Box, b: Box, ndim: int) -> str:
    """Exact RCC-8 base relation of two non-degenerate AABBs (rectangle-algebra readback).

    Mirrors the JEPD lowering used by the geometric solver, computed here on the exact
    Fraction bounds. Returns one of the 8 base-relation names."""
    alo, ahi, blo, bhi = a.lo, a.hi, b.lo, b.hi
    # DC: separated (a strict gap) on some axis.
    if any(ahi[d] < blo[d] or bhi[d] < alo[d] for d in range(ndim)):
        return "DC"
    # Connected on every axis from here on.
    equal = all(alo[d] == blo[d] and ahi[d] == bhi[d] for d in range(ndim))
    if equal:
        return "EQ"
    a_in_b = all(alo[d] >= blo[d] and ahi[d] <= bhi[d] for d in range(ndim))
    if a_in_b:
        strict = all(alo[d] > blo[d] and ahi[d] < bhi[d] for d in range(ndim))
        return "NTPP" if strict else "TPP"
    b_in_a = all(blo[d] >= alo[d] and bhi[d] <= ahi[d] for d in range(ndim))
    if b_in_a:
        strict = all(blo[d] > alo[d] and bhi[d] < ahi[d] for d in range(ndim))
        return "NTPPi" if strict else "TPPi"
    # Neither contained. Interiors overlap on every axis => PO, else a boundary-only
    # touch remains on some axis => EC.
    iover_all = all(alo[d] < bhi[d] and blo[d] < ahi[d] for d in range(ndim))
    return "PO" if iover_all else "EC"


def _interior_overlap_axes(a: Box, b: Box, ndim: int) -> int:
    """Number of axes on which the two boxes' interiors overlap (contact dimension of EC)."""
    return sum(1 for d in range(ndim) if a.lo[d] < b.hi[d] and b.lo[d] < a.hi[d])


def _reverse_verify(
    spec: RelSpec, realization: Realization, ndim: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Cross-check a realization against the spec by reading the geometry back.

    Returns ``(satisfied_ids, violated_ids)``. A constraint is *satisfied* iff the
    RCC-8 base relation recomputed from the realized boxes is a member of the
    constraint's mask AND, for an EC pair carrying a witness, the realized contact
    dimension equals the witness's ``intended_contact_dimension``."""
    boxes = {bx.region_id: bx for bx in realization.boxes}
    # index witnesses by the canonical pair they constrain (2-region witnesses)
    wdim: dict[frozenset[str], int] = {}
    for w in spec.witnesses:
        regs = tuple(dict.fromkeys(w.incident_regions))
        for x, y in combinations(regs, 2):
            wdim.setdefault(frozenset((x, y)), int(w.intended_contact_dimension))

    satisfied: list[str] = []
    violated: list[str] = []
    for c in spec.constraints:
        cid = _cid(c)
        a = boxes.get(c.src)
        b = boxes.get(c.dst)
        if a is None or b is None:
            violated.append(cid)
            continue
        base = _rcc_base_of_boxes(a, b, ndim)
        ok = bool(rcc8.bit(base) & c.rcc8_mask)
        if ok and base == "EC":
            key = frozenset((c.src, c.dst))
            if key in wdim:
                ok = _interior_overlap_axes(a, b, ndim) == wdim[key]
        (satisfied if ok else violated).append(cid)
    return tuple(satisfied), tuple(violated)


# ---------------------------------------------------------------- public: run
def run(
    spec: RelSpec,
    groundings: Sequence[GroundingDecision] = (),
    domain: str = "aabb_2d",
    bounds: tuple = (0, 100),
    *,
    realize: bool = True,
    scenario_id: str = "",
    min_size=None,
) -> Realization:
    """Run the two-stage realizability pipeline for one grounding scenario.

    Parameters
    ----------
    spec : RelSpec
        The relational spec to realize.
    groundings : sequence of GroundingDecision
        Decisions to apply before geometry (edit-authority flips and, when the decision
        names an RCC-8 relation, mask refinement).
    domain : str
        Geometric domain, ``'aabb_2d'`` or ``'aabb_3d'``.
    bounds : (lo, hi)
        Inclusive coordinate box every AABB must live inside.
    realize : bool, keyword-only
        When False, stop after Stage A and report the symbolic rung reached
        (RCC_CONSISTENT or INCONSISTENT) without attempting geometry.
    scenario_id : str, keyword-only
        Tag propagated onto the produced Realization (used by :func:`family`).
    min_size : Fraction, optional
        Minimum edge length forwarded to the geometric solver.

    Returns
    -------
    Realization
        With ``status`` at the top ladder rung honestly reached: INCONSISTENT (Stage A
        contradiction, no geometry), RCC_CONSISTENT (Stage A passed, geometry not run),
        REALIZED_IN_D (geometry found and reverse-verified), or UNKNOWN (geometry
        unsat/unknown in D, or reverse check disagreed).
    """
    if domain not in _DOMAIN_NDIM:
        raise PipelineError(
            f"unsupported domain {domain!r}; use one of {sorted(_DOMAIN_NDIM)}"
        )
    ndim = _DOMAIN_NDIM[domain]
    rid = f"pipeline:{scenario_id}" if scenario_id else "pipeline:0"

    # ---- Stage A: symbolic consistency (fail-closed) ----
    # Malformed witnesses (unknown/duplicate regions, bad contact dim, ...) are a spec
    # error, not an unsatisfiability; refuse to realize rather than guess.
    structural = _structural_violations(spec)
    if structural:
        raise PipelineError("malformed spec: " + "; ".join(structural))

    net = _build_network(spec)
    a_status, _closed = path_consistency(net)
    wit_pairs = _witness_contradictions(spec)
    dim_conflicts = _witness_dim_conflicts(spec)

    if a_status == "inconsistent" or wit_pairs or dim_conflicts:
        # Collect the constraints implicated in the contradiction (best-effort blame).
        blame: list[str] = []
        if a_status == "inconsistent":
            blame.extend(_constraints_on_pairs(spec, unsat_core(net)))
        if wit_pairs:
            blame.extend(_constraints_on_pairs(spec, wit_pairs))
        if dim_conflicts:
            blame.extend(_constraints_on_pairs(spec, dim_conflicts))
        if not blame:  # fall back to naming every constraint if blame is empty
            blame = [_cid(c) for c in spec.constraints]
        return Realization(
            id=rid,
            boxes=(),
            status=Realizability.INCONSISTENT,
            satisfied=(),
            violated=tuple(dict.fromkeys(blame)),
            undecided=(),
            scenario_id=scenario_id,
        )

    # Stage A passed: at least path-consistent -> RCC_CONSISTENT rung.
    if not realize:
        return Realization(
            id=rid,
            boxes=(),
            status=Realizability.RCC_CONSISTENT,
            satisfied=tuple(_cid(c) for c in spec.constraints),
            violated=(),
            undecided=(),
            scenario_id=scenario_id,
        )

    # ---- Stage B: apply groundings, realize, reverse-verify ----
    grounded = _apply_groundings(spec, tuple(groundings))
    # Materialize witnessed contacts that no constraint names, so geometry actually
    # imposes (and reverse-verify actually checks) them.
    grounded = _augment_with_witness_contacts(grounded)
    real = geom_solver.solve_boxes(
        grounded, domain=domain, bounds=bounds,
        min_size=min_size, realization_id=rid,
    )

    if real.status is Realizability.REALIZED_IN_D:
        satisfied, violated = _reverse_verify(grounded, real, ndim)
        if not violated:
            return Realization(
                id=rid,
                boxes=real.boxes,
                status=Realizability.REALIZED_IN_D,
                satisfied=satisfied,
                violated=(),
                undecided=(),
                scenario_id=scenario_id,
            )
        # Solver claimed SAT but the exact readback disagrees: fail-closed to UNKNOWN.
        return Realization(
            id=rid,
            boxes=real.boxes,
            status=Realizability.UNKNOWN,
            satisfied=satisfied,
            violated=violated,
            undecided=(),
            scenario_id=scenario_id,
        )

    # Geometry UNSAT / UNKNOWN in the selected domain. Stage A already proved the spec
    # is not contradictory, so this is realized-unsat-in-D, NOT impossible -> UNKNOWN.
    return Realization(
        id=rid,
        boxes=(),
        status=Realizability.UNKNOWN,
        satisfied=(),
        violated=(),
        undecided=tuple(_cid(c) for c in grounded.constraints),
        scenario_id=scenario_id,
    )


# ---------------------------------------------------------------- public: family
def family(
    spec: RelSpec,
    scenarios: Sequence[Sequence[GroundingDecision]],
    domain: str = "aabb_2d",
    bounds: tuple = (0, 100),
    *,
    min_size=None,
) -> list[Realization]:
    """Realize ``spec`` once per grounding scenario, returning one Realization each.

    Each scenario is a set (sequence) of :class:`GroundingDecision`. Scenarios share the
    same immutable base spec -- grounding is a pure function that returns a fresh spec,
    so branches never see order-dependent mutation. Every returned Realization carries a
    distinct ``scenario_id`` (``"s0"``, ``"s1"``, ...) for traceability."""
    out: list[Realization] = []
    for i, sc in enumerate(scenarios):
        out.append(
            run(
                spec,
                groundings=tuple(sc),
                domain=domain,
                bounds=bounds,
                scenario_id=f"s{i}",
                min_size=min_size,
            )
        )
    return out
