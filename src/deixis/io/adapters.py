"""Pure-function logic layer behind the Grasshopper (Rhino) components.

Every public function here is a *pure*, Rhino-independent function that takes JSON
(a JSON *string* or an already-parsed ``dict``/``list``) and returns a JSON *string*.
Each Grasshopper Python 3 component is meant to be a thin shell that only marshals its
inputs into one of these calls and forwards the returned JSON — so all of the real
logic (and its exactness / fail-closed guarantees) lives here in tested Python, never
in a component's scriptable body.

Design commitments (inherited from the M0 contract, ``deixis.core.types``):

* **Exact only.** Coordinates round-trip as ``{"num","den"}`` (see
  :func:`deixis.core.ids.frac_to_json`); geometry read *in* from Grasshopper is coerced
  to :class:`fractions.Fraction` via :func:`deixis.core.ids.F`. No float ever reaches a
  decision.
* **Immutable.** Every editing adapter (:func:`relate`, :func:`witness`, :func:`ground`,
  …) parses a spec, builds a NEW :class:`~deixis.core.types.RelSpec`, and serializes it
  back — the input JSON is never mutated in place.
* **Honest Lift.** :func:`lift` turns realized geometry back into a spec of *observations*,
  not authority: the recovered relations are ``DELEGATED`` and every recovered contact
  witness is tagged ``epistemic = observed_tolerant``. Lift never promotes anything to
  ``INVARIANT`` — an observation is a proposal, not a commitment.
* **Fail-closed JSON errors.** Any exception is caught and returned as
  ``{"error": {"type": ..., "message": ...}}`` JSON rather than raised, so a Grasshopper
  wire always receives parseable JSON.

There is no ``deixis.io.serialize`` module in the tree yet, so the (spec | realization)
<-> JSON serialization the components need is implemented here and exposed publicly
(:func:`spec_to_json`, :func:`spec_from_json`, :func:`realization_to_json`,
:func:`realization_from_json`) so a component can round-trip without re-parsing by hand.
"""
from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from typing import Any, Optional, Sequence, Union

from deixis import pipeline
from deixis.core import rcc8
from deixis.core.ids import F, Provenance, frac_from_json, frac_to_json
from deixis.core.types import (
    Box,
    ContactDim,
    Epistemic,
    GroundingDecision,
    Invariant,
    ObservationQuery,
    OverlayAtom,
    Realizability,
    Realization,
    Region,
    RelationConstraint,
    RelSpec,
    Status,
)
from deixis.grounding import grounding_op
from deixis.incidence.witness import make_witness
from deixis.solver.family import solve_family
from deixis.verify import reverse
from deixis.verify.reverse import (
    VerifyReport,
    WitnessViolation,
    extract_contact_dimension,
    extract_relations,
)

__all__ = [
    # serialization helpers (public so GH components can round-trip)
    "spec_to_json",
    "spec_from_json",
    "realization_to_json",
    "realization_from_json",
    # editing / query adapters
    "relate",
    "witness",
    "ground",
    "unground",
    "set_invariant",
    "solve",
    "verify",
    "family",
    "lift",
]

# 'face'/'line'/'point' (or a raw int) -> ContactDim. What RCC-8's single EC collapses.
_CONTACT_DIM: dict[str, ContactDim] = {
    "face": ContactDim.FACE,
    "line": ContactDim.LINE,
    "edge": ContactDim.LINE,   # a common synonym for line contact
    "point": ContactDim.POINT,
}

JsonLike = Union[str, dict, list]


# ============================================================ JSON in/out plumbing
def _load(x: JsonLike) -> Any:
    """Accept a JSON string OR an already-parsed dict/list; return the parsed value."""
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        return json.loads(x)
    raise TypeError(f"expected JSON str or dict/list, got {type(x).__name__}")


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _error(exc: Exception) -> str:
    """Fail-closed: render any exception as an error JSON object (never raise out)."""
    return _dump({"error": {"type": type(exc).__name__, "message": str(exc)}})


def _coord(x: Any):
    """A single box coordinate -> exact Fraction.

    Accepts a ``{"num","den"}`` object (our own serialized form) via
    :func:`frac_from_json`, or a raw number/string from Grasshopper via :func:`F`
    (which coerces float through its shortest decimal — no binary-float noise)."""
    if isinstance(x, dict):
        return frac_from_json(x)
    return F(x)


def _snap(c: Fraction, t: Fraction) -> Fraction:
    """Snap an exact coordinate ``c`` to the nearest multiple of grid ``t`` (t > 0).

    Pure exact-rational arithmetic (round half up): coordinates within one grid step of
    coincidence collapse onto the same value, so a near-touch becomes an actual ``EC``
    contact rather than a spurious ``DC``. This is Lift's tolerance policy — the reason the
    recovered relations are honestly ``observed_tolerant`` and not exact."""
    half = Fraction(1, 2)
    q = c / t + half
    n = q.numerator // q.denominator  # exact floor (denominator > 0 in a Fraction)
    return n * t


# ---------------------------------------------------------------- provenance
def _prov_to_json(p: Optional[Provenance]) -> Optional[dict]:
    return p.to_json() if p is not None else None


def _prov_from_json(d: Any) -> Optional[Provenance]:
    if not d:
        return None
    return Provenance(
        origin=d.get("origin", ""),
        actor=d.get("actor", "designer"),
        activity=d.get("activity", ""),
        inputs=tuple(d.get("inputs", ())),
        note=d.get("note", ""),
    )


# ---------------------------------------------------------------- region
def _region_to_json(r: Region) -> dict:
    return {
        "id": r.id,
        "semantic_type": r.semantic_type,
        "declared_origin": r.declared_origin,
        "prov": _prov_to_json(r.prov),
    }


def _region_from_json(d: dict) -> Region:
    return Region(
        id=d["id"],
        semantic_type=d.get("semantic_type", ""),
        declared_origin=d.get("declared_origin", True),
        prov=_prov_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- constraint
def _constraint_to_json(c: RelationConstraint) -> dict:
    # rcc8_names is emitted for human/GH readability; rcc8_mask stays the source of truth.
    return {
        "id": c.id,
        "src": c.src,
        "dst": c.dst,
        "rcc8_mask": c.rcc8_mask,
        "rcc8_names": rcc8.names(c.rcc8_mask),
        "status": c.status.value,
        "prov": _prov_to_json(c.prov),
    }


def _constraint_from_json(d: dict) -> RelationConstraint:
    mask = d.get("rcc8_mask")
    if mask is None and "rcc8_names" in d:
        mask = rcc8.mask(*d["rcc8_names"])
    return RelationConstraint(
        src=d["src"],
        dst=d["dst"],
        rcc8_mask=int(mask if mask is not None else 0),
        status=Status(d.get("status", "delegated")),
        id=d.get("id", ""),
        prov=_prov_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- witness
def _witness_to_json(w) -> dict:
    return {
        "id": w.id,
        "cell_ids": list(w.cell_ids),
        "incident_regions": list(w.incident_regions),
        "intended_contact_dimension": int(w.intended_contact_dimension),
        "component_index": w.component_index,
        "boundary_role": w.boundary_role,
        "epistemic": w.epistemic.value,
        "prov": _prov_to_json(w.prov),
    }


def _witness_from_json(d: dict):
    return make_witness(
        d.get("incident_regions", ()),
        int(d["intended_contact_dimension"]),
        cell_ids=tuple(d.get("cell_ids", ())),
        id=d.get("id", ""),
        component_index=d.get("component_index", 0),
        boundary_role=d.get("boundary_role", "shared_boundary"),
        epistemic=Epistemic(d.get("epistemic", "asserted")),
        prov=_prov_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- overlay
def _overlay_to_json(o: OverlayAtom) -> dict:
    return {
        "id": o.id,
        "members": sorted(o.members),
        "nonempty": o.nonempty,
        "prov": _prov_to_json(o.prov),
    }


def _overlay_from_json(d: dict) -> OverlayAtom:
    return OverlayAtom(
        id=d["id"],
        members=frozenset(d.get("members", ())),
        nonempty=d.get("nonempty", True),
        prov=_prov_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- grounding decision
def _grounding_to_json(g: GroundingDecision) -> dict:
    return {
        "id": g.id,
        "target": g.target,
        "fixed_value_or_domain": g.fixed_value_or_domain,
        "actor": g.actor,
        "rationale": g.rationale,
        "reversibility": g.reversibility,
        "prov": _prov_to_json(g.prov),
    }


def _grounding_from_json(d: dict) -> GroundingDecision:
    return GroundingDecision(
        id=d["id"],
        target=d["target"],
        fixed_value_or_domain=d.get("fixed_value_or_domain", ""),
        actor=d.get("actor", "designer"),
        rationale=d.get("rationale", ""),
        reversibility=d.get("reversibility", "high"),
        prov=_prov_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- invariant / observation
def _invariant_to_json(inv: Invariant) -> dict:
    return {
        "relation_invariants": sorted(inv.relation_invariants),
        "witness_invariants": sorted(inv.witness_invariants),
    }


def _invariant_from_json(d: Any) -> Invariant:
    if not d:
        return Invariant()
    return Invariant(
        relation_invariants=frozenset(d.get("relation_invariants", ())),
        witness_invariants=frozenset(d.get("witness_invariants", ())),
    )


def _observation_to_json(o: ObservationQuery) -> dict:
    return {"kinds": sorted(o.kinds)}


def _observation_from_json(d: Any) -> ObservationQuery:
    if not d:
        return ObservationQuery()
    return ObservationQuery(kinds=frozenset(d.get("kinds", ())))


# ---------------------------------------------------------------- box / realization
def _box_to_json(b: Box) -> dict:
    return {
        "region_id": b.region_id,
        "lo": [frac_to_json(v) for v in b.lo],
        "hi": [frac_to_json(v) for v in b.hi],
    }


def _box_from_json(d: dict) -> Box:
    return Box(
        region_id=d["region_id"],
        lo=tuple(_coord(v) for v in d["lo"]),
        hi=tuple(_coord(v) for v in d["hi"]),
    )


def realization_to_json(real: Realization) -> dict:
    """Serialize a :class:`Realization` to a JSON-ready dict (exact box coords)."""
    return {
        "id": real.id,
        "boxes": [_box_to_json(b) for b in real.boxes],
        "status": real.status.value,
        "satisfied": list(real.satisfied),
        "violated": list(real.violated),
        "undecided": list(real.undecided),
        "scenario_id": real.scenario_id,
        "prov": _prov_to_json(real.prov),
    }


def realization_from_json(data: JsonLike) -> Realization:
    """Deserialize a :class:`Realization` from JSON (string or dict)."""
    d = _load(data)
    return Realization(
        id=d.get("id", ""),
        boxes=tuple(_box_from_json(b) for b in d.get("boxes", ())),
        status=Realizability(d.get("status", "unknown")),
        satisfied=tuple(d.get("satisfied", ())),
        violated=tuple(d.get("violated", ())),
        undecided=tuple(d.get("undecided", ())),
        scenario_id=d.get("scenario_id", ""),
        prov=_prov_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- spec
def spec_to_json(spec: RelSpec) -> dict:
    """Serialize a :class:`RelSpec` to a JSON-ready dict (round-trips exactly)."""
    return {
        "schema_version": spec.schema_version,
        "regions": [_region_to_json(r) for r in spec.regions],
        "constraints": [_constraint_to_json(c) for c in spec.constraints],
        "witnesses": [_witness_to_json(w) for w in spec.witnesses],
        "overlays": [_overlay_to_json(o) for o in spec.overlays],
        "groundings": [_grounding_to_json(g) for g in spec.groundings],
        "invariant": _invariant_to_json(spec.invariant),
        "observation": _observation_to_json(spec.observation),
    }


def spec_from_json(data: JsonLike) -> RelSpec:
    """Deserialize a :class:`RelSpec` from JSON (string or dict). Empty -> empty spec."""
    d = _load(data)
    if not d:
        return RelSpec()
    return RelSpec(
        regions=tuple(_region_from_json(r) for r in d.get("regions", ())),
        constraints=tuple(_constraint_from_json(c) for c in d.get("constraints", ())),
        witnesses=tuple(_witness_from_json(w) for w in d.get("witnesses", ())),
        overlays=tuple(_overlay_from_json(o) for o in d.get("overlays", ())),
        groundings=tuple(_grounding_from_json(g) for g in d.get("groundings", ())),
        invariant=_invariant_from_json(d.get("invariant")),
        observation=_observation_from_json(d.get("observation")),
        schema_version=d.get("schema_version", "0.0.1"),
    )


# ---------------------------------------------------------------- internal: ensure regions
def _ensure_regions(spec: RelSpec, ids: Sequence[str], *, declared: bool = True) -> RelSpec:
    """Append a :class:`Region` for every id not already present (order-stable, no dups)."""
    have = {r.id for r in spec.regions}
    extra = [
        Region(id=rid, declared_origin=declared)
        for rid in dict.fromkeys(ids)
        if rid not in have
    ]
    if not extra:
        return spec
    return replace(spec, regions=spec.regions + tuple(extra))


# ============================================================ editing adapters
def relate(
    spec_json: JsonLike,
    src: str,
    dst: str,
    rcc8_names: Sequence[str],
    status: str = "delegated",
) -> str:
    """Add or update the RCC-8 :class:`RelationConstraint` on the ordered pair ``src->dst``.

    ``rcc8_names`` is a list of base-relation names (``["EC"]``, ``["PO","EC"]``, …) folded
    into an 8-bit disjunction mask. If a constraint already exists on that ordered pair its
    mask/status are updated in place (its id and provenance chain preserved); otherwise a new
    ``DELEGATED`` (by default) constraint is appended and any missing regions are declared.
    Returns the updated spec JSON.
    """
    try:
        spec = spec_from_json(spec_json)
        mask = rcc8.mask(*rcc8_names) if rcc8_names else 0
        st = Status(status)
        spec = _ensure_regions(spec, [src, dst])
        cons = list(spec.constraints)
        for i, c in enumerate(cons):
            if c.src == src and c.dst == dst:
                # Respect the edit-authority lattice: relate is a plain (delegated) edit
                # and must NOT silently rewrite a protected constraint. Grounded/invariant
                # constraints are only movable through the guarded grounding API
                # (ground/unground/set_invariant) -- fail closed otherwise.
                if c.status is not Status.DELEGATED:
                    raise ValueError(
                        f"constraint {c.id!r} on ({src},{dst}) is {c.status.value}, not "
                        f"delegated; use unground/grounding_op to change a protected relation"
                    )
                cons[i] = replace(c, rcc8_mask=mask, status=st)
                break
        else:
            cons.append(
                RelationConstraint(
                    src=src,
                    dst=dst,
                    rcc8_mask=mask,
                    status=st,
                    id=f"{src}->{dst}",
                    prov=Provenance(origin="asserted", activity="relate"),
                )
            )
        return _dump(spec_to_json(replace(spec, constraints=tuple(cons))))
    except Exception as exc:  # fail-closed JSON error
        return _error(exc)


def witness(
    spec_json: JsonLike,
    region_a: str,
    region_b: str,
    contact_dim: Union[str, int],
) -> str:
    """Add an :class:`IncidenceWitness` recording *how* ``region_a`` and ``region_b`` meet.

    ``contact_dim`` is ``'face'`` / ``'line'`` (``'edge'``) / ``'point'`` (or the raw int
    2/1/0) — the geometry-independent contact-dimension *requirement* that RCC-8's single
    ``EC`` collapses. Built with :func:`deixis.incidence.witness.make_witness` (deterministic
    id, ``epistemic = asserted``). Missing regions are declared. Idempotent: re-adding the
    *same* witness (same regions/dim/component -> same deterministic id) with a matching
    contact dimension is a no-op rather than a duplicate; a conflicting contact dimension on
    an existing witness id is rejected fail-closed. Returns the updated spec JSON.
    """
    try:
        spec = spec_from_json(spec_json)
        if isinstance(contact_dim, str):
            key = contact_dim.strip().lower()
            if key not in _CONTACT_DIM:
                raise ValueError(
                    f"contact_dim {contact_dim!r} must be one of "
                    f"{sorted(_CONTACT_DIM)} or an int 0/1/2"
                )
            dim = _CONTACT_DIM[key]
        else:
            dim = ContactDim(int(contact_dim))
        spec = _ensure_regions(spec, [region_a, region_b])
        w = make_witness([region_a, region_b], dim)
        existing = next((x for x in spec.witnesses if x.id == w.id), None)
        if existing is not None:
            # Same deterministic id already present: idempotent no-op if it agrees, else a
            # genuine conflict (a pair cannot be both e.g. a line and a point contact).
            if existing.intended_contact_dimension != w.intended_contact_dimension:
                raise ValueError(
                    f"witness {w.id!r} already exists with contact dimension "
                    f"{existing.intended_contact_dimension}, conflicting with {int(dim)}"
                )
            return _dump(spec_to_json(spec))
        return _dump(spec_to_json(replace(spec, witnesses=spec.witnesses + (w,))))
    except Exception as exc:
        return _error(exc)


def ground(
    spec_json: JsonLike,
    target_id: str,
    fixed: str,
    actor: str = "designer",
    rationale: str = "",
) -> str:
    """Ground a delegated constraint NOW (P -> G) via :func:`grounding_op.ground`.

    Wraps ``fixed`` (the fixed value/domain, e.g. ``"EC"`` or ``"x==4.20"``) into a fresh
    :class:`GroundingDecision` (deterministic id derived from the target and current ledger
    length) and applies it fail-closed — grounding an INVARIANT or non-DELEGATED target
    returns an error JSON, never a corrupted spec. Returns the updated spec JSON.
    """
    try:
        spec = spec_from_json(spec_json)
        decision = GroundingDecision(
            id=f"g:{target_id}:{len(spec.groundings)}",
            target=target_id,
            fixed_value_or_domain=str(fixed),
            actor=actor,
            rationale=rationale,
        )
        return _dump(spec_to_json(grounding_op.ground(spec, target_id, decision)))
    except Exception as exc:
        return _error(exc)


def unground(spec_json: JsonLike, decision_id: str) -> str:
    """Roll back a grounding (G -> P) via :func:`grounding_op.unground`, removing the decision.

    Fail-closed against unknown/invariant targets. Returns the updated spec JSON.
    """
    try:
        spec = spec_from_json(spec_json)
        return _dump(spec_to_json(grounding_op.unground(spec, decision_id)))
    except Exception as exc:
        return _error(exc)


def set_invariant(spec_json: JsonLike, constraint_id: str) -> str:
    """Promote a constraint into the un-touchable core (status -> INVARIANT).

    Idempotent; registers the id in ``spec.invariant.relation_invariants`` too. Returns the
    updated spec JSON.
    """
    try:
        spec = spec_from_json(spec_json)
        return _dump(spec_to_json(grounding_op.set_invariant(spec, constraint_id)))
    except Exception as exc:
        return _error(exc)


# ============================================================ solve / verify / family
def solve(
    spec_json: JsonLike,
    domain: str = "aabb_2d",
    bounds: Sequence = (0, 100),
    groundings: Optional[Sequence[dict]] = None,
) -> str:
    """Run the fail-closed two-stage pipeline (:func:`deixis.pipeline.run`) for one scenario.

    ``groundings`` (optional) is a list of serialized :class:`GroundingDecision` dicts applied
    before geometry. Returns the realization JSON, whose ``status`` sits at the top ladder rung
    honestly reached (``inconsistent`` / ``rcc_consistent`` / ``realized_in_selected_domain_D``
    / ``unknown``).
    """
    try:
        spec = spec_from_json(spec_json)
        gds = [_grounding_from_json(g) for g in (groundings or [])]
        # Coerce bounds to exact Fraction at the adapter boundary (via F / frac_from_json):
        # a float bound from Grasshopper must enter through the shortest-decimal conversion,
        # never as a raw binary float the solver would Fraction() with full float noise.
        raw_bounds = tuple(bounds) if bounds is not None else (0, 100)
        b = tuple(_coord(v) for v in raw_bounds)
        real = pipeline.run(spec, groundings=gds, domain=domain, bounds=b)
        return _dump(realization_to_json(real))
    except Exception as exc:
        return _error(exc)


def _witness_violation_to_json(v: WitnessViolation) -> dict:
    return {
        "witness_id": v.witness_id,
        "code": v.code,
        "regions": list(v.regions),
        "intended": v.intended,
        "actual": v.actual,
        "detail": v.detail,
    }


def _report_to_json(rep: VerifyReport) -> dict:
    return {
        "satisfied": list(rep.satisfied),
        "violated": list(rep.violated),
        "undecided": list(rep.undecided),
        "witness_violations": [
            _witness_violation_to_json(v) for v in rep.witness_violations
        ],
    }


def verify(spec_json: JsonLike, realization_json: JsonLike) -> str:
    """Reverse-verify a realization against its spec (:func:`deixis.verify.reverse.verify`).

    Reads the RCC-8 relation (and, for EC pairs carrying a witness, the realized contact
    dimension) straight off the exact box coordinates and reports
    ``satisfied`` / ``violated`` / ``undecided`` constraint ids plus any ``witness_violations``
    (e.g. face demanded / edge realized). Returns the report JSON.
    """
    try:
        spec = spec_from_json(spec_json)
        real = realization_from_json(realization_json)
        return _dump(_report_to_json(reverse.verify(spec, real)))
    except Exception as exc:
        return _error(exc)


def family(spec_json: JsonLike, scenarios: Sequence[Sequence[dict]]) -> str:
    """Realize ``spec`` once per grounding scenario (:func:`deixis.solver.family.solve_family`).

    ``scenarios`` is a list of scenarios; each scenario is a list of serialized
    :class:`GroundingDecision` dicts. All scenarios branch off the same immutable base spec.
    Returns a JSON *array* of realization objects (one per scenario, tagged ``s0``, ``s1`` …),
    each carrying the family provenance ledger in its ``prov.note``.
    """
    try:
        spec = spec_from_json(spec_json)
        scen = [
            [_grounding_from_json(g) for g in (sc or [])] for sc in (scenarios or [])
        ]
        reals = solve_family(spec, scen)
        return _dump([realization_to_json(r) for r in reals])
    except Exception as exc:
        return _error(exc)


# ============================================================ lift (geometry -> observation)
def lift(
    boxes_json: JsonLike,
    region_ids: Sequence[str],
    tolerance: Any = 0,
) -> str:
    """Observe realized geometry (a set of boxes) back into a relational spec — a *proposal*.

    Reads every ordered pair of the given ``region_ids`` with
    :func:`deixis.verify.reverse.extract_relations` (exact RCC-8 base relation) and, for each
    ``EC`` (boundary-touch) pair, its contact dimension with
    :func:`extract_contact_dimension`. The recovered relations are emitted as ``DELEGATED``
    :class:`RelationConstraint`\\ s (never promoted to invariant) and every recovered contact
    becomes an :class:`IncidenceWitness` tagged ``epistemic = observed_tolerant``: Lift is an
    *observation*, not authority. When ``tolerance > 0`` every coordinate is first snapped
    (exact round-half-up) onto a grid of that step, so near-coincident endpoints from Rhino
    collapse and a near-touch is observed as a real ``EC`` contact — this snap is what makes
    the recovered relations honestly *tolerant*. ``tolerance = 0`` is exact classification;
    the ``tolerance`` used is recorded in provenance (exact-integer boxes could instead
    justify ``derived_exact``, but that is not auto-assumed here).

    ``boxes_json`` is a list of box objects (or ``{"boxes": [...]}``); each box is
    ``{"region_id", "lo":[...], "hi":[...]}`` with coordinates coerced to exact Fraction.
    Returns the observed-spec JSON.
    """
    try:
        raw = _load(boxes_json)
        box_dicts = raw.get("boxes", []) if isinstance(raw, dict) else raw
        rid_set = set(region_ids) if region_ids else {b["region_id"] for b in box_dicts}
        tol = _coord(tolerance)
        if tol < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance!r}")

        def _mk_box(bd: dict) -> Box:
            box = _box_from_json(bd)
            if tol == 0:
                return box
            lo = tuple(_snap(v, tol) for v in box.lo)
            hi = tuple(_snap(v, tol) for v in box.hi)
            for d, (l, h) in enumerate(zip(lo, hi)):
                # Snapping must not collapse a box: a zero-width axis is not a region.
                if not l < h:
                    raise ValueError(
                        f"box {box.region_id!r} collapsed on axis {d} at tolerance "
                        f"{tolerance!r} (snapped lo={l} hi={h}); tolerance too coarse"
                    )
            return Box(region_id=box.region_id, lo=lo, hi=hi)

        boxes = [_mk_box(b) for b in box_dicts if b["region_id"] in rid_set]
        real = Realization(id="lift", boxes=tuple(boxes))

        note = f"observed_tolerant;tolerance={tolerance}"
        prov = Provenance(
            origin="lift",
            actor="observer",
            activity="lift",
            inputs=tuple(sorted(rid_set)),
            note=note,
        )
        regions = tuple(
            # geometry-origin, not designer-declared -> declared_origin=False (honest)
            Region(id=r, declared_origin=False, prov=prov)
            for r in sorted(rid_set)
        )

        rels = extract_relations(real)  # {(i,j): single-bit mask}, i,j in box order
        ec = rcc8.bit("EC")
        constraints: list[RelationConstraint] = []
        witnesses: list = []
        for (a, b), mask in rels.items():
            constraints.append(
                RelationConstraint(
                    src=a,
                    dst=b,
                    rcc8_mask=mask,
                    status=Status.DELEGATED,   # observation, not a commitment
                    id=f"lift:{a}->{b}",
                    prov=prov,
                )
            )
            if mask == ec:
                dim = int(extract_contact_dimension(real, a, b))
                witnesses.append(
                    make_witness(
                        [a, b],
                        dim,
                        epistemic=Epistemic.OBSERVED_TOLERANT,
                        prov=prov,
                    )
                )

        spec = RelSpec(
            regions=regions,
            constraints=tuple(constraints),
            witnesses=tuple(witnesses),
            observation=ObservationQuery(),
        )
        return _dump(spec_to_json(spec))
    except Exception as exc:
        return _error(exc)
