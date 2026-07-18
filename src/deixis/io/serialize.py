"""Canonical JSON serialization for the Deixis IR (level-2 transport / GH distribution).

Design commitments (from the adversarially-verified spec):
- Fractions are ALWAYS carried as ``{"num", "den"}`` (via core.ids.frac_to_json /
  frac_from_json) — never collapsed to float. Exactness survives the round trip.
- Enums (Status / Epistemic / Realizability / ContactDim) go out as their value.
  Status/Epistemic/Realizability are string values; ContactDim / intended_contact_dimension
  are plain ints (IncidenceWitness stores the contact dimension as ``int``, not the enum).
- frozensets (OverlayAtom.members, Invariant.*_invariants, ObservationQuery.kinds) become
  SORTED lists (determinism); tuples become lists.
- ``dumps`` is deterministic (sort_keys) so ``content_hash`` is stable for equal specs.

The central guarantee, exercised in tests: for every type T,
``T_from_json(T_to_json(x))`` reconstructs an object equal to ``x`` (all fields, with
Fractions bit-for-bit identical).
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any, Optional

from ..core.ids import Provenance, frac_to_json, frac_from_json
from ..core.types import (
    Status, Epistemic, Realizability, ContactDim,
    Region, RelationConstraint, Cell, IncidenceWitness, OverlayAtom,
    GroundingDecision, SolverContract, Box, Realization, ObservationQuery,
    Invariant, RelSpec,
)

# On-wire schema tag, carried inside every RelSpec/Realization envelope so a reader can
# inspect provenance of the format. (No automatic refusal/migration is performed here;
# the field is recorded and round-tripped verbatim.)
SCHEMA_VERSION = "0.0.1"


# ---------------------------------------------------------------- small helpers

def _fr(fr: Fraction) -> dict:
    return frac_to_json(fr)


def _fr_tuple(t: tuple[Fraction, ...]) -> list:
    return [frac_to_json(x) for x in t]


def _frac_from(x: Any) -> Fraction:
    """Decode one coordinate, enforcing the exact ``{num,den}`` wire commitment.

    A bare JSON ``float`` on the wire is a lossy binary approximation and is rejected
    (the encoder never emits one) rather than silently reconstructed. dict {num,den},
    int, and rational strings ("1/3") remain accepted for robustness."""
    if isinstance(x, float):
        raise TypeError(
            "float coordinate on the wire violates the exact {num,den} contract"
        )
    return frac_from_json(x)


def _fr_tuple_from(seq: Any) -> tuple[Fraction, ...]:
    return tuple(_frac_from(x) for x in (seq or ()))


def _sorted_list(fs: Any) -> list:
    """frozenset/iterable of hashables -> sorted list (deterministic)."""
    return sorted(fs)


# ---------------------------------------------------------------- Provenance

def provenance_to_json(p: Optional[Provenance]) -> Optional[dict]:
    if p is None:
        return None
    return {
        "origin": p.origin,
        "actor": p.actor,
        "activity": p.activity,
        "inputs": list(p.inputs),
        "note": p.note,
    }


def provenance_from_json(d: Any) -> Optional[Provenance]:
    if d is None:
        return None
    return Provenance(
        origin=d.get("origin", ""),
        actor=d.get("actor", "designer"),
        activity=d.get("activity", ""),
        inputs=tuple(d.get("inputs", ()) or ()),
        note=d.get("note", ""),
    )


# ---------------------------------------------------------------- Region

def region_to_json(r: Region) -> dict:
    return {
        "id": r.id,
        "semantic_type": r.semantic_type,
        "declared_origin": r.declared_origin,
        "prov": provenance_to_json(r.prov),
    }


def region_from_json(d: Any) -> Region:
    return Region(
        id=d["id"],
        semantic_type=d.get("semantic_type", ""),
        declared_origin=d.get("declared_origin", True),
        prov=provenance_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- RelationConstraint

def relation_constraint_to_json(c: RelationConstraint) -> dict:
    from deixis.core import rcc8  # local import: rcc8_mask is authoritative; names are derived
    return {
        "src": c.src,
        "dst": c.dst,
        "rcc8_mask": c.rcc8_mask,
        # human-readable convenience, always derivable from rcc8_mask (ignored on decode)
        "rcc8_names": sorted(rcc8.names(c.rcc8_mask)),
        "status": c.status.value,
        "id": c.id,
        "prov": provenance_to_json(c.prov),
    }


def relation_constraint_from_json(d: Any) -> RelationConstraint:
    return RelationConstraint(
        src=d["src"],
        dst=d["dst"],
        rcc8_mask=d["rcc8_mask"],
        status=Status(d.get("status", Status.DELEGATED.value)),
        id=d.get("id", ""),
        prov=provenance_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- Cell

def cell_to_json(c: Cell) -> dict:
    return {"id": c.id, "dim": c.dim}


def cell_from_json(d: Any) -> Cell:
    return Cell(id=d["id"], dim=d["dim"])


# ---------------------------------------------------------------- IncidenceWitness

def witness_to_json(w: IncidenceWitness) -> dict:
    return {
        "id": w.id,
        "cell_ids": list(w.cell_ids),
        "incident_regions": list(w.incident_regions),
        "intended_contact_dimension": int(w.intended_contact_dimension),
        "component_index": w.component_index,
        "boundary_role": w.boundary_role,
        "epistemic": w.epistemic.value,
        "prov": provenance_to_json(w.prov),
    }


def witness_from_json(d: Any) -> IncidenceWitness:
    return IncidenceWitness(
        id=d["id"],
        cell_ids=tuple(d.get("cell_ids", ()) or ()),
        incident_regions=tuple(d.get("incident_regions", ()) or ()),
        intended_contact_dimension=int(d["intended_contact_dimension"]),
        component_index=d.get("component_index", 0),
        boundary_role=d.get("boundary_role", "shared_boundary"),
        epistemic=Epistemic(d.get("epistemic", Epistemic.ASSERTED.value)),
        prov=provenance_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- OverlayAtom

def overlay_to_json(o: OverlayAtom) -> dict:
    return {
        "id": o.id,
        "members": _sorted_list(o.members),
        "nonempty": o.nonempty,
        "prov": provenance_to_json(o.prov),
    }


def overlay_from_json(d: Any) -> OverlayAtom:
    return OverlayAtom(
        id=d["id"],
        members=frozenset(d.get("members", ()) or ()),
        nonempty=d.get("nonempty", True),
        prov=provenance_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- GroundingDecision

def grounding_to_json(g: GroundingDecision) -> dict:
    return {
        "id": g.id,
        "target": g.target,
        "fixed_value_or_domain": g.fixed_value_or_domain,
        "actor": g.actor,
        "rationale": g.rationale,
        "reversibility": g.reversibility,
        "prov": provenance_to_json(g.prov),
    }


def grounding_from_json(d: Any) -> GroundingDecision:
    return GroundingDecision(
        id=d["id"],
        target=d["target"],
        fixed_value_or_domain=d.get("fixed_value_or_domain", ""),
        actor=d.get("actor", "designer"),
        rationale=d.get("rationale", ""),
        reversibility=d.get("reversibility", "high"),
        prov=provenance_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- SolverContract

def solver_contract_to_json(s: SolverContract) -> dict:
    return {
        "domain": s.domain,
        "solver": s.solver,
        "objectives": list(s.objectives),
        "completeness_claim": s.completeness_claim,
        "timeout_ms": s.timeout_ms,
    }


def solver_contract_from_json(d: Any) -> SolverContract:
    return SolverContract(
        domain=d.get("domain", "aabb_2d"),
        solver=d.get("solver", "z3"),
        objectives=tuple(d.get("objectives", ()) or ()),
        completeness_claim=d.get("completeness_claim", "sound_within_selected_domain"),
        timeout_ms=d.get("timeout_ms", 10000),
    )


# ---------------------------------------------------------------- Invariant

def invariant_to_json(inv: Invariant) -> dict:
    return {
        "relation_invariants": _sorted_list(inv.relation_invariants),
        "witness_invariants": _sorted_list(inv.witness_invariants),
    }


def invariant_from_json(d: Any) -> Invariant:
    if d is None:
        return Invariant()
    return Invariant(
        relation_invariants=frozenset(d.get("relation_invariants", ()) or ()),
        witness_invariants=frozenset(d.get("witness_invariants", ()) or ()),
    )


# ---------------------------------------------------------------- ObservationQuery

def observation_to_json(o: ObservationQuery) -> dict:
    return {"kinds": _sorted_list(o.kinds)}


def observation_from_json(d: Any) -> ObservationQuery:
    if d is None:
        return ObservationQuery()
    return ObservationQuery(kinds=frozenset(d.get("kinds", ()) or ()))


# ---------------------------------------------------------------- Box

def box_to_json(b: Box) -> dict:
    return {
        "region_id": b.region_id,
        "lo": _fr_tuple(b.lo),
        "hi": _fr_tuple(b.hi),
    }


def box_from_json(d: Any) -> Box:
    return Box(
        region_id=d["region_id"],
        lo=_fr_tuple_from(d.get("lo", ())),
        hi=_fr_tuple_from(d.get("hi", ())),
    )


# ---------------------------------------------------------------- Realization

def realization_to_json(real: Realization) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": real.id,
        "boxes": [box_to_json(b) for b in real.boxes],
        "status": real.status.value,
        "satisfied": list(real.satisfied),
        "violated": list(real.violated),
        "undecided": list(real.undecided),
        "scenario_id": real.scenario_id,
        "prov": provenance_to_json(real.prov),
    }


def realization_from_json(d: Any) -> Realization:
    return Realization(
        id=d["id"],
        boxes=tuple(box_from_json(b) for b in (d.get("boxes", ()) or ())),
        status=Realizability(d.get("status", Realizability.UNKNOWN.value)),
        satisfied=tuple(d.get("satisfied", ()) or ()),
        violated=tuple(d.get("violated", ()) or ()),
        undecided=tuple(d.get("undecided", ()) or ()),
        scenario_id=d.get("scenario_id", ""),
        prov=provenance_from_json(d.get("prov")),
    )


# ---------------------------------------------------------------- RelSpec

def relspec_to_json(spec: RelSpec) -> dict:
    return {
        "schema_version": spec.schema_version,
        "regions": [region_to_json(r) for r in spec.regions],
        "constraints": [relation_constraint_to_json(c) for c in spec.constraints],
        "witnesses": [witness_to_json(w) for w in spec.witnesses],
        "overlays": [overlay_to_json(o) for o in spec.overlays],
        "groundings": [grounding_to_json(g) for g in spec.groundings],
        "invariant": invariant_to_json(spec.invariant),
        "observation": observation_to_json(spec.observation),
    }


def relspec_from_json(d: Any) -> RelSpec:
    return RelSpec(
        regions=tuple(region_from_json(x) for x in (d.get("regions", ()) or ())),
        constraints=tuple(relation_constraint_from_json(x) for x in (d.get("constraints", ()) or ())),
        witnesses=tuple(witness_from_json(x) for x in (d.get("witnesses", ()) or ())),
        overlays=tuple(overlay_from_json(x) for x in (d.get("overlays", ()) or ())),
        groundings=tuple(grounding_from_json(x) for x in (d.get("groundings", ()) or ())),
        invariant=invariant_from_json(d.get("invariant")),
        observation=observation_from_json(d.get("observation")),
        schema_version=d.get("schema_version", "0.0.1"),
    )


# ---------------------------------------------------------------- string codec

def dumps(obj: Any) -> str:
    """Deterministic canonical JSON string (sorted keys, compact separators).

    ``obj`` may be an already-serialized dict/list, or a RelSpec/Realization (auto-encoded).
    Determinism (sort_keys + stable separators) is what makes ``content_hash`` reproducible.
    """
    if isinstance(obj, RelSpec):
        obj = relspec_to_json(obj)
    elif isinstance(obj, Realization):
        obj = realization_to_json(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def loads(s: str) -> Any:
    """Inverse of the JSON encoding (returns plain dict/list, not a typed object)."""
    return json.loads(s)


# ---------------------------------------------------------------- content hash

def content_hash(spec: RelSpec) -> str:
    """sha256 hex digest of the canonical serialization of ``spec``.

    Equal specs -> equal hash; any field difference -> (with overwhelming probability)
    a different hash. Stable across processes because the encoding is deterministic and
    Fractions are exact (no float noise)."""
    payload = dumps(relspec_to_json(spec))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
