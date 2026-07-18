"""M0: the shared type contract for the Witnessed Relational Spatial IR.

This module is the *fixed interface* every other module builds against. It encodes
the adversarially-verified data model (v7 spec §14):

  Region / RelationConstraint(status) / Cell(dim) / IncidenceWitness /
  OverlayAtom / GroundingDecision / SolverContract / Realization / Invariant /
  ObservationQuery / RelSpec (the container).

Key honest commitments baked into the types:
- RCC-8 relations are 8-bit disjunction *masks* (see deixis.core.rcc8 for the algebra).
- ``epistemic_status`` distinguishes asserted / derived_exact / observed_tolerant / undecided
  (Lift observations must NOT be auto-promoted to invariant).
- Realizability is a *fail-closed* ladder of separate statuses, NOT a single boolean:
  abstract < rcc_consistent < realized_in_selected_domain_D < fabrication_valid.
- All coordinates are exact Fraction.
- Objects are frozen (immutable); grounding operations return NEW RelSpec (see grounding_op).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
from typing import Optional

from .ids import Provenance

# ---------------------------------------------------------------- enums / status

class Status(str, Enum):
    """Edit-authority / commitment of a relation constraint (deferred grounding I/G/P)."""
    INVARIANT = "invariant"    # I: never changes; fail-closed guard protects it
    GROUNDED = "grounded"      # G: fixed *now* by a GroundingDecision
    DELEGATED = "delegated"    # P: left to the solver


class Epistemic(str, Enum):
    ASSERTED = "asserted"                # user wrote it directly as spec
    DERIVED_EXACT = "derived_exact"      # exactly derived from an integer/rational model
    OBSERVED_TOLERANT = "observed_tolerant"  # tolerance-aware observation from Rhino geometry (Lift)
    UNDECIDED = "undecided"              # not decidable from numeric geometry


class Realizability(str, Enum):
    """Fail-closed ladder. Downstream output allowed only at the top rung reached."""
    ABSTRACT = "abstract"
    RCC_CONSISTENT = "rcc_consistent"
    REALIZED_IN_D = "realized_in_selected_domain_D"
    FABRICATION_VALID = "fabrication_valid"
    INCONSISTENT = "inconsistent"
    UNKNOWN = "unknown"


class ContactDim(int, Enum):
    """Contact dimension a witness records (what RCC-8's single EC symbol collapses)."""
    POINT = 0
    LINE = 1
    FACE = 2


# ---------------------------------------------------------------- core objects

@dataclass(frozen=True)
class Region:
    id: str
    semantic_type: str = ""            # "living", "circulation", ... (may be empty)
    declared_origin: bool = True       # designer-declared vs field-origin (threshold projection)
    prov: Optional[Provenance] = None


@dataclass(frozen=True)
class RelationConstraint:
    """A qualitative relation ``src (rcc8_mask) dst`` with an edit-authority status.

    ``rcc8_mask`` is an 8-bit disjunction (see deixis.core.rcc8). status is only ever
    changed by grounding_op (never by mutating this frozen object)."""
    src: str                            # Region id
    dst: str                            # Region id
    rcc8_mask: int                      # disjunction of base relations, 0..255
    status: Status = Status.DELEGATED
    id: str = ""
    prov: Optional[Provenance] = None


@dataclass(frozen=True)
class Cell:
    """A cell in the shared witnessed cell complex K (dim 0=vertex,1=edge,2=face,3=box)."""
    id: str                             # STABLE id (identity anchor across the whole IR)
    dim: int


@dataclass(frozen=True)
class IncidenceWitness:
    """Evidence that distinguishes *how* two (or more) regions meet — what RCC-8 collapses.

    A witness is a connected contact sub-complex. ``intended_contact_dimension`` is a
    geometry-independent *requirement* (face=2 / line=1 / point=0). It is held BEFORE any
    geometry exists — this is the central superiority of the IR over RCC-8-only tools."""
    id: str
    cell_ids: tuple[str, ...]                     # cells forming the connected contact sub-complex
    incident_regions: tuple[str, ...]             # region ids meeting at this witness (>=2; may be >2)
    intended_contact_dimension: int               # ContactDim value (requirement)
    component_index: int = 0                       # which connected contact component
    boundary_role: str = "shared_boundary"
    epistemic: Epistemic = Epistemic.ASSERTED
    prov: Optional[Provenance] = None


@dataclass(frozen=True)
class OverlayAtom:
    """A non-empty overlap region identified by its membership signature sigma(x)={i in members}.

    Overlay atoms form a *poset* (by signature inclusion) + a non-empty predicate — NOT a
    meet-semilattice (realized signatures are not closed under intersection)."""
    id: str
    members: frozenset[str]              # region ids whose overlap this atom occupies
    nonempty: bool = True
    prov: Optional[Provenance] = None


@dataclass(frozen=True)
class GroundingDecision:
    """A typed, externalized decision that fixes some freedom NOW (turns delegated->grounded).

    Records who/why/when and whether it can be reversed — this is the decision-rights ledger.
    NOTE: thresholding a field into a region is *also* a GroundingDecision (field->room)."""
    id: str
    target: str                          # id of the RelationConstraint / Region / Cell being grounded
    fixed_value_or_domain: str           # human/JSON description of what is fixed (e.g. "x==4.20", "south")
    actor: str = "designer"
    rationale: str = ""
    reversibility: str = "high"          # high | med | low
    prov: Optional[Provenance] = None


@dataclass(frozen=True)
class SolverContract:
    domain: str = "aabb_2d"              # aabb_2d | aabb_3d | orthopolytope
    solver: str = "z3"                   # z3 | cp-sat
    objectives: tuple[str, ...] = ()
    completeness_claim: str = "sound_within_selected_domain"
    timeout_ms: int = 10000


@dataclass(frozen=True)
class Box:
    """An axis-aligned box with EXACT rational bounds (the selected geometric domain)."""
    region_id: str
    lo: tuple[Fraction, ...]             # (x0,y0[,z0])
    hi: tuple[Fraction, ...]             # (x1,y1[,z1])


@dataclass(frozen=True)
class Realization:
    """A concrete geometry produced by the solver, with what it satisfied/violated."""
    id: str
    boxes: tuple[Box, ...] = ()
    status: Realizability = Realizability.UNKNOWN
    satisfied: tuple[str, ...] = ()      # constraint ids
    violated: tuple[str, ...] = ()
    undecided: tuple[str, ...] = ()
    scenario_id: str = ""                # which grounding scenario produced this (Family branch)
    prov: Optional[Provenance] = None


@dataclass(frozen=True)
class ObservationQuery:
    """The explicit, pre-registered set O relative to which preservation D(E(x)) ~=_O x holds.

    Preservation claims are ONLY meaningful relative to a pre-registered O (guard against
    shrinking O post-hoc). Minimal O must include RCC-8 relation, contact dimension, cell-id
    identity, atom occupancy."""
    kinds: frozenset[str] = frozenset({"rcc8", "contact_dimension", "cell_identity", "atom_occupancy"})


@dataclass(frozen=True)
class Invariant:
    """The un-touchable core I: the subset of constraints/witnesses marked invariant."""
    relation_invariants: frozenset[str] = frozenset()   # RelationConstraint ids with status INVARIANT
    witness_invariants: frozenset[str] = frozenset()    # IncidenceWitness ids required to hold


@dataclass(frozen=True)
class RelSpec:
    """The container S = (regions, constraints, witnesses, overlays, grounding, ...).

    Immutable: every edit returns a NEW RelSpec (grounding_op enforces this so branched
    GH wires never see order-dependent mutation). Field-view (DEC cochains etc.) is an
    optional extension carried alongside — see deixis.field (M-field), kept out of the core.
    """
    regions: tuple[Region, ...] = ()
    constraints: tuple[RelationConstraint, ...] = ()
    witnesses: tuple[IncidenceWitness, ...] = ()
    overlays: tuple[OverlayAtom, ...] = ()
    groundings: tuple[GroundingDecision, ...] = ()
    invariant: Invariant = field(default_factory=Invariant)
    observation: ObservationQuery = field(default_factory=ObservationQuery)
    schema_version: str = "0.0.1"

    # -- convenience lookups (do not mutate) --
    def region(self, rid: str) -> Optional[Region]:
        return next((r for r in self.regions if r.id == rid), None)

    def constraint(self, cid: str) -> Optional[RelationConstraint]:
        return next((c for c in self.constraints if c.id == cid), None)

    def witness(self, wid: str) -> Optional[IncidenceWitness]:
        return next((w for w in self.witnesses if w.id == wid), None)

    def with_constraints(self, constraints: tuple[RelationConstraint, ...]) -> "RelSpec":
        return replace(self, constraints=constraints)

    def with_groundings(self, groundings: tuple[GroundingDecision, ...]) -> "RelSpec":
        return replace(self, groundings=groundings)
