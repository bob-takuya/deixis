"""demo7 — the strongest comparison case: one full round-trip, seven stages, headless.

This is the integration demo the whole thesis argues toward. It walks **one complete
round-trip** — geometry -> witnessed spec -> approval -> deferred grounding -> family
of realizations -> exact + reverse verification — and then puts the Witnessed
Relational IR head-to-head against an **RCC-8-only** description of the *same* scene,
showing exactly what RCC-8 cannot say.

The scene has three boxes A, B, C in 3D. Two *qualitatively different* contacts:

    A - B : FACE contact  (they share a 2-cell)   intended_contact_dimension = FACE(2)
    B - C : EDGE contact  (they share a 1-cell)   intended_contact_dimension = LINE(1)

In RCC-8 **both** contacts are the single symbol ``EC`` (externally connected). RCC-8
alone therefore cannot tell a face meet from an edge meet — the exact information a
designer routinely means and must preserve. The Witnessed IR keeps them apart with an
:class:`~deixis.core.types.IncidenceWitness` per contact, and this demo shows the
consequence at every stage of the pipeline.

Seven stages (each has a ``stage_N_*`` function and a matching assertion in
``tests/test_demo7.py``):

  (1) :func:`build_original_geometry` — the *original* exact-rational geometry: a box
      group where A-B is a face contact and B-C is an edge contact (all ``Fraction``).
  (2) :func:`lift` — observe that geometry. **Both** contacts read back as ``EC`` in
      RCC-8, yet they are lifted to *distinct* witnesses (A-B dim 2, B-C dim 1). The
      observation is ``OBSERVED_TOLERANT`` — it is NOT auto-promoted to the core.
  (3) :func:`approve` — the designer *approves* the two witnesses (and their ``EC``
      relations) into the un-touchable invariant core
      (``set_invariant`` + ``witness_invariants``). Approval is an explicit act.
  (4) :func:`grounding_scenarios` — against the *same* invariant spec, two different
      grounding sets are swapped in (orientation vs. basepoint, plus how the *delegated*
      A-C freedom is fixed). The invariant core is fail-closed against grounding.
  (5) :func:`solve` — :func:`deixis.solver.family.solve_family` realizes the spec once
      per scenario: a *family* of genuinely different solutions (A-C is ``DC`` in one,
      ``PO`` in the other) that nonetheless share the invariant contacts.
  (6) :func:`verify_solution` — each solution is checked two ways that agree: the exact
      reverse layer (:func:`deixis.verify.reverse.verify`) reports **no** witness
      violation, and box read-back recovers contact dimension 2 for A-B and 1 for B-C.
      :func:`deixis.solver.family.compare_realizations` confirms the two solutions
      *preserve* the witnessed contacts while only the grounded A-C part changed.
  (7) :func:`rcc8_only_cannot_distinguish` — strip the witnesses (the RCC-8-only view)
      and feed it an *edge*-contact geometry where a *face* was required: RCC-8-only
      **accepts** it (it is just ``EC``), while the witnessed spec **rejects** it
      (``contact_dim_mismatch``). This is the concrete failure the IR repairs.

Everything is exact: coordinates are :class:`fractions.Fraction`, RCC-8 read-back is
integer bit arithmetic, contact dimension is an exact rational endpoint comparison. No
float enters any decision. This module *uses* the fixed cores (``deixis.core``,
``deixis.incidence.witness``, ``deixis.solver.geom_solver`` / ``family``,
``deixis.grounding.grounding_op``, ``deixis.verify.reverse``, ``deixis.pipeline``) and
edits none of them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Optional, Sequence

from deixis.core import rcc8
from deixis.core.ids import F
from deixis.core.types import (
    Box,
    ContactDim,
    Epistemic,
    GroundingDecision,
    Invariant,
    Realizability,
    Realization,
    Region,
    RelationConstraint,
    RelSpec,
    Status,
)
from deixis.grounding.grounding_op import set_invariant
from deixis.incidence.witness import (
    collapses_to_same_rcc,
    make_witness,
    witnesses_distinguishable,
)
from deixis.solver.family import (
    canonical_key,
    compare_realizations,
    solve_family,
)
from deixis.solver.geom_solver import solve_boxes
from deixis.verify import reverse

# --------------------------------------------------------------------- domain
# FACE(2) contact needs three spatial axes (2 axes overlap + 1 axis touches), so
# the whole demo lives in 3D. There EDGE(1) contact is realizable too, and both
# contacts are the single RCC-8 relation EC.
DOMAIN: str = "aabb_3d"
NDIM: int = 3
BOUNDS: tuple[int, int] = (0, 20)
MIN_SIZE: Fraction = Fraction(1)

# region ids
A, B, C = "A", "B", "C"

# constraint / witness ids (stable — they anchor identity across the round-trip)
CID_AB, CID_BC, CID_AC = "c:A-B", "c:B-C", "c:A-C"
WID_AB, WID_BC = "w:A-B:face", "w:B-C:edge"

_EC: int = rcc8.mask("EC")
# The *delegated* design envelope for the A-C pair: the designer leaves it open to be
# either fully apart (DC) or partially overlapping (PO). This is the freedom the family
# explores; it is deliberately NOT part of the invariant core.
_AC_ENVELOPE: int = rcc8.mask("DC", "PO")


# ===================================================================== stage 1
def _box(rid: str, lo: tuple, hi: tuple) -> Box:
    """An exact-rational AABB (every bound coerced to :class:`Fraction`)."""
    return Box(region_id=rid, lo=tuple(F(v) for v in lo), hi=tuple(F(v) for v in hi))


def build_original_geometry() -> Realization:
    """Stage 1 — the *original* exact geometry: A-B face contact, B-C edge contact.

    A concrete box group (all coordinates exact ``Fraction``):

    * ``B = [0,4]^3`` — the central box.
    * ``A = [1,3] x [1,3] x [4,6]`` — sits on top of B: interiors overlap on x and y,
      the intervals only touch on z (``A.lo_z == B.hi_z == 4``). Two overlap axes + one
      touch axis => a **FACE** contact (contact dimension 2).
    * ``C = [4,6] x [4,6] x [1,3]`` — meets B along a vertical edge: the intervals only
      touch on x and y (``4 == 4``), interiors overlap on z. One overlap axis + two touch
      axes => an **EDGE** contact (contact dimension 1). (A and C are far apart => DC.)

    Returned as a :class:`Realization` so the same reverse-verification machinery used on
    solver output can observe it.
    """
    boxes = (
        _box(B, (0, 0, 0), (4, 4, 4)),
        _box(A, (1, 1, 4), (3, 3, 6)),
        _box(C, (4, 4, 1), (6, 6, 3)),
    )
    return Realization(
        id="demo7:original",
        boxes=boxes,
        status=Realizability.REALIZED_IN_D,
    )


# ===================================================================== stage 2
def lift(geometry: Realization) -> RelSpec:
    """Stage 2 — observe the geometry into a witnessed :class:`RelSpec` (geometry -> spec).

    Both the A-B and the B-C contacts read back as the **same** RCC-8 relation ``EC``;
    the lift records that (``c:A-B`` and ``c:B-C``, both ``EC``). But the two contacts are
    lifted to *distinct* witnesses carrying the observed contact dimension — A-B at FACE(2),
    B-C at EDGE/LINE(1) — the very distinction RCC-8's single ``EC`` symbol discards.

    Witnesses are marked :class:`~deixis.core.types.Epistemic.OBSERVED_TOLERANT`: a lift
    observation is *evidence*, deliberately NOT auto-promoted to the invariant core (that
    is the designer's separate act in :func:`approve`).

    The A-C pair is lifted as a **delegated** relation over the design envelope ``DC|PO``:
    what was observed (they are apart) is left *open* for grounding, so the family has room
    to vary it. Only the two contacts are candidates for the invariant core.
    """
    # observed base relations (exact read-back, canonical to box order)
    rel_ab = _relation_of(geometry, A, B)
    rel_bc = _relation_of(geometry, B, C)
    assert rel_ab == _EC and rel_bc == _EC, "both contacts must observe as EC"

    dim_ab = int(reverse.extract_contact_dimension(geometry, A, B))
    dim_bc = int(reverse.extract_contact_dimension(geometry, B, C))

    regions = (Region(A, semantic_type="unit"),
               Region(B, semantic_type="unit"),
               Region(C, semantic_type="unit"))
    constraints = (
        RelationConstraint(src=A, dst=B, rcc8_mask=_EC, status=Status.DELEGATED, id=CID_AB),
        RelationConstraint(src=B, dst=C, rcc8_mask=_EC, status=Status.DELEGATED, id=CID_BC),
        # A-C: delegated freedom over the DC|PO envelope (observed DC, left open).
        RelationConstraint(src=A, dst=C, rcc8_mask=_AC_ENVELOPE, status=Status.DELEGATED, id=CID_AC),
    )
    w_ab = make_witness(
        incident_regions=(A, B), intended_contact_dimension=dim_ab,
        id=WID_AB, epistemic=Epistemic.OBSERVED_TOLERANT,
    )
    w_bc = make_witness(
        incident_regions=(B, C), intended_contact_dimension=dim_bc,
        id=WID_BC, epistemic=Epistemic.OBSERVED_TOLERANT,
    )
    return RelSpec(regions=regions, constraints=constraints, witnesses=(w_ab, w_bc))


def _relation_of(real: Realization, x: str, y: str) -> int:
    """Definite RCC-8 base relation of ``x`` w.r.t. ``y``, canonical to box order."""
    rels = reverse.extract_relations(real)
    for (i, j), m in rels.items():
        if (i, j) == (x, y):
            return m
        if (i, j) == (y, x):
            return rcc8.converse(m)
    raise KeyError(f"no relation for {x},{y}")


# ===================================================================== stage 3
def approve_witnesses(spec: RelSpec, *witness_ids: str) -> RelSpec:
    """Promote witnesses into the invariant core: add their ids to ``witness_invariants``.

    ``grounding_op.set_invariant`` covers *relation* constraints (``relation_invariants``);
    a witness lives in the parallel ``Invariant.witness_invariants`` set. This helper is
    the witness-side analogue — a pure function returning a NEW spec (the input is frozen
    and untouched), idempotent in the ids it adds. It also flips the approved witnesses'
    epistemic status from OBSERVED_TOLERANT to ASSERTED: approval is the designer taking
    ownership of the observation as a requirement.

    Fail-closed: approving a witness id that does not exist in ``spec`` raises
    :class:`ValueError` rather than registering a dangling invariant (mirrors the
    discipline of :func:`deixis.grounding.grounding_op.set_invariant`).
    """
    want = frozenset(witness_ids)
    known = frozenset(w.id for w in spec.witnesses)
    missing = want - known
    if missing:
        raise ValueError(
            f"cannot approve unknown witness id(s) {sorted(missing)}; "
            f"present witnesses are {sorted(known)}"
        )
    new_witnesses = tuple(
        replace(w, epistemic=Epistemic.ASSERTED) if w.id in want else w
        for w in spec.witnesses
    )
    new_invariant = replace(
        spec.invariant,
        witness_invariants=spec.invariant.witness_invariants | want,
    )
    return replace(spec, witnesses=new_witnesses, invariant=new_invariant)


def approve(spec: RelSpec) -> RelSpec:
    """Stage 3 — approve the two contacts into the un-touchable invariant core.

    The two ``EC`` relation constraints (A-B, B-C) are promoted with
    :func:`deixis.grounding.grounding_op.set_invariant` (status -> INVARIANT, registered
    in ``relation_invariants``); the two witnesses are promoted with
    :func:`approve_witnesses` (registered in ``witness_invariants``). The A-C constraint is
    *left delegated*: it is the freedom the family will exercise. Returns a NEW spec.
    """
    spec = set_invariant(spec, CID_AB)
    spec = set_invariant(spec, CID_BC)
    spec = approve_witnesses(spec, WID_AB, WID_BC)
    return spec


# ===================================================================== stage 4
def _gd(gid: str, target: str, value: str, rationale: str) -> GroundingDecision:
    return GroundingDecision(
        id=gid, target=target, fixed_value_or_domain=value,
        actor="designer", rationale=rationale,
    )


def grounding_scenarios() -> tuple[list[GroundingDecision], list[GroundingDecision]]:
    """Stage 4 — two grounding sets swapped in against the *same* invariant spec.

    Each scenario fixes some *delegated* freedom without touching the invariant core:

    * **scenario 0 (apart)** — orients A to the "south" datum (a region decision, recorded
      in the ledger) and grounds the delegated A-C relation to ``DC`` (A and C stay apart).
    * **scenario 1 (overlap)** — pins A's basepoint to the "origin" datum and grounds the
      A-C relation to ``PO`` (A and C partially overlap).

    The region-targeting decisions (orientation / basepoint) are the "方位 / 基点" example:
    they are logged in the decision ledger but impose no geometry by themselves; the A-C
    relation decision is what makes the two family members genuinely different. Neither
    scenario can touch the invariant A-B / B-C contacts — grounding is fail-closed against
    the core.
    """
    sc0 = [
        _gd("g:orient:south", A, "south", "orient A to south datum"),
        _gd("g:AC:dc", CID_AC, "DC", "keep A and C apart"),
    ]
    sc1 = [
        _gd("g:base:origin", A, "origin", "pin A basepoint to origin"),
        _gd("g:AC:po", CID_AC, "PO", "let A and C partially overlap"),
    ]
    return sc0, sc1


# ===================================================================== stage 5
def solve(
    spec: RelSpec,
    scenarios: Sequence[Sequence[GroundingDecision]],
) -> list[Realization]:
    """Stage 5 — realize the invariant spec once per scenario: a *family* of solutions.

    Delegates to :func:`deixis.solver.family.solve_family` (which reuses the fail-closed
    two-stage :mod:`deixis.pipeline`). Each returned :class:`Realization` carries its own
    ``scenario_id`` and a provenance ledger of which decision grounded which constraint.
    The two solutions are genuinely different geometries (different A-C relation) that
    still honour the shared invariant contacts.
    """
    return solve_family(
        spec, scenarios, domain=DOMAIN, bounds=BOUNDS, min_size=MIN_SIZE,
    )


# ===================================================================== stage 6
@dataclass(frozen=True)
class SolutionCheck:
    """Reverse-verification summary for one family solution."""
    scenario_id: str
    status: str
    contact_dim_ab: Optional[int]
    contact_dim_bc: Optional[int]
    ac_relation: str
    satisfied: tuple[str, ...]
    violated: tuple[str, ...]
    undecided: tuple[str, ...]
    witness_violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            self.status == Realizability.REALIZED_IN_D.value
            and self.contact_dim_ab == int(ContactDim.FACE)
            and self.contact_dim_bc == int(ContactDim.LINE)
            and not self.violated
            and not self.undecided
            and not self.witness_violations
        )


def verify_solution(spec: RelSpec, real: Realization) -> SolutionCheck:
    """Stage 6 — exact + reverse verification of one solution against the invariant spec.

    Runs :func:`deixis.verify.reverse.verify` (the exact layer: RCC-8 read-back +
    witness contact-dimension check) and independently reads back the A-B and B-C contact
    dimensions from the boxes. A passing check (``.ok``) means the geometry realizes both
    invariant contacts at exactly their required dimension (A-B FACE=2, B-C EDGE=1) with
    no witness violation.
    """
    report = reverse.verify(spec, real)
    try:
        dim_ab = int(reverse.extract_contact_dimension(real, A, B))
    except ValueError:
        dim_ab = None
    try:
        dim_bc = int(reverse.extract_contact_dimension(real, B, C))
    except ValueError:
        dim_bc = None
    ac_names = rcc8.names(_relation_of(real, A, C)) if real.boxes else ()
    return SolutionCheck(
        scenario_id=real.scenario_id,
        status=real.status.value,
        contact_dim_ab=dim_ab,
        contact_dim_bc=dim_bc,
        ac_relation=ac_names[0] if ac_names else "EMPTY",
        satisfied=report.satisfied,
        violated=report.violated,
        undecided=report.undecided,
        witness_violations=tuple(v.code for v in report.witness_violations),
    )


def compare_family(r1: Realization, r2: Realization, spec: Optional[RelSpec] = None) -> dict:
    """Diff two family solutions relative to the pre-registered observation set O.

    Thin wrapper over :func:`deixis.solver.family.compare_realizations`. When ``spec`` is
    given, its *pre-registered* ``spec.observation`` (an :class:`ObservationQuery`) is the
    O the comparison is made against — honouring the "fixed O, not shrunk post-hoc"
    discipline; otherwise the minimal default O is used. The witnessed contact dimensions
    (A-B FACE, B-C EDGE) land in ``preserved`` (the invariant core held across the grounding
    swap) while the delegated A-C relation lands in ``changed``.
    """
    observation = spec.observation if spec is not None else None
    return compare_realizations(r1, r2, observation=observation)


# ===================================================================== stage 7
def rcc8_only_view(spec: RelSpec) -> RelSpec:
    """The RCC-8-only description of the same scene: witnesses stripped away.

    This is exactly what an RCC-8-only tool 'sees': the ``EC`` masks survive, but the
    contact-dimension requirement is gone — nothing distinguishes a face meet from an edge
    meet. The ``witness_invariants`` registry is cleared alongside the witnesses so the
    result is a *consistent* spec (no invariant id dangles onto a removed witness). Returns
    a NEW spec (input untouched)."""
    new_invariant = replace(spec.invariant, witness_invariants=frozenset())
    return replace(spec, witnesses=(), invariant=new_invariant)


def build_edge_geometry() -> Realization:
    """An alternative geometry where A-B is an **EDGE** contact (dimension 1), not a face.

    Produced by solving the scene with the A-B witness demanding EDGE instead of FACE, so
    it is a genuine, exact solution — but it violates the *face* requirement the approved
    spec carries. It uses the *same* relation-constraint structure as the approved spec
    (the two ``EC`` contacts plus the delegated A-C ``DC|PO`` envelope), so it satisfies
    **every RCC-8 mask of the approved spec** while realizing A-B as an *edge* (contact
    dimension 1) rather than a face. That is all stage 7 needs: an RCC-8-satisfying
    geometry whose only defect is at the witnessed contact dimension. (It is an independent
    Z3 solution, not a minimal local edit of the original boxes, so we do not claim the
    rest of the geometry is byte-identical — only that the RCC-8 relations coincide, which
    the stage-7 check and its tests verify.)
    """
    regions = (Region(A), Region(B), Region(C))
    constraints = (
        RelationConstraint(src=A, dst=B, rcc8_mask=_EC, status=Status.DELEGATED, id=CID_AB),
        RelationConstraint(src=B, dst=C, rcc8_mask=_EC, status=Status.DELEGATED, id=CID_BC),
        RelationConstraint(src=A, dst=C, rcc8_mask=_AC_ENVELOPE, status=Status.DELEGATED, id=CID_AC),
    )
    w_ab_edge = make_witness((A, B), ContactDim.LINE, id="w:A-B:edge-alt")
    w_bc = make_witness((B, C), ContactDim.LINE, id=WID_BC)
    edge_spec = RelSpec(regions=regions, constraints=constraints, witnesses=(w_ab_edge, w_bc))
    return solve_boxes(edge_spec, domain=DOMAIN, bounds=BOUNDS, min_size=MIN_SIZE,
                       realization_id="demo7:edge-alt")


@dataclass(frozen=True)
class DistinguishResult:
    """Outcome of feeding an edge geometry to the witnessed vs. RCC-8-only specs."""
    edge_contact_dim_ab: int
    witnessed_rejects: bool          # witnessed spec flags a contact_dim_mismatch
    rcc8_only_accepts: bool          # RCC-8-only spec sees no violation at all

    @property
    def ir_strictly_stronger(self) -> bool:
        """True iff the IR rejects what RCC-8-only accepts — the whole point."""
        return self.witnessed_rejects and self.rcc8_only_accepts


def rcc8_only_cannot_distinguish(spec: RelSpec) -> DistinguishResult:
    """Stage 7 — RCC-8-only accepts an edge geometry a face spec must reject.

    Take the *edge*-contact geometry (A-B is an edge, dimension 1) and check it two ways:

    * against the **witnessed** (approved, FACE-requiring) ``spec`` — reverse verification
      raises a ``contact_dim_mismatch`` witness violation on A-B: the geometry is rejected.
    * against the **RCC-8-only** view (:func:`rcc8_only_view`) — every ``EC`` mask is still
      satisfied (an edge contact *is* ``EC``) and there are no witnesses to check, so it is
      accepted. RCC-8 simply cannot express the difference.

    The gap between these two verdicts is the concrete capability the Witnessed IR adds.
    """
    edge_geom = build_edge_geometry()
    dim_ab = int(reverse.extract_contact_dimension(edge_geom, A, B))

    witnessed = reverse.verify(spec, edge_geom)
    witnessed_rejects = any(
        v.code == "contact_dim_mismatch" and set(v.regions) == {A, B}
        for v in witnessed.witness_violations
    )

    rcc_spec = rcc8_only_view(spec)
    rcc_report = reverse.verify(rcc_spec, edge_geom)
    # Accept only on a *fully decided* pass: every RCC-8 mask satisfied (nothing violated,
    # nothing left undecided by a missing box), no witness check outstanding, and the
    # disputed A-B contact among the satisfied set. This is a strict acceptance, so the
    # "RCC-8-only accepts the edge geometry" claim is not vacuously true.
    rcc8_only_accepts = (
        edge_geom.status is Realizability.REALIZED_IN_D
        and not rcc_report.violated
        and not rcc_report.undecided
        and not rcc_report.witness_violations
        and CID_AB in rcc_report.satisfied
    )
    return DistinguishResult(
        edge_contact_dim_ab=dim_ab,
        witnessed_rejects=witnessed_rejects,
        rcc8_only_accepts=rcc8_only_accepts,
    )


# ===================================================================== driver
def _fmt_box(bx: Box) -> str:
    lo = ",".join(str(v) for v in bx.lo)
    hi = ",".join(str(v) for v in bx.hi)
    return f"{bx.region_id}=[({lo})->({hi})]"


def main() -> None:  # pragma: no cover - human-readable driver
    print("=" * 74)
    print("demo7 — one full round-trip; witnessed IR vs RCC-8-only (headless)")
    print("=" * 74)

    # (1) original geometry ----------------------------------------------------
    geom = build_original_geometry()
    print("\n(1) ORIGINAL geometry (exact Fraction boxes)")
    for b in geom.boxes:
        print(f"    {_fmt_box(b)}")
    print(f"    A-B contact dim = {int(reverse.extract_contact_dimension(geom, A, B))} (FACE)")
    print(f"    B-C contact dim = {int(reverse.extract_contact_dimension(geom, B, C))} (EDGE)")

    # (2) lift -----------------------------------------------------------------
    spec = lift(geom)
    print("\n(2) LIFT — both contacts are ONE RCC-8 symbol EC, kept apart by witnesses")
    print(f"    A-B RCC-8 = {rcc8.names(_relation_of(geom, A, B))}   "
          f"B-C RCC-8 = {rcc8.names(_relation_of(geom, B, C))}")
    w_ab, w_bc = spec.witness(WID_AB), spec.witness(WID_BC)
    print(f"    witness A-B: dim={ContactDim(w_ab.intended_contact_dimension).name} "
          f"epistemic={w_ab.epistemic.value}")
    print(f"    witness B-C: dim={ContactDim(w_bc.intended_contact_dimension).name} "
          f"epistemic={w_bc.epistemic.value}")
    print(f"    collapse to same RCC-8? {collapses_to_same_rcc(w_ab, w_bc)}   "
          f"distinguishable as witnesses? {witnesses_distinguishable(w_ab, w_bc)}")

    # (3) approve --------------------------------------------------------------
    spec = approve(spec)
    print("\n(3) APPROVE — witnesses + EC relations promoted to the invariant core")
    print(f"    relation_invariants = {sorted(spec.invariant.relation_invariants)}")
    print(f"    witness_invariants  = {sorted(spec.invariant.witness_invariants)}")
    print(f"    A-C left DELEGATED  = {spec.constraint(CID_AC).status.value} "
          f"(mask {rcc8.names(spec.constraint(CID_AC).rcc8_mask)})")

    # (4) grounding swap -------------------------------------------------------
    sc0, sc1 = grounding_scenarios()
    print("\n(4) GROUNDING SWAP — two decision sets against the SAME invariant spec")
    for tag, sc in (("apart", sc0), ("overlap", sc1)):
        decs = ", ".join(f"{d.target}={d.fixed_value_or_domain}" for d in sc)
        print(f"    scenario '{tag}': {decs}")

    # (5) solve family ---------------------------------------------------------
    reals = solve(spec, [sc0, sc1])
    print("\n(5) SOLVE FAMILY — genuinely different solutions sharing the invariant core")
    for r in reals:
        print(f"    {r.scenario_id}: status={r.status.value}  "
              f"A-C={rcc8.names(_relation_of(r, A, C))}")
    print(f"    two solutions distinct? {canonical_key(reals[0]) != canonical_key(reals[1])}")

    # (6) verify ---------------------------------------------------------------
    print("\n(6) VERIFY — exact reverse verification of each solution")
    for r in reals:
        chk = verify_solution(spec, r)
        print(f"    {chk.scenario_id}: A-B dim={chk.contact_dim_ab} B-C dim={chk.contact_dim_bc} "
              f"A-C={chk.ac_relation} witness_violations={chk.witness_violations} ok={chk.ok}")
    cmp = compare_family(reals[0], reals[1], spec)
    ab_key = ("contact_dimension", A, B)
    bc_key = ("contact_dimension", B, C)
    ac_key = ("rcc8", A, C)
    print(f"    preserved A-B dim = {cmp['preserved'].get(ab_key)}  "
          f"B-C dim = {cmp['preserved'].get(bc_key)}   (invariant core held)")
    print(f"    changed  A-C rel  = {cmp['changed'].get(ac_key)}          (delegated freedom moved)")

    # (7) RCC-8-only contrast --------------------------------------------------
    dist = rcc8_only_cannot_distinguish(spec)
    print("\n(7) RCC-8-ONLY vs WITNESSED — an EDGE geometry where a FACE was required")
    print(f"    edge geometry A-B contact dim = {dist.edge_contact_dim_ab} (EDGE, not FACE)")
    print(f"    witnessed (FACE) spec rejects it? {dist.witnessed_rejects}  <- contact_dim_mismatch")
    print(f"    RCC-8-only spec accepts it?       {dist.rcc8_only_accepts}  <- just 'EC'")
    print(f"    => IR strictly stronger than RCC-8 here: {dist.ir_strictly_stronger}")
    print("=" * 74)


if __name__ == "__main__":  # pragma: no cover
    main()
