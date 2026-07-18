"""demo1 — the *superiority-of-spatial-description* core (卒論の中心主張, minimal).

The strongest comparison case. Two regions ``A`` and ``B`` are, in RCC-8, in
**exactly one** relation: ``EC`` (externally connected — boundaries touch,
interiors disjoint). Yet there are *qualitatively different* ways for them to
meet, and a designer routinely means one and not the others:

    FACE contact (share a 2-cell)   intended_contact_dimension = FACE(2)
    LINE/edge    (share a 1-cell)   intended_contact_dimension = LINE(1)
    POINT/corner (share a 0-cell)   intended_contact_dimension = POINT(0)

RCC-8 collapses all three onto the single symbol ``EC``. This module builds the
three specifications that differ *only* in the witnessed contact dimension and
shows three things:

  (1) The three specs carry the **identical** RCC-8 mask (``EC``) — RCC-8 alone
      cannot tell them apart — but their :class:`IncidenceWitness`es *do*
      distinguish them (information available *before* any geometry exists).

  (2) Solving each spec (``run`` -> exact-rational AABBs, the pipeline role) and
      verifying it (``verify``, the reverse-verification role) shows the realized
      geometry actually meets that spec's contact-dimension requirement.

  (3) In an **RCC-8-only** spec (witness stripped) the FACE requirement is lost:
      an *edge*-contact geometry (contact dim 1) is happily accepted, because at
      the RCC-8 level it is just ``EC``. The witnessed spec rejects it. This is
      the concrete failure the IR repairs.

Everything is exact: coordinates are :class:`fractions.Fraction`, comparisons are
exact rational, and the RCC-8 classification of the realized boxes is structural.

No ``pipeline`` / ``reverse`` modules exist in this tree yet, so the pipeline-run
and reverse-verify *roles* are provided here directly on top of the fixed M0
contract (``deixis.core``), ``deixis.incidence.witness`` and
``deixis.solver.geom_solver`` — this demo edits none of them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Optional

from deixis.core import rcc8
from deixis.core.types import (
    Box,
    Cell,
    ContactDim,
    Realizability,
    Realization,
    Region,
    RelationConstraint,
    RelSpec,
    Status,
)
from deixis.incidence.witness import (
    collapses_to_same_rcc,
    make_witness,
    witnesses_distinguishable,
)
from deixis.solver.geom_solver import solve_boxes

# --------------------------------------------------------------------- domain
# FACE(2) contact requires three spatial axes (2 axes overlap + 1 axis touches),
# so the whole demo lives in 3D. There LINE(1) and POINT(0) are realizable too,
# and all three are the single RCC-8 relation EC.
DOMAIN: str = "aabb_3d"
_NDIM: int = 3
BOUNDS: tuple[int, int] = (0, 10)

_EC_MASK: int = rcc8.mask("EC")

# region ids used throughout
_A, _B = "A", "B"


# --------------------------------------------------------------------- builder
def _spec_for(dim: ContactDim, cell_dim_id: str, cell: Cell) -> RelSpec:
    """One RelSpec: A EC B (common), witnessed at contact dimension ``dim``."""
    regions = (
        Region(id=_A, semantic_type="unit"),
        Region(id=_B, semantic_type="unit"),
    )
    # The *identical* RCC-8 constraint in all three specs.
    constraint = RelationConstraint(
        src=_A, dst=_B, rcc8_mask=_EC_MASK, status=Status.DELEGATED,
        id=f"c:A-EC-B:{int(dim)}",
    )
    # The witness is what makes the three specs distinguishable. Its connected
    # contact sub-complex is a single cell whose dimension == the intended
    # contact dimension (a FACE contact is topped by a 2-cell, LINE by a 1-cell,
    # POINT by a 0-cell).
    w = make_witness(
        incident_regions=(_A, _B),
        intended_contact_dimension=dim,
        cell_ids=(cell_dim_id,),
        id=f"w:A-B:d{int(dim)}",
    )
    return RelSpec(regions=regions, constraints=(constraint,), witnesses=(w,))


def build_specs() -> tuple[RelSpec, RelSpec, RelSpec]:
    """Return ``(spec_face, spec_edge, spec_point)``.

    All three assert the same relation — ``A EC B`` — and differ only in the
    witnessed ``intended_contact_dimension`` (FACE / LINE / POINT). This is the
    minimal core of the thesis: identical qualitative relation, distinct spatial
    meaning.
    """
    spec_face = _spec_for(ContactDim.FACE, "f:A-B", Cell(id="f:A-B", dim=2))
    spec_edge = _spec_for(ContactDim.LINE, "e:A-B", Cell(id="e:A-B", dim=1))
    spec_point = _spec_for(ContactDim.POINT, "p:A-B", Cell(id="p:A-B", dim=0))
    return spec_face, spec_edge, spec_point


def strip_witnesses(spec: RelSpec) -> RelSpec:
    """The RCC-8-only view of a spec: same regions/constraints, witnesses removed.

    This is exactly what an RCC-8-only tool 'sees'. The contact-dimension
    requirement is *gone* — nothing distinguishes a face meet from an edge meet.
    """
    return replace(spec, witnesses=())


# --------------------------------------------------------- realized-geometry ops
def _interval_flags(a: Box, b: Box, d: int) -> tuple[bool, bool, bool]:
    """(interiors_overlap, touch, separated) for axis ``d`` — exact rational."""
    a0, a1 = a.lo[d], a.hi[d]
    b0, b1 = b.lo[d], b.hi[d]
    iover = (a0 < b1) and (b0 < a1)
    touch = (a1 == b0) or (b1 == a0)
    sep = (a1 < b0) or (b1 < a0)
    return iover, touch, sep


def realized_rcc8_name(a: Box, b: Box, ndim: int = _NDIM) -> str:
    """The exact RCC-8 base relation between two AABBs (Rectangle-Algebra view).

    Uses the same JEPD partition the solver encodes: per axis the two closed
    intervals either overlap (interiors), merely touch, or are separated. All
    comparisons are exact ``Fraction`` — no tolerance.
    """
    iover = []
    touch = []
    sep = []
    for d in range(ndim):
        o, t, s = _interval_flags(a, b, d)
        iover.append(o)
        touch.append(t)
        sep.append(s)

    if any(sep):
        return "DC"
    # connected on every axis (closed intervals intersect everywhere)
    if not all(iover):
        # connected but at least one axis only touches -> boundaries meet,
        # interiors disjoint -> EC.
        return "EC"

    # interiors overlap on every axis: EQ / TPP / NTPP / TPPi / NTPPi / PO
    eq = all(a.lo[d] == b.lo[d] and a.hi[d] == b.hi[d] for d in range(ndim))
    if eq:
        return "EQ"
    a_in_b = all(a.lo[d] >= b.lo[d] and a.hi[d] <= b.hi[d] for d in range(ndim))
    b_in_a = all(b.lo[d] >= a.lo[d] and b.hi[d] <= a.hi[d] for d in range(ndim))
    if a_in_b:
        strict = all(a.lo[d] > b.lo[d] and a.hi[d] < b.hi[d] for d in range(ndim))
        return "NTPP" if strict else "TPP"
    if b_in_a:
        strict = all(b.lo[d] > a.lo[d] and b.hi[d] < a.hi[d] for d in range(ndim))
        return "NTPPi" if strict else "TPPi"
    return "PO"


def realized_contact_dim(a: Box, b: Box, ndim: int = _NDIM) -> Optional[int]:
    """Contact dimension of an EC pair of AABBs: number of interior-overlap axes.

    Returns ``None`` when the boxes are not EC (so the notion of *contact*
    dimension does not apply). For an EC pair in 3D this is 2=FACE, 1=LINE,
    0=POINT.
    """
    if realized_rcc8_name(a, b, ndim) != "EC":
        return None
    return sum(1 for d in range(ndim) if _interval_flags(a, b, d)[0])


def _boxes_by_region(real: Realization) -> dict[str, Box]:
    return {bx.region_id: bx for bx in real.boxes}


# ------------------------------------------------------------------ run (solve)
def run(spec: RelSpec, domain: str = DOMAIN, bounds: tuple = BOUNDS) -> Realization:
    """Solve ``spec`` to exact-rational AABBs (the pipeline-run role).

    Delegates to :func:`deixis.solver.geom_solver.solve_boxes`, which honours each
    witness's ``intended_contact_dimension`` when it realizes an ``EC`` pair.
    """
    return solve_boxes(spec, domain=domain, bounds=bounds)


# -------------------------------------------------------------- verify (reverse)
@dataclass(frozen=True)
class WitnessCheck:
    witness_id: str
    regions: tuple[str, ...]
    intended: int
    realized_rcc8: Optional[str]
    realized_dim: Optional[int]
    passed: bool


@dataclass(frozen=True)
class VerifyResult:
    """Result of reverse-verifying a realization against a witnessed spec."""
    ok: bool
    checks: tuple[WitnessCheck, ...]

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def verify(spec: RelSpec, real: Realization, ndim: int = _NDIM) -> VerifyResult:
    """Reverse-verify: does ``real`` meet every contact-dimension requirement?

    For each witness, the realized boxes of its incident regions must (a) be in
    RCC-8 ``EC`` and (b) have realized contact dimension equal to the witness's
    ``intended_contact_dimension``. This is the check RCC-8 *cannot* express — it
    is exactly the added power of the witnessed IR.
    """
    boxes = _boxes_by_region(real)
    checks: list[WitnessCheck] = []
    all_ok = bool(real.boxes)  # an empty (unsat/unknown) realization verifies nothing
    for w in spec.witnesses:
        regs = tuple(w.incident_regions)
        # demo1 witnesses join exactly two regions; check that pair.
        a = boxes.get(regs[0]) if len(regs) >= 1 else None
        b = boxes.get(regs[1]) if len(regs) >= 2 else None
        if a is None or b is None:
            checks.append(WitnessCheck(w.id, regs, w.intended_contact_dimension,
                                       None, None, False))
            all_ok = False
            continue
        rel = realized_rcc8_name(a, b, ndim)
        dim = realized_contact_dim(a, b, ndim)
        passed = (rel == "EC") and (dim == w.intended_contact_dimension)
        checks.append(WitnessCheck(w.id, regs, w.intended_contact_dimension,
                                   rel, dim, passed))
        all_ok = all_ok and passed
    return VerifyResult(ok=all_ok, checks=tuple(checks))


def rcc_only_verify(spec: RelSpec, real: Realization, ndim: int = _NDIM) -> bool:
    """RCC-8-only verification: does ``real`` satisfy just the RCC-8 masks?

    For each constraint, the realized relation of ``(src, dst)`` must be one of
    the base relations allowed by ``rcc8_mask``. Contact dimension is *not*
    consulted — this is precisely what an RCC-8-only tool can check, and why it
    cannot tell a face meet from an edge meet.
    """
    if not real.boxes:
        return False
    boxes = _boxes_by_region(real)
    for c in spec.constraints:
        a = boxes.get(c.src)
        b = boxes.get(c.dst)
        if a is None or b is None:
            return False
        rel_bit = rcc8.bit(realized_rcc8_name(a, b, ndim))
        if (c.rcc8_mask & rel_bit) == 0:
            return False
    return True


# ----------------------------------------------------------------------- report
def _fmt_box(bx: Box) -> str:
    lo = ",".join(str(v) for v in bx.lo)
    hi = ",".join(str(v) for v in bx.hi)
    return f"{bx.region_id}=[({lo})->({hi})]"


def main() -> None:  # pragma: no cover - human-readable driver
    spec_face, spec_edge, spec_point = build_specs()
    named = (("FACE", spec_face), ("LINE/EDGE", spec_edge), ("POINT", spec_point))

    print("=" * 70)
    print("demo1 — superiority of witnessed spatial description over RCC-8")
    print("=" * 70)

    # (1) identical RCC-8 mask, distinguishable witnesses ----------------------
    print("\n(1) RCC-8 sees ONE relation; the witness keeps three apart")
    for label, spec in named:
        masks = [rcc8.names(c.rcc8_mask) for c in spec.constraints]
        w = spec.witnesses[0]
        print(f"    {label:10s}: RCC-8={masks}  "
              f"intended_contact_dimension={ContactDim(w.intended_contact_dimension).name}")
    ws = [s.witnesses[0] for _, s in named]
    print(f"    -> all fold to the same RCC-8 mask? {collapses_to_same_rcc(*ws)}")
    print(f"    -> witnesses pairwise distinguishable? {witnesses_distinguishable(*ws)}")

    # (2) solve + reverse-verify each spec -------------------------------------
    print("\n(2) solve (exact AABBs) then reverse-verify the contact dimension")
    reals: dict[str, Realization] = {}
    for label, spec in named:
        r = run(spec)
        reals[label] = r
        vr = verify(spec, r)
        boxes = "  ".join(_fmt_box(b) for b in r.boxes)
        print(f"    {label:10s}: status={r.status.value}  verify.ok={vr.ok}")
        print(f"                {boxes}")
        for ch in vr.checks:
            print(f"                witness {ch.witness_id}: RCC-8={ch.realized_rcc8} "
                  f"realized_dim={ch.realized_dim} intended={ch.intended} "
                  f"passed={ch.passed}")

    # (3) RCC-8-only cannot distinguish: edge geometry passes a FACE spec ------
    print("\n(3) RCC-8-only loses the requirement: an EDGE geometry is accepted")
    edge_real = reals["LINE/EDGE"]
    spec_face_rcc = strip_witnesses(spec_face)
    rcc_ok = rcc_only_verify(spec_face_rcc, edge_real)
    witnessed_ok = verify(spec_face, edge_real).ok
    print(f"    edge-contact geometry vs FACE spec:")
    print(f"      RCC-8-only verify (witness stripped): {rcc_ok}   <- accepts it (just 'EC')")
    print(f"      witnessed verify (FACE required):     {witnessed_ok}   <- rejects it")
    print("    => only the witnessed IR distinguishes face from edge contact.")
    print("=" * 70)


if __name__ == "__main__":  # pragma: no cover
    main()
