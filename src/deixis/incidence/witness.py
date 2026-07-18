"""Incidence (contact) witnesses — what RCC-8's single ``EC`` symbol collapses.

The central claim of the Witnessed Relational IR (demo1): three *qualitatively
different* ways for two regions to meet —

    face contact   (they share a 2-cell)   intended_contact_dimension = FACE(2)
    edge/line contact (share a 1-cell)     intended_contact_dimension = LINE(1)
    point contact  (share a 0-cell)        intended_contact_dimension = POINT(0)

— all *collapse to the identical RCC-8 relation* ``EC`` (externally connected):
boundaries touch, interiors are disjoint. RCC-8 alone cannot tell them apart.
An :class:`~deixis.core.types.IncidenceWitness` keeps them distinct by recording
the *intended contact dimension* (a geometry-independent requirement) and the
connected contact sub-complex (``cell_ids``) that realises it.

This module gives:

* :func:`make_witness`          — an ergonomic constructor for IncidenceWitness.
* :func:`witness_rcc_mask`      — the RCC-8 mask a contact witness *entails* (EC).
* :func:`check_local_consistency` — local well-formedness / RCC-compatibility /
  ``dim(f) == intended_contact_dimension`` checks over a RelSpec.
* :func:`collapses_to_same_rcc` — the demo1 core: face/line/point witnesses fold
  to one RCC mask, and :func:`witnesses_distinguishable` shows they are still
  distinct as witnesses.

Everything is exact integer / structural work: no floats, no geometry required.
Only :file:`witness.py`'s own types are defined here; the M0 contract in
``deixis.core.types`` / ``deixis.core.rcc8`` is used, never edited.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Optional, Union

from deixis.core import rcc8
from deixis.core.types import (
    Cell,
    ContactDim,
    IncidenceWitness,
    RelSpec,
)
from deixis.core.ids import Provenance
from deixis.core.types import Epistemic

# The valid contact dimensions a boundary witness may record (0/1/2).
_VALID_CONTACT_DIMS: frozenset[int] = frozenset(int(d) for d in ContactDim)

# A boundary/contact witness always entails exactly EC in RCC-8: the whole point
# of the IR is that FACE/LINE/POINT contact are indistinguishable at this level.
_EC: int = rcc8.mask("EC")


# ---------------------------------------------------------------- constructor
def make_witness(
    incident_regions: Iterable[str],
    intended_contact_dimension: Union[int, ContactDim],
    *,
    cell_ids: Iterable[str] = (),
    id: str = "",
    component_index: int = 0,
    boundary_role: str = "shared_boundary",
    epistemic: Epistemic = Epistemic.ASSERTED,
    prov: Optional[Provenance] = None,
) -> IncidenceWitness:
    """Build an :class:`IncidenceWitness` with normalised, immutable fields.

    ``intended_contact_dimension`` accepts a :class:`ContactDim` or a raw int
    (0=POINT, 1=LINE, 2=FACE); it is stored as a plain int (matching the M0
    field type). ``incident_regions`` and ``cell_ids`` are frozen to tuples in a
    stable order. If ``id`` is empty a deterministic id is derived from the
    incident regions, the contact dimension and the component index so equal
    inputs yield equal ids (no randomness).
    """
    regions = tuple(incident_regions)
    cells = tuple(cell_ids)
    # Contact dimension is a discrete ContactDim/int, never a coordinate — reject
    # float outright rather than silently truncating (honours the no-float rule).
    if isinstance(intended_contact_dimension, float):
        raise TypeError(
            "intended_contact_dimension must be a ContactDim or int, not float"
        )
    dim = int(intended_contact_dimension)
    if not id:
        id = "w:{regions}:d{dim}:c{comp}".format(
            regions="-".join(regions), dim=dim, comp=component_index
        )
    return IncidenceWitness(
        id=id,
        cell_ids=cells,
        incident_regions=regions,
        intended_contact_dimension=dim,
        component_index=component_index,
        boundary_role=boundary_role,
        epistemic=epistemic,
        prov=prov,
    )


# ---------------------------------------------------------------- RCC entailment
def witness_rcc_mask(w: IncidenceWitness) -> int:
    """RCC-8 mask entailed by a contact witness.

    A shared-boundary witness means: the two regions' *boundaries* meet while
    their *interiors* stay disjoint — that is exactly RCC-8 ``EC``. This holds
    regardless of the contact dimension (face, line or point), which is the
    whole reason the IR must carry the witness on top of RCC-8. The single
    returned bit is therefore ``bit('EC')`` for every contact dimension.
    """
    return _EC


# ---------------------------------------------------------------- violations
@dataclass(frozen=True)
class Violation:
    """A single local-consistency violation attached to a witness (or a pair)."""
    witness_id: str
    code: str
    detail: str
    regions: tuple[str, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        who = f" {self.regions}" if self.regions else ""
        return f"[{self.code}] witness {self.witness_id!r}{who}: {self.detail}"


def _cell_dim_lookup(
    cells: Union[None, Mapping[str, int], Iterable[Cell]],
) -> Optional[dict[str, int]]:
    """Normalise the optional cell argument to an id->dim dict (or None)."""
    if cells is None:
        return None
    if isinstance(cells, Mapping):
        return {str(k): int(v) for k, v in cells.items()}
    out: dict[str, int] = {}
    for c in cells:
        # accept M0 Cell objects (id, dim)
        out[c.id] = int(c.dim)
    return out


def check_witness(
    w: IncidenceWitness,
    *,
    constraints: Iterable = (),
    region_ids: Optional[frozenset[str]] = None,
    cell_dims: Optional[Mapping[str, int]] = None,
) -> list[Violation]:
    """Local checks for one witness. Returns a (possibly empty) list of Violations.

    Checks performed:

    * **too_few_regions** — a contact witness must join at least two regions.
    * **bad_contact_dim** — ``intended_contact_dimension`` must be a valid
      :class:`ContactDim` value (0/1/2).
    * **unknown_region** — every incident region must exist (when ``region_ids``
      is supplied).
    * **rcc_incompatible** — for each incident region pair with a stated
      constraint, the witness entailment (``EC``) must be *possible* under that
      constraint's mask, i.e. ``mask & bit('EC') != 0``. EC is self-converse so
      the check is direction-independent.
    * **dim_mismatch** — when cell dimensions are known, the *top* dimension of
      the connected contact sub-complex (``max`` over ``cell_ids``) must equal
      ``intended_contact_dimension``: a FACE contact's sub-complex is topped by a
      2-cell, a LINE contact by a 1-cell, a POINT contact by a 0-cell.
    * **unknown_cell** — a referenced cell has no known dimension (only when a
      ``cell_dims`` map is supplied and ``cell_ids`` is non-empty).
    """
    out: list[Violation] = []
    regions = tuple(w.incident_regions)

    if len(regions) < 2:
        out.append(Violation(w.id, "too_few_regions",
                             f"needs >=2 incident regions, got {len(regions)}",
                             regions))
    elif len(set(regions)) != len(regions):
        # EC (and any contact) is irreflexive: a region cannot be externally
        # connected to itself, so incident regions must be distinct.
        out.append(Violation(w.id, "duplicate_region",
                             f"incident regions must be distinct, got {regions}",
                             regions))

    if w.intended_contact_dimension not in _VALID_CONTACT_DIMS:
        out.append(Violation(w.id, "bad_contact_dim",
                             f"intended_contact_dimension={w.intended_contact_dimension} "
                             f"is not a valid ContactDim {sorted(_VALID_CONTACT_DIMS)}"))

    if region_ids is not None:
        for r in regions:
            if r not in region_ids:
                out.append(Violation(w.id, "unknown_region",
                                     f"incident region {r!r} not in spec", (r,)))

    # RCC-8 compatibility with any stated pairwise constraint.
    entailed = witness_rcc_mask(w)
    cmap: dict[frozenset[str], list[int]] = {}
    for c in constraints:
        cmap.setdefault(frozenset((c.src, c.dst)), []).append(c.rcc8_mask)
    for a, b in combinations(regions, 2):
        key = frozenset((a, b))
        for m in cmap.get(key, ()):  # possibly several constraints on the pair
            if (m & entailed) == 0:
                out.append(Violation(
                    w.id, "rcc_incompatible",
                    f"witness entails EC ({rcc8.names(entailed)}) but constraint "
                    f"mask {rcc8.names(m)} forbids it", (a, b)))

    # dim(sub-complex) == intended_contact_dimension, when dims are known.
    if cell_dims is not None and w.cell_ids:
        dims: list[int] = []
        for cid in w.cell_ids:
            if cid not in cell_dims:
                out.append(Violation(w.id, "unknown_cell",
                                     f"cell {cid!r} has no known dimension", ()))
            else:
                dims.append(int(cell_dims[cid]))
        if dims:
            top = max(dims)
            if top != w.intended_contact_dimension:
                out.append(Violation(
                    w.id, "dim_mismatch",
                    f"top cell dim {top} != intended_contact_dimension "
                    f"{w.intended_contact_dimension}"))
    return out


def check_local_consistency(
    spec: RelSpec,
    *,
    cells: Union[None, Mapping[str, int], Iterable[Cell]] = None,
) -> list[Violation]:
    """Run :func:`check_witness` for every witness in ``spec``.

    ``spec`` is an M0 :class:`RelSpec`. Because ``RelSpec`` carries no cell
    complex of its own, the optional ``cells`` argument supplies cell dimensions
    (a ``{cell_id: dim}`` map or an iterable of :class:`Cell`) so the
    ``dim(f) == intended_contact_dimension`` check can run; omit it to skip that
    check. Returns all violations across all witnesses (empty == locally
    consistent)."""
    region_ids = frozenset(r.id for r in spec.regions)
    cell_dims = _cell_dim_lookup(cells)
    out: list[Violation] = []
    for w in spec.witnesses:
        out.extend(check_witness(
            w,
            constraints=spec.constraints,
            region_ids=region_ids,
            cell_dims=cell_dims,
        ))
    return out


# ---------------------------------------------------------------- demo1 core
def collapses_to_same_rcc(*witnesses: IncidenceWitness) -> bool:
    """True iff every given witness entails the *same* RCC-8 mask.

    For the demo1 triple ``(w_face, w_edge, w_point)`` this is ``True``: all
    three fold onto ``EC``. This is the *lossy* face of the IR — the direction in
    which distinct contacts become indistinguishable in RCC-8. Pair it with
    :func:`witnesses_distinguishable` to show that the witnesses themselves are
    *not* lost.
    """
    if not witnesses:
        return True
    masks = {witness_rcc_mask(w) for w in witnesses}
    return len(masks) == 1


def witnesses_distinguishable(*witnesses: IncidenceWitness) -> bool:
    """True iff the witnesses are pairwise distinct *as witnesses*.

    Two contact witnesses are distinguishable when they differ in their intended
    contact dimension or in the connected contact sub-complex they occupy
    (``cell_ids``) — information RCC-8 discards but the IR keeps. Requires at
    least two witnesses.
    """
    if len(witnesses) < 2:
        return False
    signatures = {
        (w.intended_contact_dimension, tuple(w.cell_ids)) for w in witnesses
    }
    return len(signatures) == len(witnesses)
