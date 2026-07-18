"""M6.5 — the turn-based bidirectional *maintainer* over a relational session.

This module maintains a live design session in which the SAME artifact is viewed
through two ports:

* the **relation port** — the qualitative RCC-8 ``RelSpec`` (what the designer
  writes / edits);
* the **geometry port** — a concrete exact-rational ``Realization`` (what the
  solver produced).

A *maintainer* keeps the two roughly in step across turns by two asymmetric,
*partial* transformations — deliberately NOT a bidirectionalization/lens with
well-behavedness laws:

    put_edit_relation :  edit a relation  -> re-synthesize geometry  (pipeline.run)
    get_observe_geometry: read the geometry -> update the observed relations
                          (reverse.extract_relations)

Why "maintainer" and not "lens".  A lens promises algebraic round-trip laws
(GetPut / PutGet / PutPut) and often an *adjoint*.  We promise none of them, and
this module makes several honest concessions explicit in code and tests:

* **put is an ill-posed inverse problem.**  A single edited relation does not
  determine a geometry; the solver returns *one* witness among many.  ``put`` is
  therefore only *minimal-change oriented* (it edits one relation delta and lets
  the solver re-realize) — it is NOT a right/left inverse of ``get``.
* **get is forgetful and irreversible.**  Reading the geometry collapses each
  observed pair onto the single realized RCC-8 base relation, discarding any
  disjunction the delegated constraint used to carry.  The original relational
  *topology* (disjunctions, alternative solutions, the pre-image) is NOT
  recovered by any later ``put``.
* **confluence (PutPut) is NOT guaranteed.**  Two edits applied in different
  orders may yield different sessions; the maintainer performs no merge and makes
  no commutation claim (see ``test_maintainer`` :: confluence-not-claimed).
* **no adjoint / no delta-lens structure is provided.**  There is deliberately no
  ``adjoint``, ``merge`` or ``round_trip`` operation.

What *is* guaranteed (and tested):

* **append-only history** — each *successful* turn appends exactly one
  :class:`HistoryEntry`, preserving the prior history as an untouched prefix (earlier
  turns are never rewritten or dropped).  A refused (fail-closed) turn raises and leaves
  the input session unchanged, recording nothing.  (The maintainer does not police a
  hand-forged :class:`SessionState` built directly via its public constructor — the
  guarantee is about what the turn *operations* do.)
* **fail-closed invariant protection** — ``put`` refuses any edit that would touch
  an INVARIANT relation, and refuses a whole-spec replacement that would alter the
  invariant core or the grounding ledger.
* **complement preservation** — the session's freedom splits into two complements
  that the two directions hold *constant*:

      C_B  (invariant relations + grounding ledger)   held constant by ``put``
      C_A  (the geometry's metric degrees of freedom)  held constant by ``get``

  ``put`` re-synthesizes geometry (moves C_A) while preserving C_B; ``get`` updates
  the observed relations (moves the relational view) while preserving C_A (it does
  NOT re-solve — the realization it observed stays byte-for-byte).  The decided,
  un-edited relations (GROUNDED / INVARIANT) survive a ``get`` untouched — they are
  the relational half of the preserved complement.

Everything downstream stays exact: geometry is produced by :mod:`deixis.pipeline`
(exact-rational AABBs) and relations are read back by :mod:`deixis.verify.reverse`
(exact integer bit arithmetic).  Every RCC-8 / contact decision the maintainer relies on
is computed on exact :class:`fractions.Fraction` coordinates — no float enters a
*decision*.  (The ``bounds`` argument is forwarded verbatim to the solver, which coerces
it via ``Fraction``; passing a float bound is discouraged but not policed here.)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Union

from ..core import rcc8
from ..core.ids import Provenance
from ..core.types import (
    Realizability,
    Realization,
    RelationConstraint,
    RelSpec,
    Status,
)
from ..verify import reverse
from .. import pipeline

__all__ = [
    "MaintainerError",
    "RelationEdit",
    "HistoryEntry",
    "SessionState",
    "new_session",
    "put_edit_relation",
    "get_observe_geometry",
    "invariant_ids",
    "grounding_ledger_ids",
    "decided_relation_masks",
]


class MaintainerError(Exception):
    """Raised on an illegal maintainer turn (fail-closed invariant / ledger guard)."""


# ---------------------------------------------------------------- edit descriptor
@dataclass(frozen=True)
class RelationEdit:
    """A minimal relation-port delta: set the RCC-8 mask on one pair.

    ``constraint_id`` names the exact :class:`RelationConstraint` to edit; when it
    is empty the target is matched by the unordered pair ``{src, dst}``.  If no such
    constraint exists the edit *adds* a new DELEGATED constraint for the pair (a
    minimal, additive change — never a silent rewrite of something else).
    """
    src: str
    dst: str
    new_mask: int
    constraint_id: str = ""


# ---------------------------------------------------------------- history record
@dataclass(frozen=True)
class HistoryEntry:
    """One append-only turn record.  ``port`` is the port that was driven."""
    port: str                       # "relation" (put) | "geometry" (get)
    op: str                         # "put_edit_relation" | "get_observe_geometry"
    version: int                    # resulting SessionState.version
    detail: str = ""
    status: Optional[Realizability] = None   # realizability rung, when relevant


# ---------------------------------------------------------------- session state
@dataclass(frozen=True)
class SessionState:
    """Immutable snapshot of a maintained session (relation port + geometry port).

    Every field is immutable; a turn returns a NEW ``SessionState`` (never mutates).
    ``editing_port`` records which port the *last* turn drove ("relation" for a put,
    "geometry" for a get, "" for the initial state).
    """
    spec: RelSpec
    realization: Optional[Realization] = None
    history: tuple[HistoryEntry, ...] = ()
    version: int = 0
    editing_port: str = ""


def new_session(spec: RelSpec, realization: Optional[Realization] = None) -> SessionState:
    """Open a session at version 0 with an empty (append-only) history."""
    return SessionState(
        spec=spec,
        realization=realization,
        history=(),
        version=0,
        editing_port="",
    )


# ---------------------------------------------------------------- complement views
def invariant_ids(spec: RelSpec) -> frozenset[str]:
    """Ids in the invariant core C_B (union of the ledger set and INVARIANT status)."""
    by_status = {c.id for c in spec.constraints if c.status is Status.INVARIANT}
    return frozenset(spec.invariant.relation_invariants) | frozenset(by_status)


def grounding_ledger_ids(spec: RelSpec) -> tuple[str, ...]:
    """The grounding decision ids, in ledger order (the other half of C_B)."""
    return tuple(d.id for d in spec.groundings)


def decided_relation_masks(spec: RelSpec) -> dict[str, tuple[int, str]]:
    """``{constraint_id: (mask, status)}`` for every DECIDED (GROUNDED/INVARIANT) pair.

    These are exactly the relations ``get`` must leave untouched (the relational half
    of the preserved complement)."""
    out: dict[str, tuple[int, str]] = {}
    for c in spec.constraints:
        if c.status in (Status.GROUNDED, Status.INVARIANT):
            out[c.id] = (c.rcc8_mask, c.status.value)
    return out


# ---------------------------------------------------------------- helpers
def _canon(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


def _find_constraint(spec: RelSpec, edit: RelationEdit) -> Optional[int]:
    """Index of the constraint the edit targets, or None if it must be added."""
    if edit.constraint_id:
        for i, c in enumerate(spec.constraints):
            if c.id == edit.constraint_id:
                return i
        return None
    pair = _canon(edit.src, edit.dst)
    for i, c in enumerate(spec.constraints):
        if _canon(c.src, c.dst) == pair:
            return i
    return None


def _apply_edit(spec: RelSpec, edit: RelationEdit) -> RelSpec:
    """Apply a single relation delta, fail-closed against the invariant core.

    Editing an INVARIANT relation raises ``MaintainerError`` (the un-touchable core is
    protected).  Editing a DELEGATED/GROUNDED relation replaces its mask in place;
    a missing pair is *added* as a fresh DELEGATED constraint (minimal additive change).
    """
    if edit.new_mask <= 0 or edit.new_mask > rcc8.UNIVERSAL:
        raise MaintainerError(
            f"relation edit mask {edit.new_mask} out of range 1..255 (0 == EMPTY is refused)"
        )
    idx = _find_constraint(spec, edit)
    prov = Provenance(
        origin="put-edit",
        actor="designer",
        activity="put_edit_relation",
        note=f"set {edit.src}->{edit.dst} := {'|'.join(rcc8.names(edit.new_mask))}",
    )
    if idx is None:
        # Adding a fresh constraint: refuse to create a *shadow* second constraint on a
        # pair that another constraint already governs (that would be a silent, ambiguous
        # rewrite rather than a minimal additive change).
        pair = _canon(edit.src, edit.dst)
        if any(_canon(c.src, c.dst) == pair for c in spec.constraints):
            raise MaintainerError(
                f"pair {{{edit.src}, {edit.dst}}} is already constrained; edit it by its "
                f"constraint_id instead of adding a shadow constraint"
            )
        new_id = edit.constraint_id or f"edit:{edit.src}->{edit.dst}"
        if any(c.id == new_id for c in spec.constraints):
            raise MaintainerError(f"cannot add constraint: id {new_id!r} already exists")
        added = RelationConstraint(
            src=edit.src, dst=edit.dst, rcc8_mask=edit.new_mask,
            status=Status.DELEGATED, id=new_id, prov=prov,
        )
        return spec.with_constraints(spec.constraints + (added,))

    c = spec.constraints[idx]
    if c.status is Status.INVARIANT:
        raise MaintainerError(
            f"relation {c.id!r} is INVARIANT and cannot be edited by put (fail-closed)"
        )
    cs = list(spec.constraints)
    cs[idx] = replace(c, rcc8_mask=edit.new_mask, prov=prov)
    return spec.with_constraints(tuple(cs))


def _guard_complement_preserved(before: RelSpec, after: RelSpec) -> None:
    """Fail-closed: a whole-spec ``put`` must hold C_B (invariant core + ledger) constant.

    Every INVARIANT relation of ``before`` must survive in ``after`` with the same mask
    and INVARIANT status, and the grounding ledger must be identical.  Otherwise the put
    would silently move the complement ``put`` is contracted to preserve."""
    if grounding_ledger_ids(before) != grounding_ledger_ids(after):
        raise MaintainerError(
            "put must preserve the grounding ledger (C_B) constant; it changed"
        )
    after_by_id = {c.id: c for c in after.constraints}
    for cid in invariant_ids(before):
        b = before.constraint(cid)
        a = after_by_id.get(cid)
        if a is None:
            raise MaintainerError(f"put dropped invariant relation {cid!r} (fail-closed)")
        if a.status is not Status.INVARIANT or (b is not None and a.rcc8_mask != b.rcc8_mask):
            raise MaintainerError(
                f"put altered invariant relation {cid!r} (mask/status) (fail-closed)"
            )


# ---------------------------------------------------------------- PUT  (relation -> geometry)
def put_edit_relation(
    state: SessionState,
    new_spec_or_edit: Union[RelSpec, RelationEdit],
    domain: str = "aabb_2d",
    bounds: tuple = (0, 100),
) -> SessionState:
    """Edit a relation, then re-synthesize geometry (the *put* direction).

    ``new_spec_or_edit`` is either a :class:`RelationEdit` (the minimal, preferred
    delta) or a whole replacement :class:`RelSpec`.  In both cases the edit is
    **fail-closed** against the invariant core: touching an INVARIANT relation — or, for
    a whole-spec replacement, altering the invariant core or the grounding ledger — raises
    :class:`MaintainerError` and the session is left unchanged.

    The candidate spec is realized with :func:`deixis.pipeline.run` (Stage A symbolic
    consistency, then exact-rational geometry with reverse-verification).  The returned
    :class:`Realization` — at whatever ladder rung it honestly reached — becomes the new
    geometry port; C_B (invariant relations + grounding ledger) is preserved, while the
    geometry's metric freedom C_A is *re-chosen* by the solver (put moves C_A).

    Honest limits: ``put`` is an ill-posed inverse (the solver returns one witness among
    many, so this is only minimal-change *oriented*, not an inverse of ``get``).  The
    minimal-change orientation applies to the *relational* input — a single constraint
    delta — only: the geometry is re-synthesized from scratch each turn with **no**
    proximity objective to the previous realization, so consecutive puts may jump between
    unrelated geometries.  ``put`` offers no confluence guarantee (see the module
    docstring).  The turn appends one :class:`HistoryEntry` and increments ``version``.
    """
    if isinstance(new_spec_or_edit, RelationEdit):
        candidate = _apply_edit(state.spec, new_spec_or_edit)
        detail = (
            f"edit {new_spec_or_edit.src}->{new_spec_or_edit.dst} := "
            f"{'|'.join(rcc8.names(new_spec_or_edit.new_mask))}"
        )
    elif isinstance(new_spec_or_edit, RelSpec):
        candidate = new_spec_or_edit
        _guard_complement_preserved(state.spec, candidate)
        detail = "whole-spec replacement (C_B preserved)"
    else:  # fail-closed on an unrecognized payload
        raise MaintainerError(
            f"put expects a RelationEdit or a RelSpec, got {type(new_spec_or_edit).__name__}"
        )

    real = pipeline.run(candidate, groundings=(), domain=domain, bounds=bounds)

    version = state.version + 1
    entry = HistoryEntry(
        port="relation",
        op="put_edit_relation",
        version=version,
        detail=detail,
        status=real.status,
    )
    return SessionState(
        spec=candidate,
        realization=real,
        history=state.history + (entry,),
        version=version,
        editing_port="relation",
    )


# ---------------------------------------------------------------- GET  (geometry -> relation)
def _observed_mask(
    rels: dict[tuple[str, str], int], src: str, dst: str
) -> Optional[int]:
    """Observed single-base RCC-8 mask of ``src`` w.r.t. ``dst`` (converse if reversed)."""
    if (src, dst) in rels:
        return rels[(src, dst)]
    if (dst, src) in rels:
        return rcc8.converse(rels[(dst, src)])
    return None


def get_observe_geometry(state: SessionState) -> SessionState:
    """Read the geometry back into the observed relations (the *get* direction).

    Requires a REALIZED_IN_D geometry port (a reverse-verified :class:`Realization`
    carrying boxes); an unrealized / UNKNOWN / INCONSISTENT geometry is refused
    fail-closed — there is nothing trustworthy to observe.  For every constraint:

    * a DELEGATED relation is overwritten with the single RCC-8 base relation the
      geometry actually realizes for that pair (epistemic ``derived_exact``, recorded in
      provenance);
    * a GROUNDED or INVARIANT relation — a *decided* relation of the complement — is left
      untouched, but is fail-closed cross-checked against the geometry: if the observed
      relation lies outside that decided relation's mask the whole ``get`` is refused
      (guarding against a stale / externally-supplied realization that silently
      contradicts the invariant or grounded core).  These decided, un-edited relations are
      the relational half of the preserved complement C_B.

    ``get`` preserves C_A exactly: it does NOT re-solve, so the realization it observed is
    carried through byte-for-byte (the geometry's metric freedom is held constant while
    only the relational view moves).

    Honest limits: ``get`` is **forgetful and irreversible** — collapsing an observed pair
    onto one base relation discards whatever disjunction the delegated constraint carried,
    and the original relational topology (the pre-image / alternative solutions) is NOT
    recovered by any subsequent ``put``.  It provides no adjoint.  The turn appends one
    :class:`HistoryEntry` and increments ``version``.
    """
    real = state.realization
    if real is None or real.status is not Realizability.REALIZED_IN_D or not real.boxes:
        raise MaintainerError(
            "get_observe_geometry needs a REALIZED_IN_D geometry (a reverse-verified "
            "Realization with boxes); refusing to observe an unrealized/unknown geometry"
        )

    rels = reverse.extract_relations(real)   # {(i,j): single-base mask}, exact readback

    cs = list(state.spec.constraints)
    observed = 0
    for i, c in enumerate(cs):
        if c.status in (Status.GROUNDED, Status.INVARIANT):
            # decided complement — preserved untouched, but fail-closed cross-check that the
            # observed geometry actually still satisfies it (guards externally-supplied or
            # stale realizations that silently contradict the invariant/grounded core).
            m = _observed_mask(rels, c.src, c.dst)
            if m is not None and (m & c.rcc8_mask) == 0:
                raise MaintainerError(
                    f"observed geometry realizes {'|'.join(rcc8.names(m))} for decided "
                    f"relation {c.id!r}, which its {c.status.value} mask "
                    f"{'|'.join(rcc8.names(c.rcc8_mask))} forbids (fail-closed)"
                )
            continue
        m = _observed_mask(rels, c.src, c.dst)
        if m is None:
            continue  # pair has no box to observe -> leave the delegated relation as-is
        prov = Provenance(
            origin="get-observe",
            actor="solver",
            activity="get_observe_geometry",
            note=f"derived_exact := {'|'.join(rcc8.names(m))}",
        )
        cs[i] = replace(c, rcc8_mask=m, prov=prov)
        observed += 1

    new_spec = state.spec.with_constraints(tuple(cs))

    version = state.version + 1
    entry = HistoryEntry(
        port="geometry",
        op="get_observe_geometry",
        version=version,
        detail=f"observed {observed} delegated relation(s); decided complement preserved",
        status=real.status,
    )
    return SessionState(
        spec=new_spec,
        realization=real,          # C_A held constant: same geometry, not re-solved
        history=state.history + (entry,),
        version=version,
        editing_port="geometry",
    )
