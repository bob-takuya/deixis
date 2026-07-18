"""Machine-readable disclosure of a *lossy* projection (LossReport).

The Deixis IR is deliberately honest about what a projection throws away. Whenever a
view/export maps the relational IR onto a narrower target (a group-view, a DE-9IM
matrix, an overlay round-trip, an RDF/turtle export, …) some information is preserved
exactly, some is approximated, and some is dropped outright. ``LossReport`` is the single
type all of those emit so a downstream reader can *inspect the loss instead of guessing*.

Design commitments (consistent with the adversarially-verified spec):

- ``LossReport`` is ``@frozen`` and carries only plain strings / tuples / bool +
  optional :class:`~deixis.core.ids.Provenance`. It has NO geometry, so — unlike the
  rest of the IR — it needs no Fraction handling on the wire.
- ``to_json`` / ``from_json`` follow the ``io.serialize`` house style: tuples become
  lists, provenance is delegated to the shared codec, and the round trip reconstructs an
  equal object.
- ``merge`` composes the loss of a *pipeline of projections* (A→B then B→C) into a single
  end-to-end report.

Over-claim guard (load-bearing, tested):
    ``reversible=True`` means ONLY that the projection round-trips *on the declared
    observation set* — i.e. re-deriving the observations named in ``preserved`` from the
    target reproduces them. It does NOT assert that every geometric realization, or any
    information outside ``preserved``, survives. A projection that drops or approximates
    anything is therefore NOT reversible in the whole-object sense; ``merge`` keeps
    ``reversible`` true only when *every* stage is itself reversible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ..core.ids import Provenance

# Reuse the shared provenance codec so LossReport serializes provenance identically to
# every other envelope in io.serialize (importing does not create a cycle: serialize does
# not import this module).
from .serialize import provenance_to_json, provenance_from_json


@dataclass(frozen=True)
class LossReport:
    """A single, machine-readable record of what a lossy projection kept / bent / lost.

    Fields:
        source_kind: kind of the input representation (e.g. ``"relspec"``, ``"realization"``).
        target_kind: kind of the output representation (e.g. ``"de9im"``, ``"group_view"``,
            ``"rdf"``, ``"overlay_roundtrip"``).
        preserved:   names of observations/aspects carried across *exactly*.
        approximated: names of aspects carried across only *approximately* (lossy but present).
        dropped:     names of aspects the target cannot represent at all.
        reversible:  see the module docstring — restricted to "round-trips on the declared
            observation set (``preserved``)", NOT a whole-realization guarantee.
        scope_note:  free-text scoping of the claim (what O the reversibility is asserted over,
            caveats, domain restrictions).
        provenance:  optional PROV-like record of who/what produced this projection.
    """

    source_kind: str
    target_kind: str
    preserved: tuple[str, ...] = ()
    approximated: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    reversible: bool = False
    scope_note: str = ""
    provenance: Optional[Provenance] = None

    def __post_init__(self) -> None:
        """Integrity guard so a report is trustworthy *as a single object* (not only after merge).

        Enforced invariants:
        - ``reversible`` is a real ``bool`` (not a truthy string/int) — this is an
          over-claim-sensitive flag, so we refuse to silently coerce.
        - the three aspect sets contain no empty strings, no duplicates, and are pairwise
          disjoint: an aspect cannot be simultaneously preserved / approximated / dropped.
        - ``reversible=True`` REQUIRES a non-empty ``preserved`` — reversibility is defined
          only relative to a declared observation set O, and ``preserved`` *is* that O. A
          reversible claim with an empty O would be vacuous, exactly the kind of over-claim
          this type exists to prevent.
        """
        if not isinstance(self.reversible, bool):
            raise TypeError("reversible must be a bool (over-claim-sensitive flag)")
        for label, seq in (("preserved", self.preserved),
                           ("approximated", self.approximated),
                           ("dropped", self.dropped)):
            if any((not isinstance(x, str)) or x == "" for x in seq):
                raise ValueError(f"{label} must contain only non-empty strings")
            if len(set(seq)) != len(seq):
                raise ValueError(f"{label} contains duplicate aspect names")
        p, a, d = set(self.preserved), set(self.approximated), set(self.dropped)
        if (p & a) or (p & d) or (a & d):
            raise ValueError(
                "preserved/approximated/dropped must be pairwise disjoint "
                "(an aspect has exactly one fate)"
            )
        if self.reversible and not self.preserved:
            raise ValueError(
                "reversible=True requires a non-empty 'preserved' (the declared "
                "observation set O it round-trips); empty-O reversibility is vacuous"
            )

    # ------------------------------------------------------------------ JSON

    def to_json(self) -> dict:
        return {
            "source_kind": self.source_kind,
            "target_kind": self.target_kind,
            "preserved": list(self.preserved),
            "approximated": list(self.approximated),
            "dropped": list(self.dropped),
            "reversible": bool(self.reversible),
            "scope_note": self.scope_note,
            "provenance": provenance_to_json(self.provenance),
        }

    @classmethod
    def from_json(cls, d: Any) -> "LossReport":
        rev = d.get("reversible", False)
        if not isinstance(rev, bool):
            # Do NOT coerce: a wire ``"false"`` / ``0`` must not become a positive
            # reversibility claim. (``__post_init__`` would also reject it, but failing
            # here gives a decode-local error.)
            raise TypeError("reversible on the wire must be a JSON bool, not "
                            f"{type(rev).__name__}")
        return cls(
            source_kind=d["source_kind"],
            target_kind=d["target_kind"],
            preserved=tuple(d.get("preserved", ()) or ()),
            approximated=tuple(d.get("approximated", ()) or ()),
            dropped=tuple(d.get("dropped", ()) or ()),
            reversible=rev,
            scope_note=d.get("scope_note", ""),
            provenance=provenance_from_json(d.get("provenance")),
        )


# module-level convenience aliases (io.serialize house style: free functions per type)

def lossreport_to_json(r: LossReport) -> dict:
    return r.to_json()


def lossreport_from_json(d: Any) -> LossReport:
    return LossReport.from_json(d)


# ---------------------------------------------------------------- merge


def merge(reports: Iterable["LossReport"]) -> LossReport:
    """Compose the loss of a *chain* of projections A->B, B->C, ... into one report.

    PRECONDITION — shared aspect vocabulary. ``merge`` matches aspects across stages by
    exact string equality, so it assumes every stage names the same aspect the same way
    (a stable, shared observation vocabulary). It does NOT model rename/split/join of
    aspects; feeding it stages that use divergent names silently understates or overstates
    loss. Callers that cross vocabularies must reconcile names first.

    Chain integrity: the reports must form a connected pipeline — each stage's
    ``target_kind`` must equal the next stage's ``source_kind``; otherwise ``ValueError``.
    An empty chain is rejected (identity is type-relative: there is no untyped "X->X").

    Conservative, worst-guarantee-wins classification (never more faithful than the worst
    stage), with an *open* aspect universe so silence is never mistaken for a guarantee:

    - ``dropped`` end-to-end = union over stages (ANY stage that drops an aspect loses it).
    - ``preserved`` end-to-end = the INTERSECTION of the stages' ``preserved`` sets (an
      aspect is preserved only if EVERY stage explicitly preserves it), minus anything
      dropped. An aspect preserved by some stages but simply unmentioned by another is
      therefore NOT claimed preserved — its fate through that stage is unknown, and we stay
      silent rather than over-claim.
    - ``approximated`` end-to-end = aspects any stage approximated (and that end up neither
      dropped nor unanimously preserved).
    - ``reversible`` end-to-end is True ONLY IF every stage is reversible AND the composed
      ``preserved`` (the aspects round-tripped by all stages) is non-empty. A single
      non-reversible or fully-lossy stage collapses the chain's reversibility. This stays a
      declared-observation-set claim; it never asserts whole-realization recoverability.
    - ``provenance`` is not synthesized (kept ``None``); the caller owns end-to-end
      provenance. Per-stage provenance stays on the stage reports.

    This does NOT resurrect any whole-realization or "meaning-preserving IR" claim: it only
    tracks, per named aspect, the weakest guarantee seen along the chain.
    """
    reports = list(reports)
    if not reports:
        raise ValueError(
            "merge() needs at least one report: identity is type-relative "
            "(there is no untyped X->X projection to synthesize)"
        )

    # Chain must be connected: target of each stage feeds the source of the next.
    for prev, nxt in zip(reports, reports[1:]):
        if prev.target_kind != nxt.source_kind:
            raise ValueError(
                "discontinuous projection chain: "
                f"{prev.target_kind!r} does not feed {nxt.source_kind!r}"
            )

    dropped_any: set[str] = set()
    approximated_any: set[str] = set()
    preserved_all: set[str] = set(reports[0].preserved)
    for r in reports:
        dropped_any.update(r.dropped)
        approximated_any.update(r.approximated)
        preserved_all &= set(r.preserved)  # unanimity required

    dropped = tuple(sorted(dropped_any))
    preserved = tuple(sorted(preserved_all - dropped_any))
    approximated = tuple(sorted(approximated_any - dropped_any - set(preserved)))

    # Reversible only if every stage is reversible AND a non-empty common O survives.
    reversible = all(r.reversible for r in reports) and len(preserved) > 0

    chain = " -> ".join(
        [reports[0].source_kind] + [r.target_kind for r in reports]
    )
    scope_bits = [f"merged {len(reports)} projection(s): {chain}"]
    scope_bits += [r.scope_note for r in reports if r.scope_note]
    scope_note = "; ".join(scope_bits)

    return LossReport(
        source_kind=reports[0].source_kind,
        target_kind=reports[-1].target_kind,
        preserved=preserved,
        approximated=approximated,
        dropped=dropped,
        reversible=reversible,
        scope_note=scope_note,
        provenance=None,
    )
