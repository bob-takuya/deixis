"""RelSpec -> RDF export (BOT building-topology + PROV-O provenance) with a LossReport.

This is a **projection**, not a lossless serialization. The canonical, exact,
round-trippable transport for a :class:`RelSpec` is :mod:`deixis.io.serialize`
(JSON with ``{num,den}`` Fractions). RDF/BOT is an *interoperability view*: it maps
the parts of the IR that standard vocabularies can carry, and **honestly discloses**
the parts they cannot, via a machine-readable :class:`LossReport`.

Honest framing (this is the whole point of the module):
- We emit only what BOT / PROV-O standard vocabulary can faithfully hold.
- Everything that BOT's *crisp* topology graph cannot hold *simultaneously* is not
  silently dropped and is not silently coerced into a misleading triple — it is
  reported. The wording is deliberately "BOT標準語彙では同時保持できない" (the standard
  vocabulary cannot co-hold it), NOT "原理的に表現不能" (impossible in principle).
- Information the IR keeps that has no BOT term (RCC-8 disjunctions, contact
  dimension, edit-authority status, overlay atoms) is additionally mirrored into a
  private ``deixis:`` vocabulary so the RDF file is self-describing, while the
  LossReport still records that a *BOT-standard* consumer will not see it.

Vocabulary mapping (single, definite RCC-8 base relations only):
    EC                 -> bot:adjacentZone      (+ optional bot:Interface from witness)
    TPP / NTPP         -> dst bot:containsZone src   (src is a part of dst)
    TPPi / NTPPi       -> src bot:containsZone dst
    PO                 -> bot:adjacentZone      (approximation; interior overlap lost -> report)
    EQ                 -> (no BOT term; report — regions kept distinct)
    DC                 -> (represented by absence of a topology triple — faithful, no report)
    disjunction mask   -> (no crisp BOT term; report — no fabricated triple)

    IncidenceWitness   -> bot:Interface + bot:interfaceOf, deixis:contactDimension
    GroundingDecision  -> prov:Activity (prov:wasAssociatedWith agent, prov:generated target)
    RelationConstraint -> deixis:RelationConstraint node carrying deixis:editStatus/deixis:rcc8

Pure-Python string generation (no rdflib dependency).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core import rcc8
from ..core.types import RelSpec, RelationConstraint, Region, Status

# ---------------------------------------------------------------- namespaces

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
BOT = "https://w3id.org/bot#"
PROV = "http://www.w3.org/ns/prov#"
DEIXIS = "https://deixis.example/vocab#"
DATA = "https://deixis.example/data#"

_PREFIXES: tuple[tuple[str, str], ...] = (
    ("rdf", RDF),
    ("rdfs", RDFS),
    ("xsd", XSD),
    ("bot", BOT),
    ("prov", PROV),
    ("deixis", DEIXIS),
    ("", DATA),
)

# contact-dimension int (ContactDim value) -> label
_CONTACT_LABEL: dict[int, str] = {0: "point", 1: "line", 2: "face"}


# ---------------------------------------------------------------- LossReport

@dataclass(frozen=True)
class LossItem:
    """One disclosed loss: a piece of IR information a BOT-standard consumer cannot see.

    ``category`` is a stable machine key; ``subject`` is the affected IR id; ``detail`` is
    an honest human explanation (co-hold limitation, not impossibility-in-principle)."""
    category: str
    subject: str
    detail: str

    def to_json(self) -> dict:
        return {"category": self.category, "subject": self.subject, "detail": self.detail}


@dataclass(frozen=True)
class LossReport:
    """Machine-readable disclosure of everything the RDF/BOT projection could not carry."""
    items: tuple[LossItem, ...] = ()

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def categories(self) -> set[str]:
        return {it.category for it in self.items}

    def by_category(self, category: str) -> tuple[LossItem, ...]:
        return tuple(it for it in self.items if it.category == category)

    def to_json(self) -> list[dict]:
        return [it.to_json() for it in self.items]


# ---------------------------------------------------------------- escaping / IRIs

# characters that must not appear literally inside a Turtle IRIREF <...>
_IRI_BAD = set(' <>"{}|^`\\') | {chr(c) for c in range(0x00, 0x21)}


def _iri(local: str) -> str:
    """Full IRI ``<DATA#local>`` for a data individual, percent-escaping unsafe chars.

    Ids may contain ``:`` (from IdGen) — legal in an IRI fragment, so kept — but space,
    angle brackets, quotes, braces, backslash and control chars are percent-encoded."""
    out = []
    for ch in local:
        if ch in _IRI_BAD:
            out.append("".join(f"%{b:02X}" for b in ch.encode("utf-8")))
        else:
            out.append(ch)
    return f"<{DATA}{''.join(out)}>"


def _lit(s: str) -> str:
    """A Turtle double-quoted string literal with the minimal required escaping."""
    esc = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{esc}"'


def _bool(b: bool) -> str:
    return "true" if b else "false"


def _constraint_node_id(c: RelationConstraint) -> str:
    """Stable local id for a constraint node (constraints may have an empty id)."""
    if c.id:
        return c.id
    return f"constraint/{c.src}--{c.dst}"


# ---------------------------------------------------------------- core builder

def _build(spec: RelSpec) -> tuple[list[str], list[LossItem]]:
    """Return (triple-lines, loss-items). Triple lines are flat ``S P O .`` Turtle."""
    lines: list[str] = []
    losses: list[LossItem] = []

    # -- regions -> bot:Space (a bot:Zone subclass) -----------------------------
    for r in spec.regions:
        s = _iri(r.id)
        lines.append(f"{s} a bot:Space .")
        if r.semantic_type:
            lines.append(f"{s} rdfs:label {_lit(r.semantic_type)} .")
            lines.append(f"{s} deixis:semanticType {_lit(r.semantic_type)} .")
        lines.append(f"{s} deixis:declaredOrigin {_bool(r.declared_origin)} .")
        if not r.declared_origin:
            losses.append(LossItem(
                "field_origin", r.id,
                "region originates from a field threshold-grounding (declared_origin=false); "
                "BOT標準語彙ではゾーンの由来(field vs designer)を保持できない — kept only in deixis:declaredOrigin",
            ))

    # -- relation constraints ---------------------------------------------------
    for c in spec.constraints:
        node = _iri(_constraint_node_id(c))
        src, dst = _iri(c.src), _iri(c.dst)
        rel_names = rcc8.names(c.rcc8_mask)
        # deixis mirror node (carries everything BOT cannot: rcc8 mask, edit status)
        lines.append(f"{node} a deixis:RelationConstraint .")
        lines.append(f"{node} deixis:source {src} .")
        lines.append(f"{node} deixis:target {dst} .")
        lines.append(f"{node} deixis:rcc8 {_lit('|'.join(rel_names) or 'EMPTY')} .")
        lines.append(f"{node} deixis:editStatus {_lit(c.status.value)} .")

        # BOT projection: only for a single, definite base relation
        if not rcc8.is_base(c.rcc8_mask):
            losses.append(LossItem(
                "disjunction", _constraint_node_id(c),
                f"relation is a disjunction ({'|'.join(rel_names) or 'EMPTY'}); "
                "BOT標準語彙は crisp topology のみで論理和関係を同時保持できない — no BOT triple emitted",
            ))
        else:
            name = rel_names[0]
            if name == "EC":
                lines.append(f"{src} bot:adjacentZone {dst} .")
            elif name == "PO":
                lines.append(f"{src} bot:adjacentZone {dst} .")
                losses.append(LossItem(
                    "partial_overlap", _constraint_node_id(c),
                    "PO (interiors overlap) approximated as bot:adjacentZone; "
                    "BOT標準語彙ではゾーンの部分重なりを保持できない — interior-overlap semantics lost",
                ))
            elif name in ("TPP", "NTPP"):
                lines.append(f"{dst} bot:containsZone {src} .")
            elif name in ("TPPi", "NTPPi"):
                lines.append(f"{src} bot:containsZone {dst} .")
            elif name == "EQ":
                losses.append(LossItem(
                    "equality", _constraint_node_id(c),
                    "EQ (region equality) has no BOT topology term; regions kept distinct — "
                    "BOT標準語彙では同一性を bot:* 関係として保持できない",
                ))
            # DC: represented by absence of any topology triple (faithful) — no loss.

        # delegated edit-authority is a degree of freedom BOT's crisp graph flattens
        if c.status == Status.DELEGATED:
            losses.append(LossItem(
                "delegated_freedom", _constraint_node_id(c),
                "relation is DELEGATED (left to the solver, not yet fixed); a BOT crisp "
                "adjacency/containment triple reads as fixed — BOT標準語彙では決定権配分(delegated自由度)を"
                "同時保持できない — kept only in deixis:editStatus",
            ))

    # -- witnesses -> bot:Interface + contact dimension -------------------------
    for w in spec.witnesses:
        wn = _iri(w.id)
        lines.append(f"{wn} a bot:Interface .")
        for rid in w.incident_regions:
            lines.append(f"{wn} bot:interfaceOf {_iri(rid)} .")
        dim = int(w.intended_contact_dimension)
        label = _CONTACT_LABEL.get(dim, str(dim))
        lines.append(f"{wn} deixis:contactDimension {_lit(label)} .")
        if dim != 2:
            losses.append(LossItem(
                "contact_dimension", w.id,
                f"contact dimension is {label} ({dim}); bot:Interface implies a shared 2D "
                "surface, so BOT標準語彙では点/線接触の接触次元を同時保持できない — kept only in "
                "deixis:contactDimension",
            ))

    # -- overlay atoms: no BOT term for partial-overlap membership --------------
    for o in spec.overlays:
        losses.append(LossItem(
            "overlay_atom", o.id,
            f"overlay atom over {{{', '.join(sorted(o.members))}}} (a non-empty partial "
            "overlap region); BOT標準語彙ではゾーンは部分重なりを持たず overlay atom を保持できない",
        ))

    # -- grounding decisions -> prov:Activity -----------------------------------
    for g in spec.groundings:
        gn = _iri(g.id)
        agent = _iri(f"agent/{g.actor}")
        lines.append(f"{gn} a prov:Activity .")
        lines.append(f"{gn} prov:wasAssociatedWith {agent} .")
        lines.append(f"{agent} a prov:Agent .")
        lines.append(f"{agent} rdfs:label {_lit(g.actor)} .")
        if g.target:
            lines.append(f"{gn} prov:generated {_iri(g.target)} .")
        if g.fixed_value_or_domain:
            lines.append(f"{gn} deixis:fixedValue {_lit(g.fixed_value_or_domain)} .")
        if g.rationale:
            lines.append(f"{gn} deixis:rationale {_lit(g.rationale)} .")
        lines.append(f"{gn} deixis:reversibility {_lit(g.reversibility)} .")

    return lines, losses


def _prefix_block() -> str:
    return "\n".join(f"@prefix {p}: <{ns}> ." for p, ns in _PREFIXES)


# ---------------------------------------------------------------- public API

def to_rdf_with_report(spec: RelSpec) -> tuple[str, LossReport]:
    """Project ``spec`` to a Turtle string plus a :class:`LossReport`.

    The Turtle carries every BOT/PROV-O-expressible fact and mirrors the rest into the
    private ``deixis:`` vocabulary; the LossReport enumerates what a *BOT-standard* reader
    will not see (disjunctions, partial overlap, sub-face contact dimension, delegated
    freedom, overlay atoms, equality, field-origin regions)."""
    lines, losses = _build(spec)
    turtle = _prefix_block() + "\n\n" + "\n".join(lines) + ("\n" if lines else "")
    return turtle, LossReport(items=tuple(losses))


def to_turtle(spec: RelSpec) -> str:
    """Project ``spec`` to a Turtle string (see :func:`to_rdf_with_report` for the report)."""
    return to_rdf_with_report(spec)[0]


__all__ = ["to_turtle", "to_rdf_with_report", "LossReport", "LossItem"]
