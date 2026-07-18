"""Tests for deixis.io.rdf_export (RelSpec -> BOT/PROV-O Turtle + honest LossReport).

Verified without rdflib: a tiny pure-Python parser splits the Turtle into @prefix
declarations and flat ``S P O .`` triples and checks structure + specific mappings.
The point of the module is *honesty*, so the tests assert both what IS emitted (BOT
adjacency/containment/interface, PROV activity) and what is DISCLOSED as lost.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from deixis.core import rcc8
from deixis.core.types import (
    Status, Epistemic, ContactDim,
    Region, RelationConstraint, IncidenceWitness, OverlayAtom,
    GroundingDecision, RelSpec,
)
from deixis.io import rdf_export as R


# ---------------------------------------------------------------- tiny Turtle parser

def _parse(turtle: str):
    """Return (prefixes: dict, triples: list[(s,p,o)]). Deliberately minimal.

    Handles the exact shape rdf_export emits: one ``@prefix`` per line and one flat
    ``S P O .`` triple per line, with objects that are either <IRI>, prefixed:name,
    a double-quoted literal (no spaces-across-tokens edge cases beyond escaped quotes),
    or a boolean/number bareword."""
    prefixes: dict[str, str] = {}
    triples: list[tuple[str, str, str]] = []
    for raw in turtle.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@prefix"):
            # @prefix p: <ns> .
            body = line[len("@prefix"):].strip().rstrip(".").strip()
            pfx, ns = body.split(":", 1)
            ns = ns.strip()
            assert ns.startswith("<") and ns.endswith(">")
            prefixes[pfx.strip()] = ns[1:-1]
            continue
        assert line.endswith("."), f"triple must end with a period: {line!r}"
        s, p, o = _split_triple(line[:-1].strip())
        triples.append((s, p, o))
    return prefixes, triples


def _split_triple(body: str):
    """Split 'S P O' respecting a quoted-string object (which may contain spaces)."""
    # subject
    s, rest = _next_token(body)
    p, rest = _next_token(rest)
    o = rest.strip()
    return s, p, o


def _next_token(text: str):
    text = text.strip()
    i = 0
    n = len(text)
    while i < n and not text[i].isspace():
        i += 1
    return text[:i], text[i:]


# ---------------------------------------------------------------- fixtures

def _iri(local: str) -> str:
    return f"<{R.DATA}{local}>"


def _simple_spec() -> RelSpec:
    r1 = Region(id="r1", semantic_type="living")
    r2 = Region(id="r2", semantic_type="circulation")
    ec = RelationConstraint(src="r1", dst="r2", rcc8_mask=rcc8.bit("EC"),
                            status=Status.INVARIANT, id="c_ec")
    return RelSpec(regions=(r1, r2), constraints=(ec,))


# ---------------------------------------------------------------- structural validity

def test_prefixes_and_triple_structure():
    turtle, _ = R.to_rdf_with_report(_simple_spec())
    prefixes, triples = _parse(turtle)
    # required prefixes are declared
    for p in ("rdf", "rdfs", "bot", "prov", "deixis", ""):
        assert p in prefixes
    assert prefixes["bot"] == R.BOT
    assert prefixes["prov"] == R.PROV
    assert prefixes[""] == R.DATA
    # every triple is (subject, predicate, object), non-empty
    assert triples
    for s, p, o in triples:
        assert s and p and o


def test_to_turtle_matches_report_turtle():
    spec = _simple_spec()
    assert R.to_turtle(spec) == R.to_rdf_with_report(spec)[0]


# ---------------------------------------------------------------- region mapping

def test_region_becomes_bot_space_with_label():
    turtle, _ = R.to_rdf_with_report(_simple_spec())
    _, triples = _parse(turtle)
    assert (_iri("r1"), "a", "bot:Space") in triples
    assert (_iri("r1"), "rdfs:label", '"living"') in triples
    assert (_iri("r2"), "a", "bot:Space") in triples


# ---------------------------------------------------------------- contact mapping

def test_ec_maps_to_adjacent_zone():
    turtle, report = R.to_rdf_with_report(_simple_spec())
    _, triples = _parse(turtle)
    assert (_iri("r1"), "bot:adjacentZone", _iri("r2")) in triples
    # invariant EC has no delegated-freedom / disjunction loss
    assert "delegated_freedom" not in report.categories()
    assert "disjunction" not in report.categories()


def test_containment_direction_for_tpp_and_tppi():
    tpp = RelationConstraint(src="r1", dst="r2", rcc8_mask=rcc8.bit("TPP"),
                             status=Status.GROUNDED, id="c_tpp")
    tppi = RelationConstraint(src="r1", dst="r2", rcc8_mask=rcc8.bit("TPPi"),
                              status=Status.GROUNDED, id="c_tppi")
    spec = RelSpec(regions=(Region(id="r1"), Region(id="r2")), constraints=(tpp, tppi))
    _, triples = _parse(R.to_turtle(spec))
    # TPP: src is a part of dst -> dst contains src
    assert (_iri("r2"), "bot:containsZone", _iri("r1")) in triples
    # TPPi: src contains dst
    assert (_iri("r1"), "bot:containsZone", _iri("r2")) in triples


def test_po_emits_adjacency_but_reports_overlap_loss():
    po = RelationConstraint(src="r1", dst="r2", rcc8_mask=rcc8.bit("PO"),
                            status=Status.GROUNDED, id="c_po")
    spec = RelSpec(regions=(Region(id="r1"), Region(id="r2")), constraints=(po,))
    turtle, report = R.to_rdf_with_report(spec)
    _, triples = _parse(turtle)
    assert (_iri("r1"), "bot:adjacentZone", _iri("r2")) in triples
    assert "partial_overlap" in report.categories()
    assert report.by_category("partial_overlap")[0].subject == "c_po"


# ---------------------------------------------------------------- loss disclosure

def test_disjunction_relation_is_reported_and_not_fabricated():
    disj = RelationConstraint(src="r1", dst="r2",
                              rcc8_mask=rcc8.mask("EC", "PO"),
                              status=Status.GROUNDED, id="c_disj")
    spec = RelSpec(regions=(Region(id="r1"), Region(id="r2")), constraints=(disj,))
    turtle, report = R.to_rdf_with_report(spec)
    _, triples = _parse(turtle)
    # no crisp BOT topology triple fabricated for a disjunction
    assert not any(p in ("bot:adjacentZone", "bot:containsZone") for _, p, _ in triples)
    assert "disjunction" in report.categories()
    # but the disjunction IS mirrored into deixis:rcc8 so the file stays self-describing
    assert (_iri("c_disj"), "deixis:rcc8", '"EC|PO"') in triples


def test_delegated_relation_reported_as_freedom_loss():
    ec = RelationConstraint(src="r1", dst="r2", rcc8_mask=rcc8.bit("EC"),
                            status=Status.DELEGATED, id="c_ec")
    spec = RelSpec(regions=(Region(id="r1"), Region(id="r2")), constraints=(ec,))
    turtle, report = R.to_rdf_with_report(spec)
    _, triples = _parse(turtle)
    # still emits the best-effort BOT triple ...
    assert (_iri("r1"), "bot:adjacentZone", _iri("r2")) in triples
    # ... and discloses that its delegated freedom is not carried by BOT
    assert "delegated_freedom" in report.categories()
    assert (_iri("c_ec"), "deixis:editStatus", '"delegated"') in triples


def test_contact_dimension_line_reported():
    w_face = IncidenceWitness(id="w_face", cell_ids=("k0",), incident_regions=("r1", "r2"),
                              intended_contact_dimension=ContactDim.FACE.value)
    w_line = IncidenceWitness(id="w_line", cell_ids=("k1",), incident_regions=("r1", "r2"),
                              intended_contact_dimension=ContactDim.LINE.value)
    spec = RelSpec(regions=(Region(id="r1"), Region(id="r2")), witnesses=(w_face, w_line))
    turtle, report = R.to_rdf_with_report(spec)
    _, triples = _parse(turtle)
    assert (_iri("w_face"), "a", "bot:Interface") in triples
    assert (_iri("w_face"), "bot:interfaceOf", _iri("r1")) in triples
    assert (_iri("w_line"), "deixis:contactDimension", '"line"') in triples
    # face contact fits bot:Interface (no loss); line contact does not
    cd = report.by_category("contact_dimension")
    subjects = {it.subject for it in cd}
    assert "w_line" in subjects
    assert "w_face" not in subjects


def test_overlay_atom_reported():
    o = OverlayAtom(id="o1", members=frozenset({"r1", "r2"}), nonempty=True)
    spec = RelSpec(regions=(Region(id="r1"), Region(id="r2")), overlays=(o,))
    _, report = R.to_rdf_with_report(spec)
    assert "overlay_atom" in report.categories()
    assert report.by_category("overlay_atom")[0].subject == "o1"


# ---------------------------------------------------------------- grounding -> PROV

def test_grounding_becomes_prov_activity():
    ec = RelationConstraint(src="r1", dst="r2", rcc8_mask=rcc8.bit("EC"),
                            status=Status.GROUNDED, id="c_ec")
    g = GroundingDecision(id="g1", target="c_ec", fixed_value_or_domain="x==21/5",
                          actor="alice", rationale="south wall", reversibility="low")
    spec = RelSpec(regions=(Region(id="r1"), Region(id="r2")),
                   constraints=(ec,), groundings=(g,))
    turtle, _ = R.to_rdf_with_report(spec)
    _, triples = _parse(turtle)
    assert (_iri("g1"), "a", "prov:Activity") in triples
    assert (_iri("g1"), "prov:wasAssociatedWith", _iri("agent/alice")) in triples
    assert (_iri("agent/alice"), "a", "prov:Agent") in triples
    assert (_iri("g1"), "prov:generated", _iri("c_ec")) in triples
    assert (_iri("g1"), "deixis:rationale", '"south wall"') in triples


# ---------------------------------------------------------------- report shape / robustness

def test_loss_report_json_and_container_protocol():
    disj = RelationConstraint(src="r1", dst="r2", rcc8_mask=rcc8.mask("EC", "PO"),
                              status=Status.DELEGATED, id="c_disj")
    o = OverlayAtom(id="o1", members=frozenset({"r1", "r2"}))
    spec = RelSpec(regions=(Region(id="r1"), Region(id="r2")),
                   constraints=(disj,), overlays=(o,))
    _, report = R.to_rdf_with_report(spec)
    assert len(report) == len(list(report)) == len(report.to_json())
    assert bool(report) is True
    for item in report.to_json():
        assert set(item) == {"category", "subject", "detail"}
        # honest wording: not claiming impossibility-in-principle
        assert "原理的" not in item["detail"]


def test_empty_spec_has_no_losses_and_valid_prefixes():
    turtle, report = R.to_rdf_with_report(RelSpec())
    prefixes, triples = _parse(turtle)
    assert "bot" in prefixes
    assert triples == []
    assert len(report) == 0
    assert bool(report) is False


def test_ids_with_colons_are_kept_in_iri():
    r = Region(id="n:region:0", semantic_type="")
    turtle, _ = R.to_rdf_with_report(RelSpec(regions=(r,)))
    _, triples = _parse(turtle)
    assert (f"<{R.DATA}n:region:0>", "a", "bot:Space") in triples


def test_ids_with_spaces_are_percent_escaped():
    r = Region(id="odd id", semantic_type="")
    turtle, _ = R.to_rdf_with_report(RelSpec(regions=(r,)))
    _, triples = _parse(turtle)
    assert (f"<{R.DATA}odd%20id>", "a", "bot:Space") in triples
