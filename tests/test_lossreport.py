"""Tests for deixis.io.lossreport.LossReport.

Covers: construction, JSON round-trip (== reconstruction incl. provenance), merge
composition semantics (worst-guarantee-wins + all-stages-reversible), and the load-bearing
over-claim guard: reversible is a declared-observation-set claim, and merge refuses to
upgrade a chain that contains a single lossy / non-reversible stage.
"""
from __future__ import annotations

import pytest

from deixis.core.ids import Provenance
from deixis.io.lossreport import (
    LossReport, merge, lossreport_to_json, lossreport_from_json,
)


def _prov() -> Provenance:
    return Provenance(
        origin="project", actor="exporter", activity="group-view",
        inputs=("spec-1",), note="π",
    )


# ---------------------------------------------------------------- construction

def test_defaults():
    r = LossReport(source_kind="relspec", target_kind="de9im")
    assert r.preserved == ()
    assert r.approximated == ()
    assert r.dropped == ()
    assert r.reversible is False
    assert r.scope_note == ""
    assert r.provenance is None


def test_frozen():
    r = LossReport(source_kind="a", target_kind="b")
    with pytest.raises(Exception):
        r.source_kind = "c"  # type: ignore[misc]


# ---------------------------------------------------------------- JSON round-trip

def test_json_roundtrip_full():
    r = LossReport(
        source_kind="relspec", target_kind="group_view",
        preserved=("rcc8", "cell_identity"),
        approximated=("contact_dimension",),
        dropped=("atom_occupancy",),
        reversible=False,
        scope_note="reversible only over declared O",
        provenance=_prov(),
    )
    d = r.to_json()
    back = LossReport.from_json(d)
    assert back == r
    # module-level aliases behave identically
    assert lossreport_from_json(lossreport_to_json(r)) == r


def test_json_roundtrip_minimal():
    r = LossReport(source_kind="realization", target_kind="rdf")
    assert LossReport.from_json(r.to_json()) == r


def test_json_shapes_are_plain():
    r = LossReport(
        source_kind="a", target_kind="b",
        preserved=("x",), reversible=True, provenance=None,
    )
    d = r.to_json()
    assert isinstance(d["preserved"], list)
    assert isinstance(d["reversible"], bool)
    assert d["provenance"] is None


def test_from_json_tolerates_missing_keys():
    r = LossReport.from_json({"source_kind": "a", "target_kind": "b"})
    assert r == LossReport(source_kind="a", target_kind="b")


def test_from_json_rejects_non_bool_reversible():
    # A wire "false"/0 must NOT be coerced into a positive reversibility claim.
    with pytest.raises(TypeError):
        LossReport.from_json({"source_kind": "a", "target_kind": "b", "reversible": "false"})
    with pytest.raises(TypeError):
        LossReport.from_json({"source_kind": "a", "target_kind": "b", "reversible": 0})


# ---------------------------------------------------------------- integrity guard

def test_reject_reversible_without_declared_observation_set():
    # reversible is defined only relative to a declared O (== preserved); empty O is vacuous.
    with pytest.raises(ValueError):
        LossReport(source_kind="a", target_kind="b", preserved=(), reversible=True)


def test_reject_non_disjoint_aspect_sets():
    with pytest.raises(ValueError):
        LossReport(source_kind="a", target_kind="b",
                   preserved=("x",), dropped=("x",))


def test_reject_duplicate_and_empty_aspect_names():
    with pytest.raises(ValueError):
        LossReport(source_kind="a", target_kind="b", preserved=("x", "x"))
    with pytest.raises(ValueError):
        LossReport(source_kind="a", target_kind="b", dropped=("",))


def test_reject_non_bool_reversible_field():
    with pytest.raises(TypeError):
        LossReport(source_kind="a", target_kind="b", preserved=("x",), reversible=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------- merge

def test_merge_empty_chain_rejected():
    with pytest.raises(ValueError):
        merge([])


def test_merge_rejects_discontinuous_chain():
    a = LossReport(source_kind="relspec", target_kind="group_view")
    b = LossReport(source_kind="de9im", target_kind="rdf")  # source != a.target
    with pytest.raises(ValueError):
        merge([a, b])


def test_merge_single_is_passthrough_semantics():
    a = LossReport(
        source_kind="relspec", target_kind="de9im",
        preserved=("rcc8",), dropped=("atom_occupancy",), reversible=False,
    )
    m = merge([a])
    assert m.source_kind == "relspec"
    assert m.target_kind == "de9im"
    assert m.preserved == ("rcc8",)
    assert m.dropped == ("atom_occupancy",)
    assert m.reversible is False


def test_merge_chain_endpoints_and_worst_wins():
    a = LossReport(
        source_kind="relspec", target_kind="group_view",
        preserved=("rcc8", "cell_identity"), approximated=("contact_dimension",),
        reversible=True,
    )
    b = LossReport(
        source_kind="group_view", target_kind="rdf",
        preserved=("rcc8",), dropped=("cell_identity",), reversible=True,
    )
    m = merge([a, b])
    assert m.source_kind == "relspec"
    assert m.target_kind == "rdf"
    # cell_identity: preserved in a, dropped in b -> dropped (worst wins)
    assert m.dropped == ("cell_identity",)
    # contact_dimension: approximated somewhere, never dropped -> approximated
    assert m.approximated == ("contact_dimension",)
    # rcc8: preserved in both, never bent/lost -> preserved
    assert m.preserved == ("rcc8",)
    assert "relspec -> group_view -> rdf" in m.scope_note


def test_merge_reversible_requires_all_stages():
    a = LossReport(source_kind="s", target_kind="t", preserved=("x",), reversible=True)
    b_bad = LossReport(source_kind="t", target_kind="u", preserved=("x",), reversible=False)
    b_ok = LossReport(source_kind="t", target_kind="u", preserved=("x",), reversible=True)
    assert merge([a, b_bad]).reversible is False
    assert merge([a, b_ok]).reversible is True


def test_merge_reversible_needs_nonempty_common_O():
    # Both stages reversible, but they round-trip DISJOINT observation sets: nothing is
    # round-tripped end-to-end, so the composition must NOT claim reversibility.
    a = LossReport(source_kind="s", target_kind="t", preserved=("x",), reversible=True)
    b = LossReport(source_kind="t", target_kind="u", preserved=("y",), reversible=True)
    m = merge([a, b])
    assert m.reversible is False
    assert m.preserved == ()  # empty intersection


def test_merge_unmentioned_aspect_is_not_claimed_preserved():
    # x is preserved by stage a but simply unmentioned by stage b -> unknown -> NOT preserved.
    a = LossReport(source_kind="s", target_kind="t", preserved=("x", "y"), reversible=True)
    b = LossReport(source_kind="t", target_kind="u", preserved=("y",), reversible=True)
    m = merge([a, b])
    assert "x" not in m.preserved       # silence, not an over-claim
    assert m.preserved == ("y",)         # only the unanimous aspect survives
    assert "x" not in m.approximated and "x" not in m.dropped


def test_merge_does_not_synthesize_provenance():
    a = LossReport(source_kind="s", target_kind="s", preserved=("x",), reversible=True, provenance=_prov())
    assert merge([a, a]).provenance is None


# ---------------------------------------------------------------- over-claim guard

def test_reversible_is_not_a_whole_object_claim():
    """A projection can be reversible on its declared O yet still drop/approximate aspects.

    The type intentionally permits reversible=True alongside a non-empty dropped set: the
    claim is round-trip on ``preserved`` (declared observation set), NOT preservation of
    everything. This test pins that the type does not forbid it (semantics live in the
    docstring / consumers), while merge still refuses to upgrade a lossy chain's reversibility
    only via the all-stages rule."""
    r = LossReport(
        source_kind="relspec", target_kind="de9im",
        preserved=("rcc8",), dropped=("realization_geometry",),
        reversible=True, scope_note="round-trips rcc8 only; geometry not recoverable",
    )
    # reversible coexists with dropped aspects by design.
    assert r.reversible is True and r.dropped == ("realization_geometry",)
    # but a downstream lossy, non-reversible stage collapses the chain's reversibility.
    lossy = LossReport(source_kind="de9im", target_kind="rdf", reversible=False)
    assert merge([r, lossy]).reversible is False
