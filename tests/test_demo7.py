"""Tests for demo7 — the full round-trip / 7-stage witnessed-IR vs RCC-8-only demo.

Each stage (1)-(7) of :mod:`deixis.demos.demo7_roundtrip` has at least one assertion
here. The load-bearing claims the thesis rests on are checked explicitly:

* (2) both contacts collapse to the SAME RCC-8 relation (EC) yet lift to DISTINCT
  witnesses (contact dim 2 vs 1);
* (6) every family solution satisfies BOTH witnesses (face + edge) under exact
  reverse-verification, and the invariant contacts are *preserved* across the
  grounding swap while only the delegated A-C freedom changes;
* (7) an RCC-8-only description CANNOT distinguish an edge contact from the required
  face contact — it accepts a geometry the witnessed spec rejects.

Everything is exact (Fraction / integer bit arithmetic); no float is asserted on.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from deixis.core import rcc8
from deixis.core.types import (
    ContactDim,
    GroundingDecision,
    Realizability,
    Status,
)
from deixis.grounding.grounding_op import GroundingError, ground
from deixis.incidence.witness import (
    collapses_to_same_rcc,
    witnesses_distinguishable,
)
from deixis.solver.family import canonical_key, compare_realizations
from deixis.verify import reverse

from deixis.demos import demo7_roundtrip as d7
from deixis.demos.demo7_roundtrip import (
    A,
    B,
    C,
    CID_AB,
    CID_AC,
    CID_BC,
    WID_AB,
    WID_BC,
)


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def geometry():
    return d7.build_original_geometry()


@pytest.fixture
def lifted(geometry):
    return d7.lift(geometry)


@pytest.fixture
def approved(lifted):
    return d7.approve(lifted)


@pytest.fixture
def scenarios():
    return d7.grounding_scenarios()


@pytest.fixture
def family(approved, scenarios):
    sc0, sc1 = scenarios
    return d7.solve(approved, [sc0, sc1])


# ===================================================================== stage 1
def test_stage1_original_geometry_is_face_and_edge(geometry):
    """A-B is a FACE contact (dim 2), B-C is an EDGE contact (dim 1); all exact Fraction."""
    assert geometry.status is Realizability.REALIZED_IN_D
    # exact rational coordinates only
    for bx in geometry.boxes:
        for v in bx.lo + bx.hi:
            assert isinstance(v, Fraction)
    assert int(reverse.extract_contact_dimension(geometry, A, B)) == int(ContactDim.FACE)
    assert int(reverse.extract_contact_dimension(geometry, B, C)) == int(ContactDim.LINE)


# ===================================================================== stage 2
def test_stage2_lift_both_ec_distinct_witnesses(geometry, lifted):
    """Load-bearing: both contacts are the SAME RCC-8 symbol EC, but DISTINCT witnesses."""
    # both A-B and B-C read back as EC (the one symbol RCC-8 has for both contacts)
    assert d7._relation_of(geometry, A, B) == rcc8.mask("EC")
    assert d7._relation_of(geometry, B, C) == rcc8.mask("EC")

    w_ab = lifted.witness(WID_AB)
    w_bc = lifted.witness(WID_BC)
    assert w_ab is not None and w_bc is not None
    # same RCC-8 entailment, yet distinguishable as witnesses
    assert collapses_to_same_rcc(w_ab, w_bc) is True
    assert witnesses_distinguishable(w_ab, w_bc) is True
    assert w_ab.intended_contact_dimension == int(ContactDim.FACE)
    assert w_bc.intended_contact_dimension == int(ContactDim.LINE)


def test_stage2_lift_does_not_promote_observation(lifted):
    """A lift observation is OBSERVED_TOLERANT — NOT auto-promoted to the invariant core."""
    for w in lifted.witnesses:
        assert w.epistemic.value == "observed_tolerant"
    assert lifted.invariant.witness_invariants == frozenset()
    assert lifted.invariant.relation_invariants == frozenset()
    # A-C is left as delegated design freedom over the DC|PO envelope
    c_ac = lifted.constraint(CID_AC)
    assert c_ac.status is Status.DELEGATED
    assert c_ac.rcc8_mask == rcc8.mask("DC", "PO")


# ===================================================================== stage 3
def test_stage3_approve_promotes_core(lifted, approved):
    """Approval puts the two EC relations + two witnesses into the invariant core."""
    assert approved.invariant.relation_invariants == frozenset({CID_AB, CID_BC})
    assert approved.invariant.witness_invariants == frozenset({WID_AB, WID_BC})
    assert approved.constraint(CID_AB).status is Status.INVARIANT
    assert approved.constraint(CID_BC).status is Status.INVARIANT
    # A-C stays delegated (the freedom the family will exercise)
    assert approved.constraint(CID_AC).status is Status.DELEGATED
    # purity: approving returned a NEW spec; the lifted input is untouched
    assert lifted.invariant.witness_invariants == frozenset()
    assert lifted.constraint(CID_AB).status is Status.DELEGATED


def test_stage3_approve_flips_epistemic_to_asserted(approved):
    for w in approved.witnesses:
        assert w.epistemic.value == "asserted"


def test_stage3_approve_witnesses_failclosed_on_unknown_id(lifted):
    """Approving a non-existent witness id is rejected (no dangling invariant)."""
    with pytest.raises(ValueError):
        d7.approve_witnesses(lifted, "w:does-not-exist")


# ===================================================================== stage 4
def test_stage4_two_distinct_grounding_sets(scenarios):
    sc0, sc1 = scenarios
    # each scenario grounds the delegated A-C relation a different way
    ac0 = next(d for d in sc0 if d.target == CID_AC)
    ac1 = next(d for d in sc1 if d.target == CID_AC)
    assert ac0.fixed_value_or_domain == "DC"
    assert ac1.fixed_value_or_domain == "PO"
    # region-targeting (orientation / basepoint) decisions present and different
    assert any(d.target == A for d in sc0)
    assert any(d.target == A for d in sc1)
    assert {d.fixed_value_or_domain for d in sc0} != {d.fixed_value_or_domain for d in sc1}


def test_stage4_invariant_core_is_failclosed_against_grounding(approved):
    """The approved A-B / B-C contacts cannot be grounded — fail-closed guard."""
    bad = GroundingDecision(id="g:illegal", target=CID_AB, fixed_value_or_domain="DC")
    with pytest.raises(GroundingError):
        ground(approved, CID_AB, bad)


# ===================================================================== stage 5
def test_stage5_family_has_two_distinct_realized_solutions(family):
    """solve_family yields two genuinely different, both-realized solutions."""
    assert len(family) == 2
    for r in family:
        assert r.status is Realizability.REALIZED_IN_D
        assert r.boxes
    # genuinely different geometry (not symmetric duplicates)
    assert canonical_key(family[0]) != canonical_key(family[1])
    # the delegated A-C relation is what differs
    assert d7._relation_of(family[0], A, C) == rcc8.mask("DC")
    assert d7._relation_of(family[1], A, C) == rcc8.mask("PO")


def test_stage5_scenario_ids_and_provenance(family):
    assert [r.scenario_id for r in family] == ["s0", "s1"]


# ===================================================================== stage 6
def test_stage6_every_solution_satisfies_both_witnesses(approved, family):
    """Load-bearing: each family solution realizes A-B FACE and B-C EDGE exactly."""
    for r in family:
        chk = d7.verify_solution(approved, r)
        assert chk.ok
        assert chk.contact_dim_ab == int(ContactDim.FACE)
        assert chk.contact_dim_bc == int(ContactDim.LINE)
        assert chk.witness_violations == ()
        # the exact reverse layer agrees the invariant relations are satisfied
        assert CID_AB in chk.satisfied
        assert CID_BC in chk.satisfied


def test_stage6_reverse_verify_reports_no_witness_violation(approved, family):
    for r in family:
        report = reverse.verify(approved, r)
        assert report.witness_violations == ()
        assert report.violated == ()


def test_stage6_solution_coordinates_are_exact(family):
    """Exactness: every coordinate of every family solution is a Fraction (no float)."""
    for r in family:
        assert r.boxes
        for bx in r.boxes:
            for v in bx.lo + bx.hi:
                assert isinstance(v, Fraction)


def test_stage6_roundtrip_closes_on_original_contacts(approved, scenarios):
    """One round-trip closes: re-solving under the A-C=DC scenario reproduces the
    original scene's witnessed contacts (A-B FACE, B-C EDGE, A-C DC) — the observations
    that were lifted and approved are recovered from the regenerated geometry."""
    original = d7.build_original_geometry()
    sc0, _ = scenarios
    regenerated = d7.solve(approved, [sc0])[0]
    cmp = compare_realizations(original, regenerated, approved.observation)
    ab_key = ("contact_dimension", A, B)
    bc_key = ("contact_dimension", B, C)
    ac_key = ("rcc8", A, C)
    assert cmp["preserved"].get(ab_key) == int(ContactDim.FACE)
    assert cmp["preserved"].get(bc_key) == int(ContactDim.LINE)
    assert cmp["preserved"].get(ac_key) == "DC"
    # the witnessed contacts are not among the items that changed
    assert ab_key not in cmp["changed"]
    assert bc_key not in cmp["changed"]


def test_stage6_invariant_preserved_delegated_changed(approved, family):
    """Across the grounding swap: witnessed contacts PRESERVED, A-C relation CHANGED."""
    cmp = d7.compare_family(family[0], family[1], approved)
    ab_key = ("contact_dimension", A, B)
    bc_key = ("contact_dimension", B, C)
    ac_key = ("rcc8", A, C)
    # invariant core: identical contact dimensions in both solutions
    assert cmp["preserved"].get(ab_key) == int(ContactDim.FACE)
    assert cmp["preserved"].get(bc_key) == int(ContactDim.LINE)
    # delegated freedom: the A-C relation moved
    assert cmp["changed"].get(ac_key) == ("DC", "PO")
    # the contact-dimension observations are NOT among the changed items
    assert ab_key not in cmp["changed"]
    assert bc_key not in cmp["changed"]


# ===================================================================== stage 7
def test_stage7_edge_geometry_is_edge_not_face():
    edge = d7.build_edge_geometry()
    assert edge.status is Realizability.REALIZED_IN_D
    assert int(reverse.extract_contact_dimension(edge, A, B)) == int(ContactDim.LINE)
    # exact coordinates, and it satisfies every RCC-8 mask of the approved spec
    for bx in edge.boxes:
        for v in bx.lo + bx.hi:
            assert isinstance(v, Fraction)


def test_stage7_edge_geometry_matches_approved_rcc8_masks(approved):
    """The edge geometry satisfies every RCC-8 mask of the approved spec (its only defect
    is the witnessed A-B contact dimension) — so stage 7 isolates exactly that difference."""
    edge = d7.build_edge_geometry()
    rcc_report = reverse.verify(d7.rcc8_only_view(approved), edge)
    assert rcc_report.violated == ()
    assert rcc_report.undecided == ()
    assert set(rcc_report.satisfied) == {CID_AB, CID_BC, CID_AC}


def test_stage7_rcc8_only_view_clears_witness_invariants(approved):
    """Stripping witnesses also clears witness_invariants — no dangling invariant id."""
    assert approved.invariant.witness_invariants  # precondition: the core has them
    rcc_spec = d7.rcc8_only_view(approved)
    assert rcc_spec.witnesses == ()
    assert rcc_spec.invariant.witness_invariants == frozenset()


def test_stage7_rcc8_only_cannot_distinguish(approved):
    """Load-bearing: RCC-8-only ACCEPTS an edge geometry the witnessed FACE spec REJECTS."""
    result = d7.rcc8_only_cannot_distinguish(approved)
    assert result.edge_contact_dim_ab == int(ContactDim.LINE)   # it's an edge, not a face
    assert result.witnessed_rejects is True                      # IR: contact_dim_mismatch
    assert result.rcc8_only_accepts is True                      # RCC-8: just 'EC', fine
    assert result.ir_strictly_stronger is True                   # the whole point


def test_stage7_witnessed_flags_contact_dim_mismatch(approved):
    """The witnessed rejection is specifically a contact_dim_mismatch on the A-B pair."""
    edge = d7.build_edge_geometry()
    report = reverse.verify(approved, edge)
    codes = {(v.code, frozenset(v.regions)) for v in report.witness_violations}
    assert ("contact_dim_mismatch", frozenset({A, B})) in codes


def test_stage7_rcc8_only_view_has_no_witnesses(approved):
    rcc_spec = d7.rcc8_only_view(approved)
    assert rcc_spec.witnesses == ()
    # the RCC-8 masks themselves survive (same qualitative relations)
    assert rcc_spec.constraint(CID_AB).rcc8_mask == rcc8.mask("EC")
    assert rcc_spec.constraint(CID_BC).rcc8_mask == rcc8.mask("EC")


# --------------------------------------------------------------------- smoke: main
def test_main_runs_headless(capsys):
    """The human-readable driver runs end-to-end and prints all seven stages."""
    d7.main()
    out = capsys.readouterr().out
    for stage in ("(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)"):
        assert stage in out
    assert "IR strictly stronger than RCC-8 here: True" in out
