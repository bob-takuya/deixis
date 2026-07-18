"""Tests for the Grasshopper-facing pure-function logic layer (deixis.io.adapters).

Everything here is JSON-in / JSON-out and Rhino-independent. We check:

* each editing adapter round-trips through JSON and produces the intended spec change;
* solve -> verify agree on a realized spec, and the exact box coords survive the round trip;
* lift recovers relations as *observations* (DELEGATED) and tags recovered contact witnesses
  ``observed_tolerant`` (never invariant);
* every adapter fails closed to an ``{"error": ...}`` JSON object instead of raising.
"""
from __future__ import annotations

import json
from fractions import Fraction

import pytest

from deixis.core import rcc8
from deixis.core.ids import frac_to_json
from deixis.core.types import Epistemic, Status
from deixis.io import adapters


# ------------------------------------------------------------------ helpers
def _obj(js: str) -> dict:
    assert isinstance(js, str), "adapters must return a JSON *string*"
    return json.loads(js)


def _is_error(js: str) -> bool:
    d = _obj(js)
    return isinstance(d, dict) and "error" in d


def _empty_spec_json() -> str:
    # empty JSON object -> empty spec
    return "{}"


def _two_room_touch_spec() -> str:
    """Spec: a EC b with a LINE-contact witness (the realizable EC contact in 2D).

    In ``aabb_2d`` an EC (boundary-touch) pair shares at most a 1-D edge, so the
    intended contact dimension for a realizable 2D touch is LINE(1), not FACE(2)
    (which is a 3D-only contact). Regions are auto-declared by relate/witness.
    """
    s = adapters.relate("{}", "a", "b", ["EC"])
    assert not _is_error(s)
    s = adapters.witness(s, "a", "b", "line")
    assert not _is_error(s)
    return s


# ============================================================ serialization round trip
def test_spec_json_roundtrip_is_stable():
    s1 = _two_room_touch_spec()
    spec = adapters.spec_from_json(s1)
    s2 = json.dumps(adapters.spec_to_json(spec), ensure_ascii=False)
    # re-parsing the serialized spec yields an identical serialization (idempotent)
    assert _obj(s1) == _obj(s2)


def test_relate_accepts_dict_and_string_equivalently():
    from_str = adapters.relate("{}", "a", "b", ["EC"])
    from_dict = adapters.relate({}, "a", "b", ["EC"])
    assert _obj(from_str) == _obj(from_dict)


def test_full_spec_roundtrip_all_fields_and_fractions():
    # Build a spec that exercises EVERY container field, nullable + non-null provenance,
    # and non-integer fractional geometry (via lift), then assert serialize/deserialize
    # is a fixed point across all of it.
    from dataclasses import replace
    from deixis.core.types import OverlayAtom, Invariant, ObservationQuery

    s = _two_room_touch_spec()
    cid = _obj(s)["constraints"][0]["id"]
    s = adapters.ground(s, cid, "EC", rationale="pin")          # groundings + prov
    s = adapters.relate(s, "c", "d", ["PO"])                     # a delegated (no-prov path won't apply)
    spec = adapters.spec_from_json(s)
    # add an overlay atom, an invariant witness id, and a custom observation set
    spec = replace(
        spec,
        overlays=(OverlayAtom(id="ov0", members=frozenset({"a", "b"})),),
        invariant=replace(spec.invariant, witness_invariants=frozenset({"w:a-b:d1:c0"})),
        observation=ObservationQuery(kinds=frozenset({"rcc8", "contact_dimension"})),
        schema_version="9.9.9",
    )
    js1 = json.dumps(adapters.spec_to_json(spec), ensure_ascii=False)
    js2 = json.dumps(adapters.spec_to_json(adapters.spec_from_json(js1)), ensure_ascii=False)
    assert _obj(js1) == _obj(js2)
    d = _obj(js1)
    assert d["schema_version"] == "9.9.9"
    assert d["overlays"][0]["members"] == ["a", "b"]
    assert d["invariant"]["witness_invariants"] == ["w:a-b:d1:c0"]
    assert set(d["observation"]["kinds"]) == {"rcc8", "contact_dimension"}
    # grounding stamps provenance on the grounded *constraint* (the decision carries none)
    gc = next(c for c in d["constraints"] if c["id"] == cid)
    assert gc["status"] == "grounded" and gc["prov"]["origin"] == "ground"
    assert d["groundings"][0]["prov"] is None  # nullable prov round-trips as null


def test_realization_roundtrip_preserves_exact_fractions():
    s = _two_room_touch_spec()
    real_js = adapters.solve(s, bounds=[0, 20])
    r1 = adapters.realization_from_json(real_js)
    r2 = adapters.realization_from_json(
        json.dumps(adapters.realization_to_json(r1))
    )
    # exact box coordinates survive a full round trip (Fraction equality, not float)
    assert [ (b.region_id, b.lo, b.hi) for b in r1.boxes ] == \
           [ (b.region_id, b.lo, b.hi) for b in r2.boxes ]
    assert r1.status == r2.status and r1.satisfied == r2.satisfied


# ============================================================ relate
def test_relate_adds_constraint_with_mask_and_declares_regions():
    d = _obj(adapters.relate("{}", "living", "kitchen", ["EC", "PO"]))
    cons = d["constraints"]
    assert len(cons) == 1
    c = cons[0]
    assert c["src"] == "living" and c["dst"] == "kitchen"
    assert c["rcc8_mask"] == rcc8.mask("EC", "PO")
    assert set(c["rcc8_names"]) == {"EC", "PO"}
    assert c["status"] == "delegated"
    # regions were auto-declared
    assert {r["id"] for r in d["regions"]} == {"living", "kitchen"}


def test_relate_updates_existing_pair_in_place():
    s = adapters.relate("{}", "a", "b", ["EC"])
    s = adapters.relate(s, "a", "b", ["DC"], status="delegated")
    d = _obj(s)
    assert len(d["constraints"]) == 1, "same ordered pair must update, not duplicate"
    assert d["constraints"][0]["rcc8_mask"] == rcc8.mask("DC")


def test_relate_bad_relation_name_returns_error_json():
    out = adapters.relate("{}", "a", "b", ["NOPE"])
    assert _is_error(out)


def test_relate_refuses_to_rewrite_protected_constraint():
    # relate is a plain (delegated) edit; it must NOT silently rewrite a grounded or
    # invariant constraint (that would corrupt the edit-authority lattice).
    s = _two_room_touch_spec()
    cid = _obj(s)["constraints"][0]["id"]
    grounded = adapters.ground(s, cid, "EC")
    assert _is_error(adapters.relate(grounded, "a", "b", ["DC"]))
    inv = adapters.set_invariant(s, cid)
    assert _is_error(adapters.relate(inv, "a", "b", ["DC"]))


# ============================================================ witness
def test_witness_adds_incidence_witness_with_contact_dim():
    d = _obj(adapters.witness("{}", "a", "b", "line"))
    assert len(d["witnesses"]) == 1
    w = d["witnesses"][0]
    assert w["intended_contact_dimension"] == 1  # line -> LINE(1)
    assert set(w["incident_regions"]) == {"a", "b"}
    assert w["epistemic"] == "asserted"


def test_witness_all_contact_dims_map_correctly():
    for name, dim in (("face", 2), ("line", 1), ("edge", 1), ("point", 0)):
        d = _obj(adapters.witness("{}", "x", "y", name))
        assert d["witnesses"][0]["intended_contact_dimension"] == dim


def test_witness_bad_contact_dim_returns_error_json():
    assert _is_error(adapters.witness("{}", "a", "b", "volume"))


def test_witness_is_idempotent_on_repeat():
    s = adapters.witness("{}", "a", "b", "line")
    s2 = adapters.witness(s, "a", "b", "line")  # same deterministic id -> no duplicate
    d = _obj(s2)
    assert len(d["witnesses"]) == 1
    assert _obj(s) == d  # exact no-op, not a duplicated witness


def test_witness_different_dim_same_pair_is_kept_separately():
    # A different contact dimension yields a distinct deterministic id; both witnesses are
    # retained (the dim conflict is caught downstream by solve, not silently merged here).
    s = adapters.witness("{}", "a", "b", "line")
    s = adapters.witness(s, "a", "b", "point")
    assert len({w["id"] for w in _obj(s)["witnesses"]}) == 2
    # solve must flag the two-dimension pair as inconsistent
    s = adapters.relate(s, "a", "b", ["EC"])
    assert _obj(adapters.solve(s))["status"] == "inconsistent"


# ============================================================ ground / unground / invariant
def test_ground_flips_status_and_records_ledger():
    s = _two_room_touch_spec()
    cid = _obj(s)["constraints"][0]["id"]
    grounded = adapters.ground(s, cid, "EC", rationale="pin the touch")
    d = _obj(grounded)
    c = next(c for c in d["constraints"] if c["id"] == cid)
    assert c["status"] == "grounded"
    assert len(d["groundings"]) == 1
    assert d["groundings"][0]["target"] == cid


def test_unground_reverts_status_and_removes_decision():
    s = _two_room_touch_spec()
    cid = _obj(s)["constraints"][0]["id"]
    grounded = adapters.ground(s, cid, "EC")
    did = _obj(grounded)["groundings"][0]["id"]
    reverted = adapters.unground(grounded, did)
    d = _obj(reverted)
    c = next(c for c in d["constraints"] if c["id"] == cid)
    assert c["status"] == "delegated"
    assert d["groundings"] == []


def test_set_invariant_promotes_and_registers():
    s = _two_room_touch_spec()
    cid = _obj(s)["constraints"][0]["id"]
    inv = adapters.set_invariant(s, cid)
    d = _obj(inv)
    c = next(c for c in d["constraints"] if c["id"] == cid)
    assert c["status"] == "invariant"
    assert cid in d["invariant"]["relation_invariants"]


def test_ground_invariant_is_fail_closed():
    s = _two_room_touch_spec()
    cid = _obj(s)["constraints"][0]["id"]
    inv = adapters.set_invariant(s, cid)
    # grounding an INVARIANT constraint must fail closed to an error JSON
    assert _is_error(adapters.ground(inv, cid, "EC"))


def test_ground_unknown_target_returns_error_json():
    s = _two_room_touch_spec()
    assert _is_error(adapters.ground(s, "no-such-constraint", "EC"))


# ============================================================ solve / verify
def test_solve_realizes_and_verify_agrees():
    s = _two_room_touch_spec()
    real_js = adapters.solve(s, bounds=[0, 20])
    real = _obj(real_js)
    assert not _is_error(real_js)
    assert real["status"] == "realized_in_selected_domain_D"
    assert real["boxes"], "a realized spec must carry boxes"
    # boxes carry exact rational coordinates ({num,den}), never floats
    for b in real["boxes"]:
        for v in b["lo"] + b["hi"]:
            assert set(v.keys()) == {"num", "den"}

    # verify the realization against the spec: the EC+face witness must hold
    report = _obj(adapters.verify(s, real_js))
    assert report["violated"] == []
    assert report["witness_violations"] == []
    assert set(report["satisfied"]) == {c["id"] for c in _obj(s)["constraints"]}


def test_verify_catches_witness_dimension_mismatch():
    # Spec demands a LINE (edge) contact, but we hand-craft a realization that only
    # touches at a POINT (corner). verify must flag a contact_dim_mismatch violation.
    s = _two_room_touch_spec()
    realization = {
        "id": "hand",
        "status": "realized_in_selected_domain_D",
        "boxes": [
            {"region_id": "a",
             "lo": [frac_to_json(Fraction(0)), frac_to_json(Fraction(0))],
             "hi": [frac_to_json(Fraction(1)), frac_to_json(Fraction(1))]},
            {"region_id": "b",
             "lo": [frac_to_json(Fraction(1)), frac_to_json(Fraction(1))],
             "hi": [frac_to_json(Fraction(2)), frac_to_json(Fraction(2))]},
        ],
    }
    report = _obj(adapters.verify(s, realization))
    codes = {v["code"] for v in report["witness_violations"]}
    assert "contact_dim_mismatch" in codes


def test_solve_inconsistent_spec_reports_inconsistent():
    # a EC b  AND  a DC b on the same pair is symbolically contradictory
    s = adapters.relate("{}", "a", "b", ["EC"])
    # add a second, contradictory constraint by forcing a distinct id via a fresh pair name
    spec = adapters.spec_from_json(s)
    from dataclasses import replace
    from deixis.core.types import RelationConstraint
    spec = replace(
        spec,
        constraints=spec.constraints
        + (RelationConstraint(src="a", dst="b", rcc8_mask=rcc8.mask("DC"), id="a->b#2"),),
    )
    s2 = json.dumps(adapters.spec_to_json(spec))
    real = _obj(adapters.solve(s2))
    assert real["status"] == "inconsistent"


def test_solve_unknown_domain_returns_error_json():
    s = _two_room_touch_spec()
    assert _is_error(adapters.solve(s, domain="hyperbolic_7d"))


def test_solve_accepts_float_bounds_coerced_exactly():
    # non-integral float bounds must be accepted and coerced through exact Fraction
    # (shortest-decimal), not leaked as binary-float noise into the solver.
    s = _two_room_touch_spec()
    real = _obj(adapters.solve(s, bounds=[0.0, 12.5]))
    assert real["status"] == "realized_in_selected_domain_D"


# ============================================================ family
def test_family_returns_one_realization_per_scenario():
    s = _two_room_touch_spec()
    cid = _obj(s)["constraints"][0]["id"]
    scenarios = [
        [{"id": "g0", "target": cid, "fixed_value_or_domain": "EC"}],
        [],  # empty scenario: no extra grounding
    ]
    out = _obj(adapters.family(s, scenarios))
    assert isinstance(out, list) and len(out) == 2
    assert out[0]["scenario_id"] == "s0"
    assert out[1]["scenario_id"] == "s1"
    # family stamps a provenance ledger
    assert out[0]["prov"]["origin"] == "family"


# ============================================================ lift (observation)
def _lift_boxes_touching():
    # two unit squares sharing the full vertical edge x=1 => EC with LINE(1) contact
    return [
        {"region_id": "a", "lo": [0, 0], "hi": [1, 1]},
        {"region_id": "b", "lo": [1, 0], "hi": [2, 1]},
    ]


def test_lift_recovers_relation_as_delegated_observation():
    out = _obj(adapters.lift(_lift_boxes_touching(), ["a", "b"]))
    assert not ("error" in out)
    assert len(out["constraints"]) == 1
    c = out["constraints"][0]
    # geometry touches along an edge => EC
    assert c["rcc8_mask"] == rcc8.bit("EC")
    # lift proposes, never commits: status must be delegated, and NOTHING invariant
    assert c["status"] == "delegated"
    assert out["invariant"]["relation_invariants"] == []
    # provenance marks it as a tolerant observation
    assert c["prov"]["origin"] == "lift"
    assert "observed_tolerant" in c["prov"]["note"]


def test_lift_tags_contact_witness_observed_tolerant():
    out = _obj(adapters.lift(_lift_boxes_touching(), ["a", "b"]))
    assert len(out["witnesses"]) == 1
    w = out["witnesses"][0]
    assert w["intended_contact_dimension"] == 1  # shared edge => LINE(1)
    assert w["epistemic"] == Epistemic.OBSERVED_TOLERANT.value
    assert w["epistemic"] != Epistemic.DERIVED_EXACT.value  # not auto-promoted


def test_lift_accepts_float_coords_coerced_to_exact():
    boxes = [
        {"region_id": "a", "lo": [0.0, 0.0], "hi": [1.5, 1.0]},
        {"region_id": "b", "lo": [1.5, 0.0], "hi": [3.0, 1.0]},
    ]
    out = _obj(adapters.lift(boxes, ["a", "b"]))
    # floats coerced through exact Fraction; still an EC edge contact
    assert out["constraints"][0]["rcc8_mask"] == rcc8.bit("EC")
    assert out["witnesses"][0]["intended_contact_dimension"] == 1


def test_lift_tolerance_snaps_near_touch_to_ec():
    # two boxes with a tiny gap (0.02) that, under tolerance 0.1, snap to a real edge touch.
    boxes = [
        {"region_id": "a", "lo": [0.0, 0.0], "hi": [1.0, 1.0]},
        {"region_id": "b", "lo": [1.02, 0.0], "hi": [2.02, 1.0]},
    ]
    # exact (tolerance 0): the gap is real -> DC, no contact witness
    exact = _obj(adapters.lift(boxes, ["a", "b"], tolerance=0))
    assert exact["constraints"][0]["rcc8_mask"] == rcc8.bit("DC")
    assert exact["witnesses"] == []
    # tolerant (tolerance 0.1): endpoints snap to coincide -> EC edge contact + witness
    tol = _obj(adapters.lift(boxes, ["a", "b"], tolerance=0.1))
    assert tol["constraints"][0]["rcc8_mask"] == rcc8.bit("EC")
    assert len(tol["witnesses"]) == 1
    assert tol["witnesses"][0]["epistemic"] == Epistemic.OBSERVED_TOLERANT.value


def test_lift_tolerance_too_coarse_fails_closed():
    # a unit box under tolerance 5 would collapse to zero width -> error JSON, not a crash
    boxes = [{"region_id": "a", "lo": [0.0, 0.0], "hi": [1.0, 1.0]}]
    assert _is_error(adapters.lift(boxes, ["a"], tolerance=5))


def test_lift_disjoint_boxes_give_dc_and_no_witness():
    boxes = [
        {"region_id": "a", "lo": [0, 0], "hi": [1, 1]},
        {"region_id": "b", "lo": [5, 5], "hi": [6, 6]},
    ]
    out = _obj(adapters.lift(boxes, ["a", "b"]))
    assert out["constraints"][0]["rcc8_mask"] == rcc8.bit("DC")
    assert out["witnesses"] == []  # DC is not a contact -> no witness


def test_lift_then_solve_roundtrip():
    # lift an arrangement, then solve the observed spec: it must be realizable
    observed = adapters.lift(_lift_boxes_touching(), ["a", "b"])
    real = _obj(adapters.solve(observed, bounds=[0, 20]))
    assert real["status"] == "realized_in_selected_domain_D"


def test_lift_malformed_input_returns_error_json():
    # a box missing 'hi' must fail closed to an error JSON, not raise
    bad = [{"region_id": "a", "lo": [0, 0]}]
    assert _is_error(adapters.lift(bad, ["a"]))


# ============================================================ error discipline sweep
def test_all_adapters_return_json_strings():
    s = _two_room_touch_spec()
    assert isinstance(adapters.relate("{}", "a", "b", ["EC"]), str)
    assert isinstance(adapters.witness("{}", "a", "b", "face"), str)
    assert isinstance(adapters.solve(s), str)
    assert isinstance(adapters.verify(s, adapters.solve(s)), str)
    assert isinstance(adapters.family(s, [[]]), str)
    assert isinstance(adapters.lift(_lift_boxes_touching(), ["a", "b"]), str)
