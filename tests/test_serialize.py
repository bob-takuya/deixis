"""Round-trip / determinism tests for deixis.io.serialize.

Central guarantee: for every type T, ``T_from_json(T_to_json(x)) == x`` with Fractions
reconstructed bit-for-bit (not float-approximated). Plus: content_hash is stable for equal
specs, sensitive to any difference, and the codecs are robust to missing/unknown keys.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from deixis.core.ids import Provenance
from deixis.core.types import (
    Status, Epistemic, Realizability, ContactDim,
    Region, RelationConstraint, Cell, IncidenceWitness, OverlayAtom,
    GroundingDecision, SolverContract, Box, Realization, ObservationQuery,
    Invariant, RelSpec,
)
from deixis.io import serialize as S


# ---------------------------------------------------------------- fixtures

def _prov() -> Provenance:
    return Provenance(
        origin="asserted", actor="designer", activity="unit-test",
        inputs=("a", "b"), note="note-π",
    )


def _rich_spec() -> RelSpec:
    """A spec that exercises every field, every enum value, exact Fractions, frozensets."""
    r1 = Region(id="r1", semantic_type="living", declared_origin=True, prov=_prov())
    r2 = Region(id="r2", semantic_type="", declared_origin=False, prov=None)
    c1 = RelationConstraint(src="r1", dst="r2", rcc8_mask=0b0000_0110,
                            status=Status.INVARIANT, id="c1", prov=_prov())
    c2 = RelationConstraint(src="r2", dst="r1", rcc8_mask=0xFF,
                            status=Status.DELEGATED, id="c2", prov=None)
    w1 = IncidenceWitness(
        id="w1", cell_ids=("k0", "k1"), incident_regions=("r1", "r2"),
        intended_contact_dimension=ContactDim.LINE.value, component_index=2,
        boundary_role="shared_boundary", epistemic=Epistemic.OBSERVED_TOLERANT, prov=_prov(),
    )
    o1 = OverlayAtom(id="o1", members=frozenset({"r2", "r1", "r3"}), nonempty=True, prov=_prov())
    g1 = GroundingDecision(id="g1", target="c1", fixed_value_or_domain="x==21/5",
                           actor="designer", rationale="south wall", reversibility="low", prov=_prov())
    inv = Invariant(relation_invariants=frozenset({"c1"}),
                    witness_invariants=frozenset({"w1", "w0"}))
    obs = ObservationQuery(kinds=frozenset({"rcc8", "contact_dimension"}))
    return RelSpec(
        regions=(r1, r2), constraints=(c1, c2), witnesses=(w1,),
        overlays=(o1,), groundings=(g1,), invariant=inv, observation=obs,
        schema_version="0.0.1",
    )


def _realization() -> Realization:
    b1 = Box(region_id="r1", lo=(Fraction(0), Fraction(1, 3)),
             hi=(Fraction(21, 5), Fraction(7, 2)))
    b2 = Box(region_id="r2", lo=(Fraction(-3, 7),), hi=(Fraction(100),))
    return Realization(
        id="real1", boxes=(b1, b2), status=Realizability.REALIZED_IN_D,
        satisfied=("c1", "c2"), violated=(), undecided=("c3",),
        scenario_id="branch-2", prov=_prov(),
    )


# ---------------------------------------------------------------- per-type round trips

def test_provenance_roundtrip():
    p = _prov()
    assert S.provenance_from_json(S.provenance_to_json(p)) == p
    assert S.provenance_to_json(None) is None
    assert S.provenance_from_json(None) is None


def test_region_roundtrip():
    for r in (Region(id="r1", semantic_type="living", prov=_prov()),
              Region(id="r2")):
        assert S.region_from_json(S.region_to_json(r)) == r


def test_relation_constraint_roundtrip():
    for st in Status:
        c = RelationConstraint(src="a", dst="b", rcc8_mask=0x3C, status=st, id="c", prov=_prov())
        assert S.relation_constraint_from_json(S.relation_constraint_to_json(c)) == c


def test_cell_roundtrip():
    c = Cell(id="k7", dim=2)
    assert S.cell_from_json(S.cell_to_json(c)) == c


def test_witness_roundtrip():
    for ep in Epistemic:
        for cd in ContactDim:
            w = IncidenceWitness(
                id="w", cell_ids=("k0", "k1"), incident_regions=("r1", "r2", "r3"),
                intended_contact_dimension=cd.value, component_index=3,
                boundary_role="edge", epistemic=ep, prov=_prov(),
            )
            assert S.witness_from_json(S.witness_to_json(w)) == w


def test_witness_intended_contact_dimension_is_int_not_enum():
    w = IncidenceWitness(id="w", cell_ids=(), incident_regions=("r1", "r2"),
                         intended_contact_dimension=ContactDim.FACE.value)
    j = S.witness_to_json(w)
    assert j["intended_contact_dimension"] == 2
    assert isinstance(j["intended_contact_dimension"], int)
    assert S.witness_from_json(j) == w


def test_overlay_roundtrip_and_sorted_members():
    o = OverlayAtom(id="o", members=frozenset({"z", "a", "m"}), nonempty=False, prov=_prov())
    j = S.overlay_to_json(o)
    assert j["members"] == ["a", "m", "z"]          # deterministic sorted list
    assert S.overlay_from_json(j) == o


def test_grounding_roundtrip():
    g = _rich_spec().groundings[0]
    assert S.grounding_from_json(S.grounding_to_json(g)) == g


def test_solver_contract_roundtrip():
    s = SolverContract(domain="aabb_3d", solver="cp-sat",
                       objectives=("min_area", "compact"), completeness_claim="x", timeout_ms=500)
    assert S.solver_contract_from_json(S.solver_contract_to_json(s)) == s
    assert S.solver_contract_from_json(S.solver_contract_to_json(SolverContract())) == SolverContract()


def test_invariant_roundtrip_and_sorted():
    inv = Invariant(relation_invariants=frozenset({"c2", "c1"}),
                    witness_invariants=frozenset({"w3", "w1", "w2"}))
    j = S.invariant_to_json(inv)
    assert j["relation_invariants"] == ["c1", "c2"]
    assert j["witness_invariants"] == ["w1", "w2", "w3"]
    assert S.invariant_from_json(j) == inv


def test_observation_roundtrip_default_and_custom():
    assert S.observation_from_json(S.observation_to_json(ObservationQuery())) == ObservationQuery()
    o = ObservationQuery(kinds=frozenset({"rcc8", "atom_occupancy"}))
    assert S.observation_from_json(S.observation_to_json(o)) == o


def test_box_roundtrip_exact_fraction():
    b = Box(region_id="r", lo=(Fraction(1, 3), Fraction(-7, 11)),
            hi=(Fraction(22, 7), Fraction(0)))
    b2 = S.box_from_json(S.box_to_json(b))
    assert b2 == b
    # bit-for-bit fraction identity (not float)
    assert b2.lo[0] == Fraction(1, 3)
    assert b2.lo[1].denominator == 11


def test_realization_roundtrip():
    for st in Realizability:
        real = Realization(id="R", boxes=_realization().boxes, status=st,
                           satisfied=("c1",), violated=("c2",), undecided=(),
                           scenario_id="s", prov=_prov())
        assert S.realization_from_json(S.realization_to_json(real)) == real


def test_relspec_roundtrip_full():
    spec = _rich_spec()
    spec2 = S.relspec_from_json(S.relspec_to_json(spec))
    assert spec2 == spec


def test_relspec_empty_roundtrip():
    spec = RelSpec()
    assert S.relspec_from_json(S.relspec_to_json(spec)) == spec


# ---------------------------------------------------------------- Fraction exactness

def test_fraction_never_becomes_float():
    b = Box(region_id="r", lo=(Fraction(1, 3),), hi=(Fraction(2, 3),))
    j = S.box_to_json(b)
    assert j["lo"][0] == {"num": 1, "den": 3}
    assert isinstance(j["lo"][0]["num"], int) and isinstance(j["lo"][0]["den"], int)
    # value that no binary float can represent exactly still survives
    tricky = Fraction(1, 3) + Fraction(1, 7)   # 10/21
    b2 = Box(region_id="r", lo=(tricky,), hi=(Fraction(0),))
    assert S.box_from_json(S.box_to_json(b2)).lo[0] == Fraction(10, 21)


def test_float_coordinate_on_wire_is_rejected():
    # The exact {num,den} contract: a bare float must never be silently reconstructed.
    with pytest.raises(TypeError):
        S.box_from_json({"region_id": "r", "lo": [0.1], "hi": []})
    with pytest.raises(TypeError):
        S.realization_from_json({"id": "R", "boxes": [{"region_id": "r", "lo": [1.5], "hi": []}]})


def test_no_float_anywhere_in_serialized_realization():
    j = S.realization_to_json(_realization())

    def _walk(x):
        assert not isinstance(x, float), f"float leaked: {x!r}"
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)

    _walk(j)


# ---------------------------------------------------------------- dumps / loads

def test_dumps_deterministic_and_sorted():
    spec = _rich_spec()
    s1 = S.dumps(S.relspec_to_json(spec))
    s2 = S.dumps(S.relspec_to_json(spec))
    assert s1 == s2
    # sort_keys => top-level keys ascending
    d = S.loads(s1)
    assert list(d.keys()) == sorted(d.keys())


def test_dumps_accepts_typed_objects():
    spec = _rich_spec()
    assert S.dumps(spec) == S.dumps(S.relspec_to_json(spec))
    real = _realization()
    assert S.dumps(real) == S.dumps(S.realization_to_json(real))


def test_dumps_loads_roundtrip_via_string():
    spec = _rich_spec()
    spec2 = S.relspec_from_json(S.loads(S.dumps(spec)))
    assert spec2 == spec


def test_dumps_key_order_independent():
    # Same logical content, different insertion order of a dict -> identical canonical string.
    d1 = {"b": 1, "a": {"y": 2, "x": 3}}
    d2 = {"a": {"x": 3, "y": 2}, "b": 1}
    assert S.dumps(d1) == S.dumps(d2)


# ---------------------------------------------------------------- content_hash

def test_content_hash_stable_for_equal_specs():
    assert S.content_hash(_rich_spec()) == S.content_hash(_rich_spec())


def test_content_hash_is_hex_sha256():
    h = S.content_hash(_rich_spec())
    assert isinstance(h, str) and len(h) == 64
    int(h, 16)  # parseable as hex


def test_content_hash_changes_on_tiny_fraction_difference():
    spec = _rich_spec()
    h0 = S.content_hash(spec)
    # perturb a mask by one bit
    c = spec.constraints[0]
    from dataclasses import replace
    spec_b = spec.with_constraints((replace(c, rcc8_mask=c.rcc8_mask ^ 1),) + spec.constraints[1:])
    assert S.content_hash(spec_b) != h0


def test_content_hash_distinguishes_exact_fractions_in_grounding():
    spec = _rich_spec()
    from dataclasses import replace
    g = spec.groundings[0]
    spec_b = spec.with_groundings((replace(g, fixed_value_or_domain="x==21/6"),))
    assert S.content_hash(spec_b) != S.content_hash(spec)


def test_content_hash_insensitive_to_frozenset_order():
    # Two logically-equal specs whose frozensets were built in different orders hash equal.
    a = RelSpec(overlays=(OverlayAtom(id="o", members=frozenset(["a", "b", "c"])),),
                invariant=Invariant(witness_invariants=frozenset(["w2", "w1"])))
    b = RelSpec(overlays=(OverlayAtom(id="o", members=frozenset(["c", "a", "b"])),),
                invariant=Invariant(witness_invariants=frozenset(["w1", "w2"])))
    assert S.content_hash(a) == S.content_hash(b)


# ---------------------------------------------------------------- robustness

def test_missing_optional_keys_use_defaults():
    # Minimal region dict (only required id) -> defaults fill in.
    r = S.region_from_json({"id": "r9"})
    assert r == Region(id="r9")
    # Minimal constraint
    c = S.relation_constraint_from_json({"src": "a", "dst": "b", "rcc8_mask": 3})
    assert c == RelationConstraint(src="a", dst="b", rcc8_mask=3)
    # Minimal relspec
    assert S.relspec_from_json({}) == RelSpec()
    # Minimal realization
    assert S.realization_from_json({"id": "R"}) == Realization(id="R")


def test_unknown_keys_are_ignored():
    d = S.region_to_json(Region(id="r1", semantic_type="living"))
    d["FUTURE_FIELD"] = {"anything": [1, 2, 3]}
    assert S.region_from_json(d) == Region(id="r1", semantic_type="living")
    # full-spec version
    sd = S.relspec_to_json(_rich_spec())
    sd["EXTRA_TOP_LEVEL"] = 42
    sd["regions"][0]["EXTRA"] = "x"
    assert S.relspec_from_json(sd) == _rich_spec()


def test_null_prov_and_absent_prov_equivalent():
    d = S.region_to_json(Region(id="r"))
    assert d["prov"] is None
    d.pop("prov")
    assert S.region_from_json(d) == Region(id="r")


def test_frac_json_shape_is_num_den():
    j = S.box_to_json(Box(region_id="r", lo=(Fraction(3, 4),), hi=(Fraction(5),)))
    assert set(j["lo"][0].keys()) == {"num", "den"}
    assert j["hi"][0] == {"num": 5, "den": 1}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
