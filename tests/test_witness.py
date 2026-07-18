"""Tests for incidence (contact) witnesses (deixis.incidence.witness).

Core demo1 property: face / line / point contact witnesses all collapse to the
single RCC-8 relation EC, yet remain distinguishable as witnesses. Plus the
local-consistency checks (RCC compatibility, dim(f)==intended_contact_dimension,
structural well-formedness). All exact integer / structural work; no floats."""
from deixis.core import rcc8
from deixis.core.types import (
    Cell,
    ContactDim,
    Epistemic,
    Region,
    RelationConstraint,
    RelSpec,
    Status,
)
from deixis.incidence import witness as W
from deixis.incidence.witness import (
    Violation,
    check_local_consistency,
    check_witness,
    collapses_to_same_rcc,
    make_witness,
    witness_rcc_mask,
    witnesses_distinguishable,
)


# ---------------------------------------------------------------- make_witness
def test_make_witness_normalises_fields():
    w = make_witness(("A", "B"), ContactDim.FACE, cell_ids=["f1"])
    assert w.incident_regions == ("A", "B")
    assert w.cell_ids == ("f1",)
    assert w.intended_contact_dimension == 2
    assert isinstance(w.intended_contact_dimension, int)
    assert w.boundary_role == "shared_boundary"
    assert w.epistemic is Epistemic.ASSERTED


def test_make_witness_accepts_raw_int_dim():
    w = make_witness(("A", "B"), 1)
    assert w.intended_contact_dimension == 1


def test_make_witness_deterministic_id():
    w1 = make_witness(("A", "B"), ContactDim.POINT)
    w2 = make_witness(("A", "B"), ContactDim.POINT)
    assert w1.id == w2.id
    # dimension / component change the derived id
    assert make_witness(("A", "B"), ContactDim.LINE).id != w1.id
    assert make_witness(("A", "B"), ContactDim.POINT, component_index=1).id != w1.id


def test_make_witness_explicit_id_wins():
    w = make_witness(("A", "B"), ContactDim.FACE, id="my-witness")
    assert w.id == "my-witness"


def test_make_witness_rejects_float_dim():
    import pytest
    with pytest.raises(TypeError):
        make_witness(("A", "B"), 1.0)


# ---------------------------------------------------------------- rcc entailment
def test_every_contact_dim_entails_EC():
    for d in ContactDim:
        w = make_witness(("A", "B"), d)
        assert witness_rcc_mask(w) == rcc8.mask("EC")
        assert rcc8.names(witness_rcc_mask(w)) == ["EC"]


# ---------------------------------------------------------------- demo1 core
def _demo_triple():
    w_face = make_witness(("A", "B"), ContactDim.FACE, cell_ids=["f0"])
    w_edge = make_witness(("A", "B"), ContactDim.LINE, cell_ids=["e0"])
    w_point = make_witness(("A", "B"), ContactDim.POINT, cell_ids=["v0"])
    return w_face, w_edge, w_point


def test_demo1_collapse_to_same_rcc():
    w_face, w_edge, w_point = _demo_triple()
    assert collapses_to_same_rcc(w_face, w_edge, w_point) is True
    # they all collapse onto EC specifically
    assert {witness_rcc_mask(w) for w in (w_face, w_edge, w_point)} == {rcc8.mask("EC")}


def test_demo1_witnesses_remain_distinguishable():
    w_face, w_edge, w_point = _demo_triple()
    assert witnesses_distinguishable(w_face, w_edge, w_point) is True
    # the collapse is lossy in RCC-8 but NOT in the witness layer
    assert collapses_to_same_rcc(w_face, w_edge, w_point)
    assert witnesses_distinguishable(w_face, w_edge, w_point)


def test_distinguishable_needs_two():
    w_face, _, _ = _demo_triple()
    assert witnesses_distinguishable(w_face) is False


def test_identical_witnesses_not_distinguishable():
    w1 = make_witness(("A", "B"), ContactDim.FACE, cell_ids=["f0"])
    w2 = make_witness(("A", "B"), ContactDim.FACE, cell_ids=["f0"])
    assert collapses_to_same_rcc(w1, w2) is True
    assert witnesses_distinguishable(w1, w2) is False


def test_collapse_empty_and_single():
    assert collapses_to_same_rcc() is True
    w_face, _, _ = _demo_triple()
    assert collapses_to_same_rcc(w_face) is True


# ---------------------------------------------------------------- consistency: RCC
def _spec(witnesses, constraints=(), regions=("A", "B")):
    return RelSpec(
        regions=tuple(Region(id=r) for r in regions),
        constraints=tuple(constraints),
        witnesses=tuple(witnesses),
    )


def test_consistency_ok_with_EC_constraint():
    w = make_witness(("A", "B"), ContactDim.FACE)
    c = RelationConstraint("A", "B", rcc8.mask("EC"), Status.INVARIANT, id="c0")
    assert check_local_consistency(_spec([w], [c])) == []


def test_consistency_ok_when_EC_in_disjunction():
    w = make_witness(("A", "B"), ContactDim.LINE)
    # constraint allows DC or EC — EC still possible, so compatible
    c = RelationConstraint("A", "B", rcc8.mask("DC", "EC"), id="c0")
    assert check_local_consistency(_spec([w], [c])) == []


def test_consistency_flags_rcc_incompatible():
    w = make_witness(("A", "B"), ContactDim.FACE)
    # constraint says DC only — contact (EC) is impossible
    c = RelationConstraint("A", "B", rcc8.mask("DC"), id="c0")
    vs = check_local_consistency(_spec([w], [c]))
    assert len(vs) == 1
    assert vs[0].code == "rcc_incompatible"
    assert set(vs[0].regions) == {"A", "B"}


def test_rcc_check_direction_independent():
    w = make_witness(("A", "B"), ContactDim.FACE)
    # constraint stored in converse order (B,A) with DC only
    c = RelationConstraint("B", "A", rcc8.mask("DC"), id="c0")
    vs = check_local_consistency(_spec([w], [c]))
    assert [v.code for v in vs] == ["rcc_incompatible"]


def test_no_constraint_no_rcc_violation():
    w = make_witness(("A", "B"), ContactDim.FACE)
    assert check_local_consistency(_spec([w])) == []


# ---------------------------------------------------------------- consistency: structure
def test_too_few_regions_flagged():
    w = make_witness(("A",), ContactDim.FACE)
    vs = check_local_consistency(_spec([w], regions=("A", "B")))
    assert any(v.code == "too_few_regions" for v in vs)


def test_bad_contact_dim_flagged():
    w = make_witness(("A", "B"), 3)  # 3 is not a valid ContactDim
    vs = check_local_consistency(_spec([w]))
    assert any(v.code == "bad_contact_dim" for v in vs)


def test_self_contact_duplicate_region_flagged():
    # EC is irreflexive: a witness joining A to itself is ill-formed
    w = make_witness(("A", "A"), ContactDim.FACE)
    vs = check_local_consistency(_spec([w], regions=("A", "B")))
    codes = {v.code for v in vs}
    assert "duplicate_region" in codes
    assert "too_few_regions" not in codes


def test_unknown_region_flagged():
    w = make_witness(("A", "Z"), ContactDim.FACE)
    vs = check_local_consistency(_spec([w], regions=("A", "B")))
    codes = {v.code for v in vs}
    assert "unknown_region" in codes


# ---------------------------------------------------------------- consistency: dim(f)
def test_dim_match_face_line_point():
    cells = [Cell("f0", 2), Cell("e0", 1), Cell("v0", 0)]
    w_face = make_witness(("A", "B"), ContactDim.FACE, cell_ids=["f0", "e0", "v0"])
    w_edge = make_witness(("A", "B"), ContactDim.LINE, cell_ids=["e0", "v0"])
    w_point = make_witness(("A", "B"), ContactDim.POINT, cell_ids=["v0"])
    spec = _spec([w_face, w_edge, w_point])
    assert check_local_consistency(spec, cells=cells) == []


def test_dim_mismatch_flagged():
    # claims FACE (2) but the sub-complex tops out at an edge (dim 1)
    cells = {"e0": 1, "v0": 0}
    w = make_witness(("A", "B"), ContactDim.FACE, cell_ids=["e0", "v0"])
    vs = check_local_consistency(_spec([w]), cells=cells)
    assert [v.code for v in vs] == ["dim_mismatch"]


def test_dim_check_skipped_without_cells():
    w = make_witness(("A", "B"), ContactDim.FACE, cell_ids=["e0"])
    # no cells arg -> dim check does not run
    assert check_local_consistency(_spec([w])) == []


def test_unknown_cell_flagged():
    w = make_witness(("A", "B"), ContactDim.FACE, cell_ids=["ghost"])
    vs = check_witness(w, cell_dims={"f0": 2})
    assert any(v.code == "unknown_cell" for v in vs)


def test_cells_as_mapping_and_as_cell_iterable_agree():
    w = make_witness(("A", "B"), ContactDim.LINE, cell_ids=["e0", "v0"])
    spec = _spec([w])
    via_map = check_local_consistency(spec, cells={"e0": 1, "v0": 0})
    via_cells = check_local_consistency(spec, cells=[Cell("e0", 1), Cell("v0", 0)])
    assert via_map == via_cells == []


# ---------------------------------------------------------------- Violation type
def test_violation_is_frozen_and_stringable():
    v = Violation("w0", "rcc_incompatible", "boom", ("A", "B"))
    assert "w0" in str(v) and "rcc_incompatible" in str(v)
    import dataclasses
    assert dataclasses.is_dataclass(v)
