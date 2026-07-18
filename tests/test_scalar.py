"""Tests for M-field (a) — scalar/membership fields and thresholding-as-grounding.

Central claims exercised (demo5):

  (1) Thresholding a field into a region *produces a GroundingDecision*
      (field -> room is a typed, reversible decision, not neutral bookkeeping).
  (2) The SAME field under different ``tau`` yields DIFFERENT room partitions
      -> the threshold is where the decision rights live.
  (3) ``argmax_ground`` gives a total (whole-grid) partition; a special reading.
  (4) ``region_to_indicator`` inverts a room exactly (round trip through tau==1).
  (5) An overlapping/continuous membership field is ``ungrounded`` (first-class),
      and a crisp one-hot field is not.
  (6) Everything is exact rational — no float leaks; float is rejected on build.
"""
from fractions import Fraction

import pytest

from deixis.core.types import GroundingDecision
from deixis.field.scalar import (
    MembershipField,
    ScalarField,
    argmax_ground,
    region_to_indicator,
    threshold_ground,
    ungrounded,
    demo5_same_field_different_rooms,
)

F = Fraction


# ------------------------------------------------------------------- ScalarField
def test_scalarfield_eval_row_major_exact():
    # 2x3 grid, values row-major
    vals = tuple(F(v) for v in (0, 1, 2, 3, 4, 5))
    fld = ScalarField(grid_shape=(2, 3), values=vals, origin=(F(0), F(0)), spacing=F(1, 2))
    assert fld.eval((0, 0)) == F(0)
    assert fld.eval((0, 2)) == F(2)
    assert fld.eval((1, 0)) == F(3)
    assert fld.eval((1, 2)) == F(5)
    for v in fld.values:
        assert isinstance(v, Fraction)


def test_scalarfield_cells_enumerates_full_grid_in_order():
    fld = ScalarField(grid_shape=(2, 3), values=tuple(F(i) for i in range(6)),
                      origin=(F(0), F(0)), spacing=F(1))
    cells = list(fld.cells())
    assert cells == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    # eval over cells() reproduces values in flat order
    assert [fld.eval(c) for c in cells] == list(fld.values)


def test_cell_center_relative_to_origin_and_spacing():
    fld = ScalarField(grid_shape=(3,), values=(F(0), F(0), F(0)),
                      origin=(F(10),), spacing=F(1, 4))
    assert fld.cell_center((0,)) == (F(10),)
    assert fld.cell_center((2,)) == (F(10) + F(1, 2),)


def test_scalarfield_bounds_and_rank_checks():
    fld = ScalarField(grid_shape=(2, 2), values=(F(0), F(1), F(2), F(3)),
                      origin=(F(0), F(0)), spacing=F(1))
    with pytest.raises(IndexError):
        fld.eval((2, 0))          # out of range
    with pytest.raises(IndexError):
        fld.eval((0,))            # wrong rank


def test_scalarfield_rejects_float_and_bad_shape():
    with pytest.raises(TypeError):
        ScalarField(grid_shape=(2,), values=(0.5, F(1)), origin=(F(0),), spacing=F(1))
    with pytest.raises(TypeError):
        ScalarField(grid_shape=(1,), values=(F(1),), origin=(F(0),), spacing=0.25)
    with pytest.raises(ValueError):
        ScalarField(grid_shape=(2, 2), values=(F(1),), origin=(F(0), F(0)), spacing=F(1))
    with pytest.raises(ValueError):
        ScalarField(grid_shape=(2,), values=(F(1), F(1)), origin=(F(0),), spacing=F(0))


# ------------------------------------------------------- claim (1): threshold=grounding
def test_threshold_produces_grounding_decision():
    fld = ScalarField(grid_shape=(4,), values=(F(1, 5), F(3, 5), F(1), F(2, 5)),
                      origin=(F(0),), spacing=F(1))
    cells, decision = threshold_ground(fld, F(3, 5), "warm")
    # the projection IS a typed GroundingDecision (field -> room)
    assert isinstance(decision, GroundingDecision)
    assert decision.target == "warm"
    assert decision.reversibility == "high"
    assert decision.prov is not None and decision.prov.origin == "threshold-ground"
    # L_tau = {cell | f >= 3/5} ; cell 0 (1/5) excluded, cells 1,2 included, 3 (2/5) excluded
    assert cells == frozenset({(1,), (2,)})


def test_threshold_uses_geq_boundary_exactly():
    fld = ScalarField(grid_shape=(3,), values=(F(1, 2), F(1), F(0)),
                      origin=(F(0),), spacing=F(1))
    cells, _ = threshold_ground(fld, F(1, 2), "r")
    # cell exactly at tau is INCLUDED (>=), exact rational boundary
    assert (0,) in cells and (1,) in cells and (2,) not in cells


def test_scalarfield_normalizes_list_to_tuple_no_leak():
    # pass mutable lists; field must snapshot them into tuples ("frozen for real")
    vals = [F(0), F(1)]
    shape = [2]
    fld = ScalarField(grid_shape=shape, values=vals, origin=[F(0)], spacing=F(1))
    assert isinstance(fld.values, tuple) and isinstance(fld.grid_shape, tuple)
    assert isinstance(fld.origin, tuple)
    vals.append(F(99))
    shape.append(7)
    assert fld.values == (F(0), F(1))   # unaffected by caller mutation
    assert fld.grid_shape == (2,)


def test_scalarfield_rejects_bool_shape_and_index():
    with pytest.raises(ValueError):
        ScalarField(grid_shape=(True,), values=(F(1),), origin=(F(0),), spacing=F(1))
    fld = ScalarField(grid_shape=(2,), values=(F(0), F(1)), origin=(F(0),), spacing=F(1))
    with pytest.raises(IndexError):
        fld.eval((True,))   # bool must not sneak in as an index


def test_threshold_decision_id_distinguishes_geometry():
    # same region/tau but different spacing -> distinct ledger ids (no collision)
    a = ScalarField(grid_shape=(2,), values=(F(1), F(1)), origin=(F(0),), spacing=F(1))
    b = ScalarField(grid_shape=(2,), values=(F(1), F(1)), origin=(F(0),), spacing=F(2))
    _, da = threshold_ground(a, F(1, 2), "r")
    _, db = threshold_ground(b, F(1, 2), "r")
    assert da.id != db.id


# ------------------------------------------- claim (2): same field, different rooms
def test_same_field_different_tau_different_rooms():
    fld = ScalarField(grid_shape=(5,),
                      values=(F(1, 5), F(3, 5), F(1), F(3, 5), F(1, 5)),
                      origin=(F(0),), spacing=F(1))
    room_low, dec_low = threshold_ground(fld, F(1, 2), "warm")
    room_high, dec_high = threshold_ground(fld, F(4, 5), "warm")
    assert room_low == frozenset({(1,), (2,), (3,)})
    assert room_high == frozenset({(2,)})
    # SAME field, DIFFERENT tau => DIFFERENT room partition
    assert room_low != room_high
    assert room_high < room_low  # strictly nested here, but genuinely different
    # distinct decisions in the ledger (distinct ids)
    assert dec_low.id != dec_high.id


def test_constant_field_same_room_across_tau_honest_limit():
    # honesty: different tau need NOT give different rooms (module says "can yield")
    const = ScalarField(grid_shape=(4,), values=(F(1), F(1), F(1), F(1)),
                        origin=(F(0),), spacing=F(1))
    r1, _ = threshold_ground(const, F(1, 2), "r")
    r2, _ = threshold_ground(const, F(3, 4), "r")
    assert r1 == r2 == frozenset({(0,), (1,), (2,), (3,)})


# ------------------------------------------------------- claim (3): argmax partition
def test_argmax_ground_total_partition():
    warm = ScalarField(grid_shape=(3,), values=(F(1, 5), F(3, 5), F(1)),
                       origin=(F(0),), spacing=F(1))
    cool = ScalarField(grid_shape=(3,), values=(F(4, 5), F(2, 5), F(0)),
                       origin=(F(0),), spacing=F(1))
    mf = MembershipField(region_ids=("warm", "cool"), fields=(warm, cool))
    part = argmax_ground(mf)
    # every cell assigned exactly once (rooms tile the whole grid)
    assert set(part.keys()) == {(0,), (1,), (2,)}
    assert part[(0,)] == "cool"   # 4/5 > 1/5
    assert part[(1,)] == "warm"   # 3/5 > 2/5
    assert part[(2,)] == "warm"   # 1 > 0


def test_argmax_tie_breaks_to_earlier_region_exactly():
    a = ScalarField(grid_shape=(2,), values=(F(1, 2), F(1)), origin=(F(0),), spacing=F(1))
    b = ScalarField(grid_shape=(2,), values=(F(1, 2), F(0)), origin=(F(0),), spacing=F(1))
    mf = MembershipField(region_ids=("a", "b"), fields=(a, b))
    part = argmax_ground(mf)
    assert part[(0,)] == "a"   # exact tie 1/2 == 1/2 -> earliest region wins
    assert part[(1,)] == "a"


# --------------------------------------------- claim (4): indicator round trip
def test_region_to_indicator_round_trip():
    fld = ScalarField(grid_shape=(2, 3),
                      values=tuple(F(v) for v in (0, 1, 1, 0, 1, 0)),
                      origin=(F(2), F(5)), spacing=F(1, 3))
    cells, _ = threshold_ground(fld, F(1), "r")
    ind = region_to_indicator(cells, fld.grid_shape, origin=fld.origin, spacing=fld.spacing)
    # indicator is 0/1 and preserves geometry
    assert set(ind.values) <= {F(0), F(1)}
    assert ind.origin == fld.origin and ind.spacing == fld.spacing
    # threshold the indicator back at tau==1 -> exactly the original room
    cells_back, _ = threshold_ground(ind, F(1), "r")
    assert cells_back == cells


def test_region_to_indicator_fail_closed_on_bad_cell():
    with pytest.raises(IndexError):
        region_to_indicator(frozenset({(5,)}), (3,))


# --------------------------------------------------- claim (5): ungrounded predicate
def test_ungrounded_true_for_overlapping_and_fractional():
    warm = ScalarField(grid_shape=(2,), values=(F(3, 5), F(1)), origin=(F(0),), spacing=F(1))
    cool = ScalarField(grid_shape=(2,), values=(F(4, 5), F(1)), origin=(F(0),), spacing=F(1))
    mf = MembershipField(region_ids=("warm", "cool"), fields=(warm, cool))
    # fractional membership and overlap (cell 1 is 1 in both) => still continuous
    assert ungrounded(mf) is True


def test_ungrounded_false_for_crisp_one_hot_partition():
    warm = ScalarField(grid_shape=(3,), values=(F(1), F(0), F(0)), origin=(F(0),), spacing=F(1))
    cool = ScalarField(grid_shape=(3,), values=(F(0), F(1), F(1)), origin=(F(0),), spacing=F(1))
    mf = MembershipField(region_ids=("warm", "cool"), fields=(warm, cool))
    # every cell is 1 in exactly one region -> already grounded, nothing to decide
    assert ungrounded(mf) is False


def test_ungrounded_true_when_a_cell_is_undecided():
    a = ScalarField(grid_shape=(2,), values=(F(1), F(0)), origin=(F(0),), spacing=F(1))
    b = ScalarField(grid_shape=(2,), values=(F(0), F(0)), origin=(F(0),), spacing=F(1))
    mf = MembershipField(region_ids=("a", "b"), fields=(a, b))
    # cell (1,) is 0 in both -> undecided -> ungrounded
    assert ungrounded(mf) is True


# --------------------------------------------------------- MembershipField guards
def test_membershipfield_requires_matching_geometry():
    a = ScalarField(grid_shape=(2,), values=(F(0), F(1)), origin=(F(0),), spacing=F(1))
    b = ScalarField(grid_shape=(3,), values=(F(0), F(1), F(0)), origin=(F(0),), spacing=F(1))
    with pytest.raises(ValueError):
        MembershipField(region_ids=("a", "b"), fields=(a, b))
    with pytest.raises(ValueError):
        MembershipField(region_ids=("a", "a"),
                        fields=(a, ScalarField((2,), (F(0), F(1)), (F(0),), F(1))))


# --------------------------------------------------------------------- demo5 driver
def test_demo5_artifacts():
    d = demo5_same_field_different_rooms()
    assert d["ungrounded"] is True
    assert d["same_field_different_rooms"] is True
    assert d["room_low"] != d["room_high"]
    assert isinstance(d["decision_low"], GroundingDecision)
    assert isinstance(d["decision_high"], GroundingDecision)
    # argmax is a total partition over the 5-cell grid
    assert set(d["argmax_partition"].keys()) == {(i,) for i in range(5)}
