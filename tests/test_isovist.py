"""Tests for M-field (c) — the visibility (isovist) field.

Claims exercised:
  (1) Open space is highly visible; being near/behind an obstacle lowers the
      visible area.  ("開空間で高・障害物近傍で低")
  (2) Each metric ('area'|'occlusivity'|'perimeter') behaves sensibly and
      differently.  ("metric別")
  (3) The result is a ScalarField over the grid (values, geometry, exactness).
  (4) It is *not* a DEC cochain: it carries no coboundary structure and is not
      wired into the flow (DEC) operators.  ("DECを使わない")
  (5) With no obstacles the visible area is maximal (nothing occludes).
      ("空間全開で最大")
  (6) Everything is exact rational — no float leaks. Occupancy (which ray is
      blocked, and where) is exact; area is exact for the sampled rays.
"""
from fractions import Fraction

import pytest

from deixis.core.types import Box
from deixis.field.scalar import ScalarField
from deixis.field import isovist as iso
from deixis.field.isovist import isovist_field, isovist_at_point, square_directions

F = Fraction


# --------------------------------------------------------------------- geometry helpers
def _open_field(metric="area", n_dirs=16, n=5):
    """A 5x5 empty room (no obstacles)."""
    return isovist_field(
        None, [], (n, n), (F(0), F(0)), F(1), metric=metric, n_dirs=n_dirs
    )


# --------------------------------------------------------------------- (3) ScalarField
def test_returns_scalarfield_over_grid():
    fld = _open_field()
    assert isinstance(fld, ScalarField)
    assert fld.grid_shape == (5, 5)
    assert len(fld.values) == 25
    assert fld.origin == (F(0), F(0))
    assert fld.spacing == F(1)


def test_all_values_are_exact_fractions_no_float():
    fld = _open_field(metric="perimeter")
    for v in fld.values:
        assert isinstance(v, Fraction)
        assert not isinstance(v, float)


def test_stored_and_evaluable_per_cell():
    fld = _open_field()
    # cell center coincides with origin + index*spacing
    assert fld.cell_center((2, 2)) == (F(2), F(2))
    assert fld.eval((2, 2)) > 0


# --------------------------------------------------------------------- (1) open high / obstacle low
def test_obstacle_lowers_visible_area_vs_open():
    n_dirs = 32
    open_fld = _open_field(metric="area", n_dirs=n_dirs)
    # A solid box occupying the lower-left quadrant.
    obstacle = Box(region_id="wall", lo=(F(0), F(0)), hi=(F(1), F(1)))
    blocked = isovist_field(
        None, [obstacle], (5, 5), (F(0), F(0)), F(1), metric="area", n_dirs=n_dirs
    )
    # A cell just outside the obstacle sees less than in the empty room.
    cell = (3, 3)
    assert blocked.eval(cell) < open_fld.eval(cell)


def test_area_never_exceeds_open_case():
    """(5) With no obstacles the area is maximal: adding a wall can only reduce it."""
    n_dirs = 24
    open_fld = _open_field(metric="area", n_dirs=n_dirs)
    obstacle = Box(region_id="wall", lo=(F(1), F(1)), hi=(F(3), F(3)))
    blocked = isovist_field(
        None, [obstacle], (5, 5), (F(0), F(0)), F(1), metric="area", n_dirs=n_dirs
    )
    for c in open_fld.cells():
        assert blocked.eval(c) <= open_fld.eval(c)
    # and strictly less somewhere (the wall does occlude at least one vantage)
    assert any(blocked.eval(c) < open_fld.eval(c) for c in open_fld.cells())


def test_point_inside_obstacle_has_zero_area():
    obstacle = Box(region_id="wall", lo=(F(1), F(1)), hi=(F(3), F(3)))
    fld = isovist_field(
        None, [obstacle], (5, 5), (F(0), F(0)), F(1), metric="area", n_dirs=16
    )
    # cell (2,2) center (2,2) is strictly inside the wall -> nothing visible.
    assert fld.eval((2, 2)) == F(0)


# --------------------------------------------------------------------- (2) metric-by-metric
def test_occlusivity_zero_in_open_space():
    fld = _open_field(metric="occlusivity", n_dirs=16)
    for v in fld.values:
        assert v == F(0)


def test_occlusivity_positive_and_higher_near_obstacle():
    n_dirs = 32
    obstacle = Box(region_id="wall", lo=(F(1), F(1)), hi=(F(2), F(2)))
    fld = isovist_field(
        None, [obstacle], (7, 7), (F(0), F(0)), F(1), metric="occlusivity", n_dirs=n_dirs
    )
    near = fld.eval((3, 3))   # adjacent to the wall
    far = fld.eval((6, 6))    # opposite corner, small solid-angle subtended
    assert near > F(0)
    assert near > far
    # occlusivity is a proper fraction of the ray count
    assert F(0) <= near <= F(1)
    assert (near * n_dirs).denominator == 1


def test_occlusivity_one_inside_obstacle():
    obstacle = Box(region_id="wall", lo=(F(1), F(1)), hi=(F(3), F(3)))
    fld = isovist_field(
        None, [obstacle], (5, 5), (F(0), F(0)), F(1), metric="occlusivity", n_dirs=16
    )
    assert fld.eval((2, 2)) == F(1)


def test_perimeter_positive_and_reduced_by_obstacle():
    n_dirs = 32
    open_fld = _open_field(metric="perimeter", n_dirs=n_dirs)
    assert all(v > 0 for v in open_fld.values)
    obstacle = Box(region_id="wall", lo=(F(0), F(0)), hi=(F(1), F(1)))
    blocked = isovist_field(
        None, [obstacle], (5, 5), (F(0), F(0)), F(1), metric="perimeter", n_dirs=n_dirs
    )
    assert blocked.eval((3, 3)) < open_fld.eval((3, 3))


def test_metrics_are_distinct_quantities():
    n_dirs = 24
    obstacle = Box(region_id="wall", lo=(F(1), F(1)), hi=(F(2), F(2)))
    args = ([obstacle], (5, 5), (F(0), F(0)), F(1))
    a = isovist_field(None, *args, metric="area", n_dirs=n_dirs)
    o = isovist_field(None, *args, metric="occlusivity", n_dirs=n_dirs)
    p = isovist_field(None, *args, metric="perimeter", n_dirs=n_dirs)
    c = (3, 3)
    vals = {a.eval(c), o.eval(c), p.eval(c)}
    assert len(vals) == 3  # three genuinely different scalars


# --------------------------------------------------------------------- (4) not a DEC cochain
def test_not_a_dec_cochain_no_coboundary_link():
    """isovist is a bare ScalarField: it has none of the DEC cochain machinery,
    and the flow (DEC) operators do not accept it."""
    fld = _open_field()
    # It is a ScalarField, and carries no cochain/coboundary attributes.
    assert isinstance(fld, ScalarField)
    for attr in ("coboundary", "d0", "d1", "divergence", "edges", "cochain_degree"):
        assert not hasattr(fld, attr)
    # The flow module's DEC operators are not part of the isovist API.
    for name in ("coboundary_d0", "coboundary_d1", "divergence"):
        assert not hasattr(iso, name)


# --------------------------------------------------------------------- directions / exactness
def test_square_directions_are_rational_ordered_and_nonzero():
    dirs = square_directions(8)
    assert len(dirs) == 8
    for dx, dy in dirs:
        assert isinstance(dx, Fraction) and isinstance(dy, Fraction)
        assert not (dx == 0 and dy == 0)
    # 8 dirs = 4 axes + 4 diagonals on the Chebyshev square
    assert (F(1), F(0)) in dirs
    assert (F(0), F(1)) in dirs
    assert (F(1), F(1)) in dirs


def test_square_directions_rejects_too_few():
    with pytest.raises(ValueError):
        square_directions(2)


def test_symmetry_of_open_field():
    """In an empty square room the area field is symmetric about the center."""
    fld = _open_field(metric="area", n_dirs=16, n=5)
    # opposite corners see the same area by symmetry
    assert fld.eval((0, 0)) == fld.eval((4, 4))
    assert fld.eval((0, 4)) == fld.eval((4, 0))
    # center is a strict local maximum among these samples
    assert fld.eval((2, 2)) >= fld.eval((0, 0))


# --------------------------------------------------------------------- sample_points subset
def test_sample_points_evaluates_only_listed_cells():
    obstacle = Box(region_id="wall", lo=(F(1), F(1)), hi=(F(2), F(2)))
    # only probe two cell centers; the rest must be 0
    fld = isovist_field(
        [(F(3), F(3)), (F(4), F(4))],
        [obstacle], (5, 5), (F(0), F(0)), F(1), metric="area", n_dirs=16,
    )
    assert fld.eval((3, 3)) > 0
    assert fld.eval((4, 4)) > 0
    assert fld.eval((0, 0)) == F(0)
    assert fld.eval((2, 2)) == F(0)
    # exactly two non-zero cells
    assert sum(1 for v in fld.values if v != 0) == 2


# --------------------------------------------------------------------- validation
def test_rejects_3d_grid():
    with pytest.raises(ValueError):
        isovist_field(None, [], (3, 3, 3), (F(0), F(0), F(0)), F(1))


def test_rejects_3d_obstacle():
    b3 = Box(region_id="w", lo=(F(0), F(0), F(0)), hi=(F(1), F(1), F(1)))
    with pytest.raises(ValueError):
        isovist_field(None, [b3], (3, 3), (F(0), F(0)), F(1))


def test_rejects_unknown_metric():
    with pytest.raises(ValueError):
        isovist_field(None, [], (3, 3), (F(0), F(0)), F(1), metric="nope")


def test_rejects_inverted_obstacle():
    bad = Box(region_id="w", lo=(F(2), F(2)), hi=(F(1), F(1)))
    with pytest.raises(ValueError):
        isovist_field(None, [bad], (3, 3), (F(0), F(0)), F(1))


def test_rejects_bool_grid_shape():
    # bool is an int subclass; ScalarField rejects it, and so must we (pre-coerce).
    with pytest.raises(ValueError):
        isovist_field(None, [], (True, 3), (F(0), F(0)), F(1))


def test_rejects_nonpositive_grid_shape():
    with pytest.raises(ValueError):
        isovist_field(None, [], (0, 3), (F(0), F(0)), F(1))


def test_isovist_at_point_validates_metric_and_directions():
    dirs = square_directions(8)
    with pytest.raises(ValueError):
        isovist_at_point((F(0), F(0)), [], (F(-1), F(-1)), (F(1), F(1)), dirs, "nope")
    with pytest.raises(ValueError):
        isovist_at_point((F(0), F(0)), [], (F(-1), F(-1)), (F(1), F(1)), [], "area")


def test_sample_point_maps_to_nearest_cell_center():
    # a point at (2.8, 3.1) is nearest to cell center (3,3), not floor (2,3).
    fld = isovist_field(
        [(F(28, 10), F(31, 10))],
        [], (5, 5), (F(0), F(0)), F(1), metric="area", n_dirs=8,
    )
    assert fld.eval((3, 3)) > 0
    assert sum(1 for v in fld.values if v != 0) == 1


def test_isovist_at_point_matches_field_cell():
    obstacle = Box(region_id="wall", lo=(F(1), F(1)), hi=(F(2), F(2)))
    dirs = square_directions(16)
    direct = isovist_at_point(
        (F(3), F(3)), [obstacle], (F(-1, 2), F(-1, 2)), (F(9, 2), F(9, 2)), dirs, "area"
    )
    fld = isovist_field(
        None, [obstacle], (5, 5), (F(0), F(0)), F(1), metric="area", n_dirs=16
    )
    assert direct == fld.eval((3, 3))
