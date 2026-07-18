"""Tests for deixis.geom.group_view: transformation-group *views* (read-only
projection onto group invariants) plus honest LossReport disclosure.

Covered:
  * the TransformGroup poset (leq / meet / join / chain);
  * each group observes exactly its invariants (Euclidean coords/distances,
    Similarity ratios only, Affine parallelism/ordinal, Projective incidence/
    collinearity/cross-ratio, Homeo contact only);
  * LossReport discloses the dropped categories (absolute coords / metric /
    orientation / linking) and grows monotonically along the chain;
  * projection is READ-ONLY (realization unchanged; forgetting != reverse-grounding,
    so there is no inverse) and invariants are genuinely group-invariant;
  * canonical_key is a hashable X/G orbit coordinate (invariant under the action).
"""
from __future__ import annotations

import copy
from fractions import Fraction as Fr

import pytest

from deixis.core.types import Realization, Box, Realizability
from deixis.geom import (
    TransformGroup,
    LossReport,
    project,
    canonical_key,
    groups_chain,
)
from deixis.geom.group_view import _cross_ratio


def _box(rid, lo, hi):
    return Box(rid, tuple(Fr(v) for v in lo), tuple(Fr(v) for v in hi))


def _real(*boxes):
    return Realization(id="r", boxes=tuple(boxes), status=Realizability.REALIZED_IN_D)


@pytest.fixture
def two_touch():
    # A [0,2]^2 ; B touches A's right edge => EC (face/edge contact in 2D).
    return _real(_box("A", (0, 0), (2, 2)), _box("B", (2, 0), (3, 1)))


# ------------------------------------------------------------------ the poset

def test_chain_is_totally_ordered_ascending():
    chain = groups_chain()
    assert chain == (
        TransformGroup.EUCLIDEAN,
        TransformGroup.SIMILARITY,
        TransformGroup.AFFINE,
        TransformGroup.PROJECTIVE,
        TransformGroup.HOMEO,
    )
    ranks = [g.rank for g in chain]
    assert ranks == sorted(ranks) == [0, 1, 2, 3, 4]


def test_leq_reflexive_and_inclusion():
    for g in groups_chain():
        assert g.leq(g)  # reflexive
    assert TransformGroup.EUCLIDEAN.leq(TransformGroup.HOMEO)
    assert not TransformGroup.HOMEO.leq(TransformGroup.EUCLIDEAN)
    assert TransformGroup.SIMILARITY.leq(TransformGroup.AFFINE)


def test_meet_is_greatest_lower_bound_in_chain():
    assert TransformGroup.AFFINE.meet(TransformGroup.SIMILARITY) is TransformGroup.SIMILARITY
    assert TransformGroup.EUCLIDEAN.meet(TransformGroup.HOMEO) is TransformGroup.EUCLIDEAN
    # meet is commutative and idempotent
    for a in groups_chain():
        assert a.meet(a) is a
        for b in groups_chain():
            assert a.meet(b) is b.meet(a)
            # meet is a lower bound of both
            assert a.meet(b).leq(a) and a.meet(b).leq(b)


def test_join_is_least_upper_bound_in_chain():
    assert TransformGroup.AFFINE.join(TransformGroup.SIMILARITY) is TransformGroup.AFFINE
    assert TransformGroup.EUCLIDEAN.join(TransformGroup.HOMEO) is TransformGroup.HOMEO
    for a in groups_chain():
        for b in groups_chain():
            assert a.join(b) is b.join(a)
            assert a.leq(a.join(b)) and b.leq(a.join(b))


# ------------------------------------------------------- per-group invariants

def test_euclidean_observes_coordinates_and_squared_distances(two_touch):
    obs, loss = project(two_touch, TransformGroup.EUCLIDEAN)
    assert obs["group"] == "EUCLIDEAN"
    assert obs["coordinates"]["A"] == ((Fr(0), Fr(0)), (Fr(2), Fr(2)))
    assert obs["side_lengths"]["A"] == (Fr(2), Fr(2))
    assert obs["centroids"]["A"] == (Fr(1), Fr(1))
    # centroid of B = (5/2, 1/2); squared distance = (3/2)^2 + (1/2)^2 = 10/4 = 5/2
    assert obs["squared_centroid_distances"][("A", "B")] == Fr(5, 2)
    assert loss.dropped == ()  # Euclidean drops nothing


def test_similarity_observes_ratios_only_and_drops_coords_metric(two_touch):
    obs, loss = project(two_touch, TransformGroup.SIMILARITY)
    assert obs["group"] == "SIMILARITY"
    assert "coordinates" not in obs and "side_lengths" not in obs
    assert obs["aspect_ratios"]["A"] == (Fr(1), Fr(1))     # square
    assert obs["aspect_ratios"]["B"] == (Fr(1), Fr(1))     # 1x1 also square
    assert obs["pairwise_size_ratios"][("A", "B")] == Fr(2)  # side0 2 / 1
    assert set(loss.dropped) == {"absolute_coordinates", "metric"}


def test_affine_observes_parallelism_and_ordinal_no_ratios(two_touch):
    obs, loss = project(two_touch, TransformGroup.AFFINE)
    assert obs["group"] == "AFFINE"
    # all boxes axis-aligned => both axes have all regions parallel
    assert obs["parallel_edge_classes"][0] == ("A", "B")
    assert obs["parallel_edge_classes"][1] == ("A", "B")
    # A [0,2] vs B [2,3] on axis0 meet; on axis1 A[0,2] contains B[0,1]
    assert obs["axis_ordinal"][("A", "B")] == ("meets", "b_inside_a")
    assert "ratios" in loss.dropped and "parallelism" not in loss.dropped


def test_homeo_observes_contact_only(two_touch):
    obs, loss = project(two_touch, TransformGroup.HOMEO)
    assert obs["group"] == "HOMEO"
    assert obs["rcc8_names"][("A", "B")] == ("EC",)  # externally connected
    assert set(obs.keys()) == {"group", "rcc8", "rcc8_names"}
    # everything but contact is dropped
    for cat in ("absolute_coordinates", "metric", "ratios", "parallelism",
                "orientation", "linking"):
        assert cat in loss.dropped


def test_projective_incidence_collinearity_cross_ratio():
    # Three boxes whose top/bottom edges are collinear along y=0 line gives >=4
    # collinear vertices on that line for cross-ratio; and a shared corner.
    r = _real(
        _box("A", (0, 0), (1, 1)),
        _box("B", (2, 0), (4, 1)),
        _box("C", (5, 0), (8, 1)),
    )
    obs, loss = project(r, TransformGroup.PROJECTIVE)
    assert obs["group"] == "PROJECTIVE"
    assert obs["partial"] is True
    # along axis0 at perpendicular y=0 the x endpoints are {0,1,2,4,5,8} => >=4
    cr_lines = [entry for entry in obs["cross_ratios"] if entry[0] == 0]
    assert cr_lines, "expected cross-ratios along a genuinely collinear line"
    # projective drops parallelism (relative to affine) but keeps collinearity
    assert "parallelism" in loss.dropped
    assert "collinearity" in loss.retained


def test_projective_reports_coincident_vertices():
    # B shares corner (2,0) with A's corner (2,0).
    r = _real(_box("A", (0, 0), (2, 2)), _box("B", (2, 0), (3, 1)))
    obs, _ = project(r, TransformGroup.PROJECTIVE)
    shared = dict((pt, regs) for pt, regs in obs["coincident_vertices"])
    assert (Fr(2), Fr(0)) in shared and shared[(Fr(2), Fr(0))] == ("A", "B")


def test_cross_ratio_exact_value():
    # cross-ratio of 0,1,2,3 = ((2-0)(3-1))/((2-1)(3-0)) = 4/3
    assert _cross_ratio(Fr(0), Fr(1), Fr(2), Fr(3)) == Fr(4, 3)


# ------------------------------------------------------------- loss disclosure

def test_dropped_grows_monotonically_along_chain(two_touch):
    prev: set[str] = set()
    for g in groups_chain():
        _, loss = project(two_touch, g)
        dropped = set(loss.dropped)
        assert prev.issubset(dropped), f"{g.name} must drop a superset of finer groups"
        prev = dropped
    # by HOMEO the four explicitly-named categories are all dropped
    _, homeo_loss = project(two_touch, TransformGroup.HOMEO)
    for cat in ("absolute_coordinates", "metric", "orientation", "linking"):
        assert cat in homeo_loss.dropped


def test_loss_report_retained_and_dropped_disjoint(two_touch):
    for g in groups_chain():
        _, loss = project(two_touch, g)
        assert isinstance(loss, LossReport)
        assert loss.group is g
        assert set(loss.retained).isdisjoint(set(loss.dropped))
        assert "reverse-grounding" in loss.notes  # honesty note present
        d = loss.to_dict()
        assert d["group"] == g.name and d["rank"] == g.rank


# ------------------------------------------------------- read-only guarantee

def test_projection_is_read_only(two_touch):
    before = copy.deepcopy(two_touch)
    for g in groups_chain():
        project(two_touch, g)
    # the realization object is untouched (same identity, equal value)
    assert two_touch == before
    assert two_touch.boxes[0].lo == (Fr(0), Fr(0))


def test_project_rejects_non_group(two_touch):
    with pytest.raises(TypeError):
        project(two_touch, "EUCLIDEAN")  # type: ignore[arg-type]


# ------------------------------------------------ invariance / canonical_key

def test_similarity_invariant_under_uniform_scale_and_translation(two_touch):
    # Apply a similarity (scale by 3, translate by (10,7)); ratios must be identical.
    def transform(box):
        s, tx, ty = Fr(3), Fr(10), Fr(7)
        lo = (box.lo[0] * s + tx, box.lo[1] * s + ty)
        hi = (box.hi[0] * s + tx, box.hi[1] * s + ty)
        return Box(box.region_id, lo, hi)

    moved = _real(*(transform(b) for b in two_touch.boxes))
    o1, _ = project(two_touch, TransformGroup.SIMILARITY)
    o2, _ = project(moved, TransformGroup.SIMILARITY)
    assert o1 == o2  # similarity-invariant read-out unchanged
    # ...while Euclidean coordinates DO change (the info similarity forgets)
    e1, _ = project(two_touch, TransformGroup.EUCLIDEAN)
    e2, _ = project(moved, TransformGroup.EUCLIDEAN)
    assert e1 != e2


def test_homeo_invariant_under_affine_shear_keeps_contact():
    # A homeomorphism (here an order-preserving rescale per axis) preserves EC contact.
    r = _real(_box("A", (0, 0), (2, 2)), _box("B", (2, 0), (3, 1)))
    scaled = _real(
        _box("A", (0, 0), (2, 4)),          # stretched in y
        _box("B", (2, 0), (5, 2)),
    )
    h1, _ = project(r, TransformGroup.HOMEO)
    h2, _ = project(scaled, TransformGroup.HOMEO)
    assert h1["rcc8_names"] == h2["rcc8_names"] == {("A", "B"): ("EC",)}


def test_canonical_key_is_hashable_and_orbit_invariant(two_touch):
    def translate(box, tx, ty):
        return Box(box.region_id,
                   (box.lo[0] + tx, box.lo[1] + ty),
                   (box.hi[0] + tx, box.hi[1] + ty))

    moved = _real(*(translate(b, Fr(100), Fr(-3)) for b in two_touch.boxes))
    for g in (TransformGroup.SIMILARITY, TransformGroup.AFFINE, TransformGroup.HOMEO):
        k1 = canonical_key(two_touch, g)
        k2 = canonical_key(moved, g)
        assert hash(k1) == hash(k2)  # hashable
        assert k1 == k2              # translation-invariant orbit coordinate
    # Euclidean key distinguishes the translated configuration (finer group).
    assert canonical_key(two_touch, TransformGroup.EUCLIDEAN) != \
        canonical_key(moved, TransformGroup.EUCLIDEAN)
