"""Tests for M-field (a) intent layer — thresholding + connected-component split.

Claims exercised:

  (1) A level set that is one connected blob yields a single candidate component
      whose cells equal exactly ``threshold_ground``'s level set.
  (2) A level set that falls into disconnected blobs is split into several
      candidate components; their union is exactly ``threshold_ground``'s level set
      and they are pairwise disjoint.
  (3) Each component is a ``frozenset`` of grid cells (multi-indices).
  (4) There is exactly one GroundingDecision per component, in the same order, and
      each carries the ``intent`` label and provenance back to the base decision.
  (5) Adjacency is face-only (von Neumann): corner-touching cells are separate.
  (6) Everything is exact rational and consistent with the un-edited
      ``field/scalar.threshold_ground``.
"""
from fractions import Fraction

import pytest

from deixis.core.types import GroundingDecision
from deixis.field.scalar import ScalarField, threshold_ground
from deixis.field.scalar_intent import connected_components, threshold_regions

F = Fraction


def _field(shape, vals, spacing=F(1)):
    origin = tuple(F(0) for _ in shape)
    return ScalarField(grid_shape=shape, values=tuple(F(v) for v in vals),
                       origin=origin, spacing=spacing)


# --------------------------------------------------------- single connected blob
def test_single_connected_component_equals_level_set():
    # 1x5 corridor, a single contiguous run clears tau.
    fld = _field((5,), (0, 1, 1, 1, 0))
    comps, decisions = threshold_regions(fld, F(1), "warm")
    # one component
    assert len(comps) == 1
    assert len(decisions) == 1
    # its cells equal threshold_ground's level set exactly
    base_cells, _ = threshold_ground(fld, F(1), "warm")
    assert comps[0] == base_cells
    assert comps[0] == frozenset({(1,), (2,), (3,)})
    assert isinstance(comps[0], frozenset)


# --------------------------------------------------- multiple disconnected blobs
def test_multiple_components_split_and_partition_level_set():
    # 1x5: two runs clearing tau separated by a below-tau cell -> two components.
    fld = _field((5,), (1, 1, 0, 1, 1))
    comps, decisions = threshold_regions(fld, F(1), "warm")
    assert len(comps) == 2
    assert len(decisions) == 2
    # deterministic order by minimum cell
    assert comps[0] == frozenset({(0,), (1,)})
    assert comps[1] == frozenset({(3,), (4,)})
    # union == level set, pairwise disjoint
    base_cells, _ = threshold_ground(fld, F(1), "warm")
    assert comps[0] | comps[1] == base_cells
    assert comps[0].isdisjoint(comps[1])


def test_components_are_cell_sets_2d():
    # 3x3 with two separate 2D blobs.
    #  1 1 0
    #  0 0 0
    #  0 1 1
    fld = _field((3, 3), (1, 1, 0,
                          0, 0, 0,
                          0, 1, 1))
    comps, _ = threshold_regions(fld, F(1), "r")
    assert len(comps) == 2
    assert comps[0] == frozenset({(0, 0), (0, 1)})
    assert comps[1] == frozenset({(2, 1), (2, 2)})
    for comp in comps:
        assert isinstance(comp, frozenset)
        for cell in comp:
            assert isinstance(cell, tuple) and len(cell) == 2


# ------------------------------------------------ face adjacency, not diagonal
def test_corner_touch_is_not_connected():
    # 2x2 diagonal: (0,0) and (1,1) clear tau but touch only at a corner.
    #  1 0
    #  0 1
    fld = _field((2, 2), (1, 0,
                          0, 1))
    comps, _ = threshold_regions(fld, F(1), "r")
    assert len(comps) == 2
    assert comps[0] == frozenset({(0, 0)})
    assert comps[1] == frozenset({(1, 1)})


def test_face_adjacency_connects_orthogonal():
    #  1 1
    #  1 0   -> an L of three cells, all face-connected
    fld = _field((2, 2), (1, 1,
                          1, 0))
    comps, _ = threshold_regions(fld, F(1), "r")
    assert len(comps) == 1
    assert comps[0] == frozenset({(0, 0), (0, 1), (1, 0)})


# ----------------------------------------------------------------- empty level set
def test_empty_level_set_yields_no_components():
    fld = _field((3,), (0, 0, 0))
    comps, decisions = threshold_regions(fld, F(1), "r")
    assert comps == []
    assert decisions == []


# --------------------------------------------------------- decisions & intent label
def test_one_decision_per_component_with_intent_label():
    fld = _field((5,), (1, 1, 0, 1, 1))
    comps, decisions = threshold_regions(fld, F(1), "warm", intent="analysis_candidate")
    assert len(decisions) == len(comps) == 2
    for k, dec in enumerate(decisions):
        assert isinstance(dec, GroundingDecision)
        # intent recorded in the decision
        assert "analysis_candidate" in dec.fixed_value_or_domain
        assert "analysis_candidate" in dec.rationale
        # target names a per-component candidate
        assert dec.target == f"warm#c{k}"
        # not overclaimed as a convex space
        assert "candidate" in dec.fixed_value_or_domain.lower()
        assert "NOT a verified convex space" in dec.fixed_value_or_domain
        # provenance traces back to the delegated threshold_ground decision
        assert dec.prov is not None
        assert dec.prov.activity == "threshold_regions"
        base_cells, base_dec = threshold_ground(fld, F(1), "warm")
        assert base_dec.id in dec.prov.inputs


def test_decision_ids_disambiguate_different_cell_sets():
    # Same region_id / tau / geometry, but different field values threshold to
    # different single components -> decision ids must NOT collide (cells digest).
    fa = _field((3,), (1, 0, 0))
    fb = _field((3,), (0, 0, 1))
    _, da = threshold_regions(fa, F(1), "r")
    _, db = threshold_regions(fb, F(1), "r")
    assert len(da) == len(db) == 1
    assert da[0].id != db[0].id
    # the actual cells are recorded in the decision for auditability
    assert "cells=[0]" in da[0].fixed_value_or_domain
    assert "cells=[2]" in db[0].fixed_value_or_domain


def test_custom_intent_label_propagates():
    fld = _field((3,), (1, 1, 1))
    _, decisions = threshold_regions(fld, F(1), "r", intent="corridor_probe")
    assert len(decisions) == 1
    assert "corridor_probe" in decisions[0].fixed_value_or_domain
    assert decisions[0].prov.note.find("corridor_probe") >= 0


# -------------------------------------------------- consistency with threshold_ground
def test_union_of_components_matches_threshold_ground_for_various_tau():
    fld = _field((6,), (1, 3, 1, 4, 4, 1))  # some structure
    for tau in (F(1), F(2), F(3), F(4), F(5)):
        comps, decisions = threshold_regions(fld, tau, "r")
        base_cells, _ = threshold_ground(fld, tau, "r")
        union = frozenset().union(*comps) if comps else frozenset()
        assert union == base_cells
        assert len(decisions) == len(comps)
        # pairwise disjoint
        seen = set()
        for c in comps:
            assert seen.isdisjoint(c)
            seen |= c


# ---------------------------------------------------- connected_components directly
def test_connected_components_singletons_and_order():
    cells = frozenset({(0,), (2,), (4,)})  # all isolated
    comps = connected_components(cells)
    assert comps == [frozenset({(0,)}), frozenset({(2,)}), frozenset({(4,)})]


def test_connected_components_mixed_rank_rejected():
    with pytest.raises(ValueError):
        connected_components(frozenset({(0,), (1, 1)}))


def test_connected_components_empty():
    assert connected_components(frozenset()) == []


# ----------------------------------------------------------------- exactness / types
def test_values_are_exact_and_scalar_unmodified():
    fld = _field((4,), (1, 0, 1, 1))
    comps, _ = threshold_regions(fld, F(1), "r")
    # cells are int-index tuples, exact
    for comp in comps:
        for cell in comp:
            for i in cell:
                assert isinstance(i, int) and not isinstance(i, bool)
    # threshold_ground still behaves identically (module did not edit scalar)
    base_cells, _ = threshold_ground(fld, F(1), "r")
    assert frozenset().union(*comps) == base_cells
