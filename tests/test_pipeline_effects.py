"""Tests for the opt-in capability/effect signature checker (deixis.pipeline_effects).

These exercise capability *matching* (not a reduction to one invariant group) and the
fail-closed behaviour on unknown operations. The checker is a mis-signature/mis-ordering
detector, not a correctness proof — the tests assert exactly that scope, no more.
"""
from __future__ import annotations

import pytest

from deixis.core.ids import Provenance
from deixis.core.types import Realization, Realizability
from deixis import pipeline_effects as pe
from deixis.pipeline_effects import (
    Effect,
    EffectViolation,
    OPERATION_EFFECTS,
    INITIAL_CAPABILITIES,
    check_effects,
    annotate_realization_effects,
)


# ---------------------------------------------------------------- Effect dataclass
def test_effect_is_frozen():
    eff = Effect(requires=frozenset({"a"}))
    with pytest.raises(Exception):
        eff.requires = frozenset({"b"})  # type: ignore[misc]


def test_operation_table_populated():
    # Sanity: the declared vocabulary is non-empty and every entry is an Effect.
    assert OPERATION_EFFECTS
    assert all(isinstance(v, Effect) for v in OPERATION_EFFECTS.values())


# ---------------------------------------------------------------- clean chains
def test_empty_chain_is_clean():
    assert check_effects([]) == []


def test_correct_chain_passes():
    # Symbolic closure establishes rcc-consistent; solve consumes it and makes geometry;
    # centroid is affine-equivariant (frame still present) -> no violation.
    chain = ["path_consistency", "solve", "centroid"]
    assert check_effects(chain) == []


def test_correct_euclidean_query_passes():
    # distance requires the Euclidean metric, which nothing has invalidated yet.
    chain = ["path_consistency", "solve", "distance"]
    assert check_effects(chain) == []


def test_full_pipeline_shape_passes():
    chain = ["path_consistency", "ground", "solve", "reverse_verify"]
    assert check_effects(chain) == []


# ---------------------------------------------------------------- invalidate then require
def test_invalidate_then_require_violates():
    # affine_map destroys the Euclidean metric; a following distance requires it -> violation.
    chain = ["path_consistency", "solve", "affine_map", "distance"]
    violations = check_effects(chain)
    assert len(violations) == 1
    v = violations[0]
    assert v.kind == "missing-requirement"
    assert v.operation == "distance"
    assert v.index == 3
    assert pe.EUCLIDEAN in v.capabilities


def test_centroid_survives_affine_map():
    # centroid is affine-equivariant: an affine map keeps the frame, so no violation.
    chain = ["path_consistency", "solve", "affine_map", "centroid"]
    assert check_effects(chain) == []


def test_float_cast_then_exact_requirement_violates():
    # A lossy cast destroys exact-rational; reverse_verify requires it -> violation.
    chain = ["path_consistency", "solve", "float_cast", "reverse_verify"]
    violations = check_effects(chain)
    assert any(
        v.kind == "missing-requirement" and pe.EXACT in v.capabilities
        for v in violations
    )


# ---------------------------------------------------------------- missing requirement (no prior)
def test_solve_without_consistency_violates():
    # solve requires rcc-consistent, which nothing established -> missing-requirement.
    violations = check_effects(["solve"])
    assert len(violations) == 1
    assert violations[0].kind == "missing-requirement"
    assert pe.RCC in violations[0].capabilities


def test_geometry_query_before_solve_violates():
    # centroid needs a realized geometry; none exists before solve.
    violations = check_effects(["path_consistency", "centroid"])
    assert len(violations) == 1
    assert violations[0].kind == "missing-requirement"
    assert pe.GEOM in violations[0].capabilities


# ---------------------------------------------------------------- unknown operation fail-closed
def test_unknown_operation_fails_closed():
    violations = check_effects(["frobnicate"])
    assert len(violations) == 1
    assert violations[0].kind == "unknown-operation"
    assert violations[0].operation == "frobnicate"


def test_unknown_operation_does_not_grant_capabilities():
    # An unknown op must not be assumed to establish anything: a following solve still
    # lacks rcc-consistent -> its own missing-requirement violation is reported too.
    violations = check_effects(["frobnicate", "solve"])
    kinds = {(v.kind, v.operation) for v in violations}
    assert ("unknown-operation", "frobnicate") in kinds
    assert ("missing-requirement", "solve") in kinds


# ---------------------------------------------------------------- contradictory signature
def test_contradictory_signature_detected():
    # A hand-built bad signature (same capability preserved AND invalidated) is flagged.
    bad = "bad_op_for_test"
    OPERATION_EFFECTS[bad] = Effect(
        requires=frozenset(),
        preserves=frozenset({pe.EXACT}),
        invalidates=frozenset({pe.EXACT}),
    )
    try:
        violations = check_effects([bad])
        assert any(v.kind == "contradictory-signature" for v in violations)
    finally:
        del OPERATION_EFFECTS[bad]


# ---------------------------------------------------------------- custom initial set
def test_custom_initial_capabilities():
    # Starting without exact-rational, solve's EXACT requirement is missing even after
    # path_consistency (which preserves EXACT only if it was there — it wasn't).
    violations = check_effects(
        ["path_consistency", "solve"],
        initial=frozenset({pe.EUCLIDEAN, pe.AFFINE}),
    )
    assert any(
        v.kind == "missing-requirement" and pe.EXACT in v.capabilities and v.operation == "solve"
        for v in violations
    )


# ---------------------------------------------------------------- annotation
def _real() -> Realization:
    return Realization(id="r0", status=Realizability.REALIZED_IN_D)


def test_annotate_attaches_provenance_and_preserves_status():
    real = _real()
    out = annotate_realization_effects(real, ["path_consistency", "solve"])
    assert out.prov is not None
    assert out.prov.origin == "effect-signature"
    assert out.prov.inputs == ("path_consistency", "solve")
    # Annotation only: status and boxes are untouched.
    assert out.status is Realizability.REALIZED_IN_D
    assert out.status is real.status


def test_annotate_records_violations_in_note():
    real = _real()
    out = annotate_realization_effects(real, ["path_consistency", "solve", "affine_map", "distance"])
    assert "violation" in out.prov.note.lower()
    # Original realization is not mutated (frozen).
    assert real.prov is None


def test_annotate_preserves_prior_provenance():
    # Append-only: an existing prov must not be silently discarded.
    prior = Provenance(origin="solve", actor="geom_solver", note="realized 2 boxes")
    real = Realization(id="r0", status=Realizability.REALIZED_IN_D, prov=prior)
    out = annotate_realization_effects(real, ["path_consistency", "solve"])
    assert out.prov.origin == "effect-signature"
    # prior origin + note are folded into the new record.
    assert "solve" in out.prov.note
    assert "realized 2 boxes" in out.prov.note
    assert any("solve" in x for x in out.prov.inputs)


def test_annotate_clean_note_is_not_a_proof_claim():
    out = annotate_realization_effects(_real(), ["path_consistency", "solve"])
    assert "not a correctness proof" in out.prov.note.lower()


# ---------------------------------------------------------------- violation str
def test_violation_str_is_informative():
    v = EffectViolation(index=2, operation="distance", kind="missing-requirement",
                        capabilities=frozenset({pe.EUCLIDEAN}))
    s = str(v)
    assert "distance" in s and "missing-requirement" in s and pe.EUCLIDEAN in s
