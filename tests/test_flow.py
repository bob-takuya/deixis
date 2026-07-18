"""Tests for M-field(b) — the minimal DEC flow layer (deixis.field.flow).

Covers the four honest claims of the field/flow module:
  * ``d1 ∘ d0 = 0``   (the discrete ∂∂ = 0: a gradient field has no circulation)
  * flux conservation at internal vertices (conservative source→sink flow passes)
  * conservation-violation detection (a leaky / non-conservative flow is caught)
  * the non-manifold guard warns where DEC ⋆ / Hodge is undefined (edge with ≥3 faces)

All exact rational; no floats.
"""
from fractions import Fraction

import pytest

from deixis.field.flow import (
    CellComplex,
    CellComplex1D,
    Cochain,
    ConservationViolation,
    Edge,
    Face,
    ManifoldWarning,
    coboundary_d0,
    coboundary_d1,
    divergence,
    flux_conservation,
    make_grid,
    manifold_guard,
)


# ---------------------------------------------------------------- fixtures
def _grid():
    return make_grid(2, 2)


def _unit_L_flow():
    """Unit flow along v:0:0 -> v:1:0 -> v:2:0 -> v:2:1 -> v:2:2."""
    path = ["eh:0:0", "eh:1:0", "ev:2:0", "ev:2:1"]
    return Cochain(degree=1, values={e: Fraction(1) for e in path})


# ---------------------------------------------------------------- cochain basics
def test_cochain_missing_key_reads_zero():
    c = Cochain(degree=1, values={"eh:0:0": Fraction(3)})
    assert c.get("eh:0:0") == Fraction(3)
    assert c.get("nonexistent") == Fraction(0)


def test_cochain_is_immutable_and_exact():
    c = Cochain(degree=0, values={"v": Fraction(1, 2)})
    with pytest.raises(TypeError):
        c.values["v"] = Fraction(9)          # MappingProxyType is read-only
    with pytest.raises(Exception):
        c.degree = 5                          # frozen dataclass
    assert isinstance(c.get("v"), Fraction)


def test_cochain_coerces_ints_to_fraction():
    c = Cochain(degree=1, values={"e": 4})
    assert c.get("e") == Fraction(4)
    assert isinstance(c.get("e"), Fraction)


def test_cellcomplex1d_alias():
    assert CellComplex1D is CellComplex


# ---------------------------------------------------------------- operators
def test_coboundary_d0_is_potential_difference():
    g = _grid()
    pot = Cochain(degree=0, values={"v:0:0": Fraction(0), "v:1:0": Fraction(5)})
    grad = coboundary_d0(pot, g)
    # edge eh:0:0 goes v:0:0 -> v:1:0, so d0 = f(head)-f(tail) = 5 - 0
    assert grad.get("eh:0:0") == Fraction(5)
    assert grad.degree == 1


def test_constant_potential_has_zero_gradient():
    g = _grid()
    const = Cochain(degree=0, values={v: Fraction(7) for v in g.vertices})
    grad = coboundary_d0(const, g)
    assert grad.is_zero()


def test_dd_is_zero_on_grid():
    """d1 . d0 = 0 for an arbitrary potential (discrete ∂∂ = 0)."""
    g = _grid()
    pot = Cochain(degree=0, values={v: Fraction(i * i - 3 * i + 1, 4)
                                    for i, v in enumerate(g.vertices)})
    curl = coboundary_d1(coboundary_d0(pot, g), g)
    assert curl.degree == 2
    assert curl.is_zero()
    # explicitly: every face circulation is exactly zero
    for f in g.faces:
        assert curl.get(f.id) == Fraction(0)


def test_d1_detects_real_circulation():
    """A non-gradient flow (a loop) has non-zero circulation — d1 is not trivially zero."""
    g = make_grid(1, 1)
    # CCW unit loop around the single face f:0:0
    loop = Cochain(degree=1, values={
        "eh:0:0": Fraction(1),   # +
        "ev:1:0": Fraction(1),   # +
        "eh:0:1": Fraction(-1),  # traversed against, contributes + via face sign (-1)*(-1)
        "ev:0:0": Fraction(-1),
    })
    curl = coboundary_d1(loop, g)
    # face sign pattern (+ + - -): 1 + 1 - (-1) - (-1) = 4
    assert curl.get("f:0:0") == Fraction(4)


def test_wrong_degree_raises():
    g = _grid()
    c1 = _unit_L_flow()
    with pytest.raises(ValueError):
        coboundary_d0(c1, g)          # needs degree 0
    with pytest.raises(ValueError):
        divergence(Cochain(degree=0, values={}), g)  # needs degree 1
    with pytest.raises(ValueError):
        coboundary_d1(Cochain(degree=0, values={}), g)  # needs degree 1


# ---------------------------------------------------------------- divergence
def test_divergence_marks_source_and_sink():
    g = _grid()
    flow = _unit_L_flow()
    div = divergence(flow, g)
    assert div.degree == 0
    assert div.get("v:0:0") == Fraction(1)    # source: net outflow
    assert div.get("v:2:2") == Fraction(-1)   # sink: net inflow
    # every other vertex is internal and balances
    for v in g.vertices:
        if v not in ("v:0:0", "v:2:2"):
            assert div.get(v) == Fraction(0)


def test_total_divergence_is_zero():
    """Global conservation: sum of divergence over all vertices is identically 0."""
    g = _grid()
    # arbitrary flow
    import itertools
    flow = Cochain(degree=1, values={e.id: Fraction(k - 3, 2)
                                     for k, e in enumerate(g.edges)})
    div = divergence(flow, g)
    total = sum((div.get(v) for v in g.vertices), Fraction(0))
    assert total == Fraction(0)


# ---------------------------------------------------------------- conservation
def test_conservative_flow_has_no_violations():
    g = _grid()
    flow = _unit_L_flow()
    source = Cochain(degree=0, values={"v:0:0": Fraction(1), "v:2:2": Fraction(-1)})
    violations = flux_conservation(flow, g, source=source)
    assert violations == []


def test_leaky_flow_violation_detected():
    """A flow that emits at an internal vertex with no declared source is caught."""
    g = _grid()
    leak = Cochain(degree=1, values={"eh:0:0": Fraction(1)})  # v:0:0 out, v:1:0 in, unbalanced
    violations = flux_conservation(leak, g)  # no source declared
    ids = {x.vertex_id for x in violations}
    assert "v:0:0" in ids   # +1 outflow, not a declared source
    assert "v:1:0" in ids   # -1 inflow
    for x in violations:
        assert isinstance(x, ConservationViolation)
        assert x.expected == Fraction(0)


def test_source_exempts_declared_vertices_only():
    """Declaring the wrong divergence at a source still flags it."""
    g = _grid()
    flow = _unit_L_flow()  # real divergence at v:0:0 is +1
    bad_source = Cochain(degree=0, values={"v:0:0": Fraction(5), "v:2:2": Fraction(-1)})
    violations = flux_conservation(flow, g, source=bad_source)
    assert len(violations) == 1
    v = violations[0]
    assert v.vertex_id == "v:0:0"
    assert v.divergence == Fraction(1)
    assert v.expected == Fraction(5)


def test_flux_conservation_rejects_non_zero_degree_source():
    g = _grid()
    flow = _unit_L_flow()
    with pytest.raises(ValueError):
        flux_conservation(flow, g, source=Cochain(degree=1, values={}))


# ---------------------------------------------------------------- manifold guard
def test_grid_is_manifold_no_warnings():
    g = make_grid(3, 3)
    assert manifold_guard(g) == []


def _nonmanifold_fan():
    """Three faces all sharing a single edge e0 (a three-region seam / T-junction)."""
    return CellComplex(
        vertices=("a", "b", "c", "d", "e"),
        edges=(
            Edge("e0", "a", "b"),
            Edge("ea", "b", "c"),
            Edge("eb", "b", "d"),
            Edge("ec", "b", "e"),
        ),
        faces=(
            Face("F1", (("e0", +1), ("ea", +1))),
            Face("F2", (("e0", +1), ("eb", +1))),
            Face("F3", (("e0", +1), ("ec", +1))),
        ),
    )


def test_manifold_guard_warns_on_shared_edge():
    warns = manifold_guard(_nonmanifold_fan())
    assert len(warns) == 1
    w = warns[0]
    assert isinstance(w, ManifoldWarning)
    assert w.kind == "nonmanifold_edge"
    assert w.cell_id == "e0"


def test_edge_repeated_within_one_face_is_not_nonmanifold():
    """An edge listed multiple times in a single face loop counts as one sharing face."""
    cx = CellComplex(
        vertices=("a", "b"),
        edges=(Edge("e0", "a", "b"),),
        faces=(Face("F1", (("e0", +1), ("e0", -1), ("e0", +1))),),
    )
    assert manifold_guard(cx) == []


def test_two_faces_sharing_edge_is_still_manifold():
    """Exactly two faces on an edge is fine (interior of a manifold) — no warning."""
    cx = CellComplex(
        vertices=("a", "b", "c", "d"),
        edges=(Edge("e0", "a", "b"), Edge("ea", "b", "c"), Edge("eb", "b", "d")),
        faces=(
            Face("F1", (("e0", +1), ("ea", +1))),
            Face("F2", (("e0", +1), ("eb", +1))),
        ),
    )
    assert manifold_guard(cx) == []


# ---------------------------------------------------------------- demo smoke
def test_demo6_runs():
    from deixis.field.flow import demo6
    demo6()  # should not raise
