"""M-field(b) — *space as flow*: a minimal Discrete Exterior Calculus (DEC) layer.

This is a **covering-domain extension**, explicitly *outside* the core claim of the
Witnessed Relational IR (``deixis.core``). The core describes space as *regions +
qualitative relations + witnessed contacts*. Here we describe the *complementary*
reading — space as a **flow field** — with the smallest honest DEC skeleton:

    circulation / movement / airflow / sightlines  →  1-cochain (values on directed edges)
    sources & sinks (湧出 / 吸込)                    →  0-cochain (values on vertices)
    face flux / circulation around a cell           →  2-cochain (values on faces)

and the two discrete operators that connect them:

    coboundary_d0 : 0-cochain → 1-cochain      (discrete gradient of a potential)
    coboundary_d1 : 1-cochain → 2-cochain      (discrete curl / circulation on faces)
    divergence    : 1-cochain → 0-cochain      (net out-flow per vertex)

Two identities are exact and testable here (all rational, no geometry needed):

  * ``d∘d = 0``  —  on a **well-formed** complex (every face is a *closed* signed
    edge loop referencing real edges, as :func:`make_grid` produces),
    :func:`coboundary_d1` ∘ :func:`coboundary_d0` is the zero 2-cochain for *any*
    potential (the discrete ``∂∂ = 0``; a gradient field has no circulation). We do
    **not** validate face loops, so feeding a non-closed / dangling face breaks the
    identity — the guarantee is a property of well-formed input, not enforced here.
  * **flux conservation** — at an *internal* vertex (not a declared source/sink)
    the net out-flow (divergence) of a flow is exactly ``0``. Sources/sinks are
    the vertices where a prescribed non-zero divergence is *allowed* and checked.

Honest limits (do NOT read past these):

  * Everything here is the **combinatorial / topological** half of DEC: coboundary
    ``d`` and the boundary-adjoint divergence. We deliberately do **not** build the
    Hodge star ``⋆`` or the primal↔dual metric. A correct ``⋆`` needs a
    *well-centered* (circumcentric-dual-valid) mesh and a manifold complex; on a
    general or non-manifold complex it is ill-defined, so we refuse to fake it.
  * :func:`manifold_guard` makes that refusal *loud*: it flags the places
    (an edge shared by ≥3 faces — three regions meeting along one seam) where
    ``⋆`` / Hodge / a dual mesh break down, instead of silently returning a number.
  * Conservation / divergence statements are relative to *this* discrete complex
    and the declared source/sink set — they are a (ρ, band)-relative reading of a
    flow, not a claim about a continuum vector field.

No floats anywhere: all cochain values are exact :class:`fractions.Fraction`
(coerced through :func:`deixis.core.ids.F`). All structural objects are frozen.
This module defines only its own types; it never edits the M0 contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping, Optional

from deixis.core.ids import F

__all__ = [
    "Edge",
    "Face",
    "CellComplex",
    "CellComplex1D",
    "Cochain",
    "ConservationViolation",
    "ManifoldWarning",
    "coboundary_d0",
    "coboundary_d1",
    "divergence",
    "flux_conservation",
    "manifold_guard",
    "make_grid",
    "demo6",
]

_ZERO = Fraction(0)


# ---------------------------------------------------------------- structure
@dataclass(frozen=True)
class Edge:
    """A directed 1-cell ``tail → head`` (orientation carries the sign of DEC ``d``)."""
    id: str
    tail: str          # vertex id
    head: str          # vertex id


@dataclass(frozen=True)
class Face:
    """A 2-cell as an oriented loop of edges.

    ``edges`` is a tuple of ``(edge_id, sign)`` where ``sign`` is ``+1`` if the face's
    boundary traverses that edge along its own direction, ``-1`` if against it. This is
    all the incidence ``d1`` needs; no coordinates are required.
    """
    id: str
    edges: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class CellComplex:
    """A minimal oriented cell complex: vertices (0), directed edges (1), faces (2).

    Faces are optional — a pure 1-complex (movement graph) just leaves ``faces`` empty.
    Everything is frozen; lookups are recomputed on demand (small, exact, no caching
    surprises)."""
    vertices: tuple[str, ...] = ()
    edges: tuple[Edge, ...] = ()
    faces: tuple[Face, ...] = ()

    # -- lookups (never mutate) --
    def edge(self, eid: str) -> Optional[Edge]:
        return next((e for e in self.edges if e.id == eid), None)

    def in_edges(self, vid: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.head == vid)

    def out_edges(self, vid: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.tail == vid)


# ``CellComplex1D`` is the same object used as a pure vertex+edge 1-complex.
CellComplex1D = CellComplex


# ---------------------------------------------------------------- cochain
@dataclass(frozen=True, eq=False)
class Cochain:
    """A discrete ``k``-form: exact rational ``values`` keyed by cell id.

    * ``degree == 0`` → keyed by vertex id (potentials, sources/sinks).
    * ``degree == 1`` → keyed by directed-edge id (flow / movement / airflow).
    * ``degree == 2`` → keyed by face id (flux / circulation).

    Missing keys read as exactly ``0`` (a cochain is a *total* function on cells).
    Values are coerced to :class:`fractions.Fraction`; the map is exposed read-only.
    """
    degree: int
    values: Mapping[str, Fraction] = field(default_factory=dict)

    def __post_init__(self) -> None:
        norm = {k: F(v) for k, v in dict(self.values).items()}
        object.__setattr__(self, "values", MappingProxyType(norm))

    def get(self, cell_id: str) -> Fraction:
        return self.values.get(cell_id, _ZERO)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cochain):
            return NotImplemented
        if self.degree != other.degree:
            return False
        keys = set(self.values) | set(other.values)
        return all(self.get(k) == other.get(k) for k in keys)

    def __hash__(self) -> int:
        nz = tuple(sorted((k, v) for k, v in self.values.items() if v != _ZERO))
        return hash((self.degree, nz))

    def is_zero(self) -> bool:
        return all(v == _ZERO for v in self.values.values())


# ---------------------------------------------------------------- warnings
@dataclass(frozen=True)
class ConservationViolation:
    """An internal vertex whose net out-flow (divergence) is not the expected value."""
    vertex_id: str
    divergence: Fraction
    expected: Fraction = _ZERO


@dataclass(frozen=True)
class ManifoldWarning:
    """A place where DEC ``⋆`` / Hodge / a dual mesh is ill-defined — flagged, not faked."""
    kind: str          # e.g. "nonmanifold_edge"
    cell_id: str
    detail: str = ""


# ---------------------------------------------------------------- operators
def coboundary_d0(c0: Cochain, complex: CellComplex) -> Cochain:
    """Discrete gradient ``d0``: 0-cochain → 1-cochain.

    ``(d0 f)(e) = f(head(e)) − f(tail(e))`` for each directed edge ``e``. Exactly the
    potential difference along the edge; a constant potential maps to the zero flow.
    """
    if c0.degree != 0:
        raise ValueError(f"coboundary_d0 expects a 0-cochain, got degree {c0.degree}")
    out: dict[str, Fraction] = {}
    for e in complex.edges:
        out[e.id] = c0.get(e.head) - c0.get(e.tail)
    return Cochain(degree=1, values=out)


def coboundary_d1(c1: Cochain, complex: CellComplex) -> Cochain:
    """Discrete curl ``d1``: 1-cochain → 2-cochain (circulation around each face).

    ``(d1 w)(f) = Σ sign · w(edge)`` over the oriented boundary loop of face ``f``.
    Together with :func:`coboundary_d0` this gives the exact ``d1 ∘ d0 = 0`` identity.
    """
    if c1.degree != 1:
        raise ValueError(f"coboundary_d1 expects a 1-cochain, got degree {c1.degree}")
    out: dict[str, Fraction] = {}
    for f in complex.faces:
        total = _ZERO
        for eid, sign in f.edges:
            total += F(sign) * c1.get(eid)
        out[f.id] = total
    return Cochain(degree=2, values=out)


def _divergence_at(c1: Cochain, complex: CellComplex, vid: str) -> Fraction:
    """Net out-flow at a vertex: (Σ flow leaving) − (Σ flow entering)."""
    out = sum((c1.get(e.id) for e in complex.out_edges(vid)), _ZERO)
    inn = sum((c1.get(e.id) for e in complex.in_edges(vid)), _ZERO)
    return out - inn


def divergence(c1: Cochain, complex: CellComplex) -> Cochain:
    """Discrete divergence of a flow: 1-cochain → 0-cochain, the **net out-flow** per vertex.

    ``div(w)(v) = Σ_{tail(e)=v} w(e) − Σ_{head(e)=v} w(e)``. Positive = a source (湧出,
    net outflow), negative = a sink (吸込). (This is the physics/source-positive sign; with
    the module's boundary orientation ``∂e = head − tail`` it equals the *negative* of the
    bare transpose ``∂ᵀ`` — we use the out-flow sign so that sources are positive, and do
    not claim it is the raw ``∂``-adjoint.) Summed over *all* vertices it is identically
    ``0`` (every edge contributes ``+w`` at its tail and ``−w`` at its head) — the discrete
    global-conservation / telescoping identity.
    """
    if c1.degree != 1:
        raise ValueError(f"divergence expects a 1-cochain, got degree {c1.degree}")
    out = {v: _divergence_at(c1, complex, v) for v in complex.vertices}
    return Cochain(degree=0, values=out)


def flux_conservation(
    c1: Cochain,
    complex: CellComplex,
    source: Optional[Cochain] = None,
) -> list[ConservationViolation]:
    """Check flux conservation: internal vertices must have ``divergence == 0``.

    ``source`` (an optional 0-cochain of *prescribed* net out-flow) declares the
    湧出 / 吸込 vertices: at each vertex present in ``source`` the divergence must equal
    the prescribed value (and those vertices are therefore *exempt* from the ``==0``
    test); every other vertex is *internal* and must balance to exactly ``0``.

    Returns the list of violating vertices (empty ⇒ conserved). All comparisons exact.
    """
    if c1.degree != 1:
        raise ValueError(f"flux_conservation expects a 1-cochain, got degree {c1.degree}")
    if source is not None and source.degree != 0:
        raise ValueError("source must be a 0-cochain (vertex-keyed)")
    prescribed = source.values if source is not None else {}
    violations: list[ConservationViolation] = []
    for v in complex.vertices:
        d = _divergence_at(c1, complex, v)
        expected = prescribed.get(v, _ZERO)
        if d != expected:
            violations.append(ConservationViolation(vertex_id=v, divergence=d, expected=expected))
    return violations


def manifold_guard(complex: CellComplex) -> list[ManifoldWarning]:
    """Flag where DEC ``⋆`` / Hodge / the dual mesh break down — instead of faking a number.

    A well-defined circumcentric dual (needed for the Hodge star) requires a *manifold*
    complex: every interior 1-cell borders **at most two** 2-cells. An edge shared by
    ``≥3`` faces — three regions meeting along one seam (a T-junction / branching wall) —
    has no single dual edge, so ``⋆`` is ill-defined there. We emit a warning per such
    edge rather than let a downstream Hodge computation silently produce nonsense.

    This is a *guard*, not a repair: it names the non-manifold locus honestly.
    """
    warnings: list[ManifoldWarning] = []
    face_count: dict[str, int] = {}
    for f in complex.faces:
        # count DISTINCT faces per edge, not raw occurrences: an edge listed twice in
        # one face loop must not masquerade as two faces sharing it.
        for eid in {eid for eid, _sign in f.edges}:
            face_count[eid] = face_count.get(eid, 0) + 1
    for eid, n in sorted(face_count.items()):
        if n >= 3:
            warnings.append(
                ManifoldWarning(
                    kind="nonmanifold_edge",
                    cell_id=eid,
                    detail=f"edge borders {n} faces (>2): Hodge star / dual mesh undefined here",
                )
            )
    return warnings


# ---------------------------------------------------------------- grid builder
def make_grid(nx: int, ny: int) -> CellComplex:
    """A rectangular ``nx × ny`` grid as an oriented 2-complex (well-centered, manifold).

    Vertices ``v:i:j`` (0 ≤ i ≤ nx, 0 ≤ j ≤ ny). Horizontal edges ``eh:i:j`` go
    ``(i,j)→(i+1,j)``; vertical edges ``ev:i:j`` go ``(i,j)→(i,j+1)`` — every edge points
    in the ``+x`` / ``+y`` direction. Faces ``f:i:j`` are unit cells with the CCW loop
    ``bottom(+) right(+) top(−) left(−)``. Every interior edge borders exactly two faces,
    so :func:`manifold_guard` is clean on a grid.
    """
    if nx < 1 or ny < 1:
        raise ValueError("grid needs nx >= 1 and ny >= 1")
    verts = tuple(f"v:{i}:{j}" for i in range(nx + 1) for j in range(ny + 1))
    edges: list[Edge] = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            if i < nx:
                edges.append(Edge(id=f"eh:{i}:{j}", tail=f"v:{i}:{j}", head=f"v:{i+1}:{j}"))
            if j < ny:
                edges.append(Edge(id=f"ev:{i}:{j}", tail=f"v:{i}:{j}", head=f"v:{i}:{j+1}"))
    faces: list[Face] = []
    for i in range(nx):
        for j in range(ny):
            faces.append(
                Face(
                    id=f"f:{i}:{j}",
                    edges=(
                        (f"eh:{i}:{j}", +1),        # bottom, left→right
                        (f"ev:{i+1}:{j}", +1),      # right,  bottom→top
                        (f"eh:{i}:{j+1}", -1),      # top,    traversed right→left
                        (f"ev:{i}:{j}", -1),        # left,   traversed top→bottom
                    ),
                )
            )
    return CellComplex(vertices=verts, edges=tuple(edges), faces=tuple(faces))


# ---------------------------------------------------------------- demo6
def demo6() -> None:  # pragma: no cover
    """demo6 — *space as flow*: divergence, flux conservation, ∂∂=0, non-manifold guard.

    (1) A conservative source→sink flow on a 2×2 grid: one 湧出 vertex, one 吸込 vertex,
        every internal vertex balances → :func:`flux_conservation` returns no violations.
    (2) A leaky flow (an internal vertex emits with no declared source) → a violation is
        reported: we do not silently "conserve" a non-conservative field.
    (3) ``d1 ∘ d0 = 0``: the circulation of any gradient potential is the zero 2-cochain.
    (4) A non-manifold complex (one edge shared by 3 faces) → :func:`manifold_guard`
        warns that the Hodge star / dual mesh is undefined there.
    """
    print("=" * 70)
    print("demo6 — space as flow (minimal DEC: d, divergence, conservation, guard)")
    print("=" * 70)

    g = make_grid(2, 2)
    print(f"grid 2x2: {len(g.vertices)} vertices, {len(g.edges)} edges, {len(g.faces)} faces")

    # (1) conservative source -> sink flow ------------------------------------
    # unit flow along the L-path v:0:0 -> v:1:0 -> v:2:0 -> v:2:1 -> v:2:2
    path = ["eh:0:0", "eh:1:0", "ev:2:0", "ev:2:1"]
    flow = Cochain(degree=1, values={e: Fraction(1) for e in path})
    source = Cochain(degree=0, values={"v:0:0": Fraction(1), "v:2:2": Fraction(-1)})
    v1 = flux_conservation(flow, g, source=source)
    print("\n(1) conservative source->sink flow (unit along an L-path):")
    print(f"    divergence(v:0:0)={_divergence_at(flow, g, 'v:0:0')} (湧出) "
          f"divergence(v:2:2)={_divergence_at(flow, g, 'v:2:2')} (吸込)")
    print(f"    flux_conservation violations = {v1}   <- [] means conserved")

    # (2) leaky flow: internal vertex emits, no declared source ----------------
    leak = Cochain(degree=1, values={"eh:0:0": Fraction(1)})  # dangles into v:1:0
    v2 = flux_conservation(leak, g)  # no source declared -> every vertex must be 0
    print("\n(2) leaky flow (no declared source):")
    print(f"    flux_conservation violations = "
          f"{[(x.vertex_id, str(x.divergence)) for x in v2]}   <- non-empty: detected")

    # (3) d1 . d0 = 0 : gradient of a potential has no circulation --------------
    pot = Cochain(degree=0, values={v: F(idx) for idx, v in enumerate(g.vertices)})
    curl = coboundary_d1(coboundary_d0(pot, g), g)
    print("\n(3) d1(d0(potential)) is the zero 2-cochain (∂∂=0):")
    print(f"    curl.is_zero() = {curl.is_zero()}")

    # (4) non-manifold guard ---------------------------------------------------
    # three faces all sharing edge e0 (a T-junction / three-region seam).
    e0 = Edge(id="e0", tail="a", head="b")
    fan = CellComplex(
        vertices=("a", "b", "c", "d", "e"),
        edges=(e0, Edge("ea", "b", "c"), Edge("eb", "b", "d"), Edge("ec", "b", "e")),
        faces=(
            Face("F1", (("e0", +1), ("ea", +1))),
            Face("F2", (("e0", +1), ("eb", +1))),
            Face("F3", (("e0", +1), ("ec", +1))),
        ),
    )
    warns = manifold_guard(fan)
    print("\n(4) non-manifold complex (edge e0 shared by 3 faces):")
    for w in warns:
        print(f"    WARN {w.kind}: {w.cell_id} — {w.detail}")
    print("    (Hodge star ⋆ is NOT computed here; the guard refuses to fake it.)")
    print("=" * 70)


if __name__ == "__main__":  # pragma: no cover
    demo6()
