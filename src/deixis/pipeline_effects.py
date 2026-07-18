"""Opt-in capability/effect signature checking for operation chains.

This module adds a *voluntary*, side-effect-free static check on top of the
realizability pipeline. ``deixis.pipeline.run`` is NOT changed and does NOT call
anything here; a caller opts in by threading its operation names through
:func:`check_effects` (and, optionally, :func:`annotate_realization_effects`).

What this is
------------
Every operation the pipeline can perform is given a *capability signature*
:class:`Effect` with three disjoint roles:

* ``requires``     — capabilities that must currently be *available* for the
  operation to be meaningful (e.g. a Euclidean ``distance`` read-out requires the
  Euclidean metric to still hold; an affine ``centroid`` requires an affine frame).
* ``preserves``    — capabilities the operation *establishes / adds* to the available
  set (e.g. ``path_consistency`` establishes ``rcc-consistent``; ``solve`` establishes
  a realized geometry). Capabilities present *before* and NOT in ``invalidates`` persist
  automatically (a frame axiom) — you need not, and should not, relist them here.
* ``invalidates``  — capabilities the operation *destroys* (e.g. a general affine
  map destroys the Euclidean metric; a lossy float cast destroys exact rationality).

An abstract "available capability set" is threaded along the chain, starting from
:data:`INITIAL_CAPABILITIES` (the properties of the raw exact geometric input).
After an operation the set becomes ``(available - invalidates) | preserves`` — invalidated
capabilities drop out, established ones are added, and everything else persists. A
*violation* is raised when an operation ``requires`` a capability that is not currently
available — most importantly when a *prior* operation ``invalidates`` it, or when it was
never established in the first place.

What this is NOT (over-claiming guard)
--------------------------------------
* This is **capability matching, not a reduction to a single group/monoid**. There is
  no claim that operations form one transformation group whose invariants are read off
  by a ``meet``. Each capability is an independent token; requiring several is a
  *conjunction of capabilities*, not membership in one algebraic structure.
* The check is a **mis-signature / mis-ordering detector, not a proof**. A clean
  :func:`check_effects` result means "no *declared* effect is contradicted by the
  chain", i.e. it can only surface signatures that are provably inconsistent with the
  declared ``requires``/``preserves``/``invalidates``. It does NOT prove the operations
  are semantically correct, does NOT prove preservation of any geometric quantity, and
  must NOT be read as reinstating a "fully bidirectional" or "meaning-preserving IR"
  guarantee. Those guarantees were deliberately withdrawn and this module does not
  resurrect them.
* Unknown operations are **fail-closed**: an operation name absent from
  :data:`OPERATION_EFFECTS` yields a violation rather than being assumed harmless.

Everything here is pure and exact-free (it manipulates only capability *labels*); it
touches no coordinates and imposes no float arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

from deixis.core.ids import Provenance
from deixis.core.types import Realization

__all__ = [
    "Effect",
    "EffectViolation",
    "OPERATION_EFFECTS",
    "INITIAL_CAPABILITIES",
    "check_effects",
    "annotate_realization_effects",
]

# ---------------------------------------------------------------- capabilities
# Capability tokens are plain strings (independent labels, NOT elements of one group).
EUCLIDEAN = "euclidean-metric"   # a fixed Euclidean metric holds (distances meaningful)
AFFINE = "affine-frame"          # a fixed affine frame holds (centroid/affine maps meaningful)
EXACT = "exact-rational"         # coordinates are exact Fraction (no float loss)
RCC = "rcc-consistent"           # the symbolic RCC-8 network reached a non-empty fixpoint
GEOM = "realized-in-D"           # a concrete geometry-in-domain exists

#: Capabilities the raw exact geometric input is assumed to provide. The chain is
#: checked against this starting set (override via ``initial=`` on :func:`check_effects`).
INITIAL_CAPABILITIES: frozenset[str] = frozenset({EUCLIDEAN, AFFINE, EXACT})


# ---------------------------------------------------------------- Effect signature
@dataclass(frozen=True)
class Effect:
    """Capability signature of one operation.

    The three roles are checked as *sets of independent capability labels*; there is
    deliberately no algebraic ``meet`` collapsing them into a single invariant group
    (see the module docstring's over-claiming guard). ``preserves`` is the ADD set (the
    capabilities the op establishes); capabilities not in ``invalidates`` persist by a
    frame axiom, so unaffected capabilities must NOT be relisted in ``preserves``.
    ``preserves`` and ``invalidates`` naming the same capability is a contradictory
    signature and is reported by :func:`check_effects` as a mis-signature (it does not
    silently pick a winner)."""

    requires: frozenset[str] = frozenset()
    preserves: frozenset[str] = frozenset()
    invalidates: frozenset[str] = frozenset()


@dataclass(frozen=True)
class EffectViolation:
    """A single detected inconsistency in an operation chain (a *detection*, not a proof)."""

    index: int                       # position in the chain (0-based); -1 for chain-global
    operation: str                   # operation name at that position ("" if global)
    kind: str                        # "unknown-operation" | "missing-requirement" | "contradictory-signature"
    capabilities: frozenset[str] = frozenset()  # the offending capabilities
    detail: str = ""

    def __str__(self) -> str:  # human-readable blame line
        where = f"#{self.index} {self.operation!r}" if self.index >= 0 else "chain"
        caps = ",".join(sorted(self.capabilities))
        base = f"[{self.kind}] {where}"
        if caps:
            base += f": {caps}"
        return base + (f" ({self.detail})" if self.detail else "")


# ---------------------------------------------------------------- operation table
#: Declared effect signatures. A caller may treat this as the closed vocabulary of
#: pipeline operations; anything outside it is fail-closed by :func:`check_effects`.
OPERATION_EFFECTS: dict[str, Effect] = {
    # Symbolic Stage A: purely relational, geometry-agnostic. ESTABLISHES rcc-consistent;
    # the metric / frame / exactness of the input persist untouched (frame axiom).
    "path_consistency": Effect(
        requires=frozenset(),
        preserves=frozenset({RCC}),
        invalidates=frozenset(),
    ),
    # Grounding: narrows masks / flips authority; needs a consistent network, adds nothing.
    "ground": Effect(
        requires=frozenset({RCC}),
        preserves=frozenset(),
        invalidates=frozenset(),
    ),
    # Stage B geometry generation: needs a consistent network + exact coords; ESTABLISHES GEOM.
    "solve": Effect(
        requires=frozenset({RCC, EXACT}),
        preserves=frozenset({GEOM}),
        invalidates=frozenset(),
    ),
    # Exact reverse-verify: reads geometry back on exact bounds; requires exactness + GEOM.
    "reverse_verify": Effect(
        requires=frozenset({GEOM, EXACT}),
        preserves=frozenset(),
        invalidates=frozenset(),
    ),
    # Affine-equivariant query: meaningful under any affine frame (NOT metric-dependent).
    "centroid": Effect(
        requires=frozenset({AFFINE, GEOM}),
        preserves=frozenset(),
        invalidates=frozenset(),
    ),
    # Euclidean-invariant query: requires the metric to still hold.
    "distance": Effect(
        requires=frozenset({EUCLIDEAN, GEOM}),
        preserves=frozenset(),
        invalidates=frozenset(),
    ),
    # General (non-isometric) affine map: keeps the affine frame, DESTROYS the metric.
    "affine_map": Effect(
        requires=frozenset({AFFINE, GEOM}),
        preserves=frozenset(),
        invalidates=frozenset({EUCLIDEAN}),
    ),
    # Lossy numeric cast: keeps a geometry but DESTROYS exact rationality.
    "float_cast": Effect(
        requires=frozenset({GEOM}),
        preserves=frozenset(),
        invalidates=frozenset({EXACT}),
    ),
}


# ---------------------------------------------------------------- checker
def check_effects(
    operation_chain: Sequence[str],
    *,
    initial: Optional[frozenset[str]] = None,
) -> list[EffectViolation]:
    """Check an operation chain against the declared capability signatures.

    Threads an abstract capability set through the chain, starting from ``initial``
    (default :data:`INITIAL_CAPABILITIES`). For each operation, in order:

    * an **unknown operation** (absent from :data:`OPERATION_EFFECTS`) is a fail-closed
      violation, and the available set is left unchanged (we cannot know its effect).
      Checking continues past it as a best-effort diagnostic: the chain as a whole never
      reads clean, but per-operation diagnostics *after* an unknown op assume the unknown
      op neither added nor removed capabilities, so treat them as advisory only;
    * a **contradictory signature** (``preserves`` and ``invalidates`` share a
      capability) is flagged as a mis-signature;
    * a **missing requirement** (a ``requires`` capability not currently available —
      typically because an earlier operation ``invalidates`` it) is a violation;
    * then the available set becomes ``(available - invalidates) | preserves``.

    Returns the list of :class:`EffectViolation` (empty iff the chain is clean). An empty
    chain trivially returns ``[]``.

    This is a mis-signature/mis-ordering *detector*, not a correctness proof, and it is
    capability matching rather than a reduction to a single invariant group — see the
    module docstring.
    """
    available = INITIAL_CAPABILITIES if initial is None else frozenset(initial)
    violations: list[EffectViolation] = []

    for i, op in enumerate(operation_chain):
        eff = OPERATION_EFFECTS.get(op)
        if eff is None:
            # Fail-closed: do not assume an unknown op is a no-op; capabilities unchanged.
            violations.append(
                EffectViolation(
                    index=i,
                    operation=op,
                    kind="unknown-operation",
                    detail="operation not in OPERATION_EFFECTS (fail-closed)",
                )
            )
            continue

        overlap = eff.preserves & eff.invalidates
        if overlap:
            violations.append(
                EffectViolation(
                    index=i,
                    operation=op,
                    kind="contradictory-signature",
                    capabilities=overlap,
                    detail="capability listed in both preserves and invalidates",
                )
            )

        missing = eff.requires - available
        if missing:
            violations.append(
                EffectViolation(
                    index=i,
                    operation=op,
                    kind="missing-requirement",
                    capabilities=missing,
                    detail="required capability not available at this point in the chain",
                )
            )

        available = (available - eff.invalidates) | eff.preserves

    return violations


# ---------------------------------------------------------------- annotation
def annotate_realization_effects(
    real: Realization,
    operation_chain: Sequence[str],
    *,
    initial: Optional[frozenset[str]] = None,
) -> Realization:
    """Return a copy of ``real`` with a :class:`Provenance` recording the effect check.

    Annotation only: this does **not** change ``status`` or any box — the fail-closed
    ladder is owned by :func:`deixis.pipeline.run`, and this module never promotes a
    realization. It records the operation chain and the :func:`check_effects` result so a
    downstream consumer can see whether the chain was effect-consistent (and gate on it
    itself). The provenance ``origin`` is ``"effect-signature"``; violations, if any, are
    summarized in ``note``. Recording a clean check is NOT a proof of correctness (see the
    module docstring's over-claiming guard).

    Provenance is append-only: if ``real`` already carries a ``prov``, its origin/note are
    folded into the new record's ``note`` (and its origin into ``inputs``) rather than being
    discarded, so no prior history is lost.
    """
    violations = check_effects(operation_chain, initial=initial)
    if violations:
        note = "effect violations: " + "; ".join(str(v) for v in violations)
    else:
        note = "no effect violations detected (not a correctness proof)"
    inputs = tuple(operation_chain)
    if real.prov is not None:
        # Do not drop existing provenance: preserve it in the append-only history.
        prior = real.prov
        note = f"{note} | prior[{prior.origin}]: {prior.note}" if prior.note else f"{note} | prior[{prior.origin}]"
        inputs = inputs + (f"prov:{prior.origin}",)
    prov = Provenance(
        origin="effect-signature",
        actor="pipeline_effects",
        activity="check_effects",
        inputs=inputs,
        note=note,
    )
    return replace(real, prov=prov)
