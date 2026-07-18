# Deixis — Witnessed Relational Spatial IR with Deferred Grounding

A **relation-first** spatial description: you describe space by qualitative relations
(RCC-8) with **contact witnesses** (face/line/point) *before* any geometry, mark each
relation as invariant / grounded / delegated (**deferred grounding = decision-rights**),
solve to concrete geometry in a limited exact domain, and **reverse-verify** the geometry
against the spec. Delivered as a standalone Python core and a Grasshopper layer
(*Deixis for Grasshopper*).

Central claim (demo1): RCC-8's single `EC` symbol collapses face/edge/point contact, but
the IR holds the intended contact dimension as a geometry-independent invariant — a thing
IFC/Topologic/Grasshopper's standard first-class constructs cannot hold *before* geometry.

Scope & honesty (adversarially verified): this is an integrated executable IR + explicit
preservation contract for the spatial-relations subdomain — NOT a new topology theory,
NOT "principled inexpressibility", NOT full bidirectionality. See `docs/` for the spec.

## Layout
- `src/deixis/core` — M0 types (`types.py`), ids/Fraction/provenance (`ids.py`), RCC-8 algebra (`rcc8.py`)
- `src/deixis/solver` — path-consistency + tractable fragments (`pathconsistency.py`), geometry solver (`geom_solver.py`)
- `src/deixis/incidence` — contact witnesses (`witness.py`), overlay poset (`overlay.py`)
- `src/deixis/grounding` — `grounding_op.py` (ground/unground, immutable)
- `src/deixis/verify` — `reverse.py` (geometry -> re-extract relations -> compare)
- `src/deixis/pipeline.py` — two-stage judgment + fail-closed realizability
- `src/deixis/demos` — `demo1_contact_dimension.py`
- `tests/` — pytest

## Dev
```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,geom]"
pytest -q
```
