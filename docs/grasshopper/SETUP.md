# Deixis for Grasshopper — setup & first experiment

These are the **thin Rhino-side shells** for *Deixis for Grasshopper*: an insertable
**relational control / re-synthesis stage** (not a magic wrapper over arbitrary GH).
All real logic + exactness live in the tested Python core (`src/deixis`, 397 tests);
each Grasshopper component just marshals inputs into one `deixis.io.adapters` call.

> Honest scope: `Lift` is a *tolerance-aware observation* (never auto-invariant).
> `Solve` is **relation-preserving re-synthesis** — it re-grows axis-aligned boxes that
> satisfy the relations/witnesses, it does **not** preserve the original topology / surfaces
> / Wasp parts / Kangaroo physics. It works standalone (author relations from scratch) or
> downstream of any plugin's geometry.

## 0. Requirements
- Rhino **8** (has the Python 3 / CPython "Script" component). macOS or Windows.
- The `deixis` checkout on disk (this repo). Note the absolute path to its `src/`
  (the scripts default to the placeholder `/path/to/deixis/src` — edit it,
  or feed a `deixis_src` string input).

## 1. Make one Script component per file
For each `gh_*.py` (except `gh_00_bootstrap.py`, which is just the shared header):

1. Grasshopper → drop a **Script** component (Rhino 8) and set its language to **Python 3**.
2. Open it, **paste the file's contents**.
3. Add the **input params** named exactly as in the file's docstring (right-click the
   component → Manage inputs), and the **output params** likewise. Types:
   - string inputs (`spec`, `src`, `dst`, `rcc8`, `status`, `region_a`, …): item access, `str`.
   - `geometry` (Lift): list access, Geometry/Brep. `region_ids`: list access, `str`.
   - `bounds` (Solve): list access, `float`. `tolerance`: item, `float`.
   - outputs (`spec`, `realization`, `report`, `geometry`, `status`, `realizations`): as named.
4. First run installs `z3-solver`, `pyrsistent`, `networkx` into Rhino's CPython (the
   `# r:` lines). Give it a moment on the very first solve.

The 7 components: **Lift · Relate · Witness · Ground · Solve · Verify · Family**.

## 2. The wire convention
A `RelSpec` travels as a **JSON string** on the wires (robust, saveable). Chain the editing
components (`Relate`/`Witness`/`Ground`) — each takes a `spec` string and returns a new
`spec` string (immutable; branch freely). `Solve` turns a spec into geometry (Breps) +
a `realization` string; `Verify` checks geometry back against the spec.

Start a spec from scratch with an empty one: feed the literal string `{}` (or
`{"schema_version":"0.0.1"}`) into the first `Relate`/`Witness`.

## 3. First experiment — the central claim (demo7, on the canvas)
Reproduce "RCC-8 can't tell a face from an edge contact, but the witnessed IR can":

1. `Relate`: spec=`{}`, src=`A`, dst=`B`, rcc8=`EC` → spec1.
2. `Witness`: spec=spec1, region_a=`A`, region_b=`B`, contact_dim=`face` → spec2.
   (Do the same for a second pair with `contact_dim=line` if you want the B–C edge case.)
3. `Solve`: spec=spec2, domain=`aabb_3d`, bounds=`[0,0,0,10,10,10]` → geometry (two boxes
   sharing a **face**) + realization.
4. `Verify`: spec=spec2, realization=realization → report shows `A-B dim=2` satisfied.
5. Now hand `Verify` a geometry where A and B meet only on an **edge** (move one box):
   the witnessed spec **rejects** it (`contact_dim_mismatch`); strip the witness (a spec with
   only `EC`) and it is **accepted** — the witnessed IR is strictly stronger.

`Family` (spec + a JSON list of grounding scenarios) yields several solutions that share the
invariant core and differ only where you left freedom — the decision-rights demonstration.

## 4. Lifting existing work
Feed any plugin's output (Kangaroo relaxed mesh, Wasp aggregation, hand-modeled Breps) into
`Lift` with matching `region_ids`. It observes RCC-8 relations + contact dimensions
(`epistemic=observed_tolerant`). Approve the ones you want as spec with `Witness`/set-invariant,
change a grounding, `Solve` a variant, `Verify`. That round-trip — geometry → relations →
re-synthesis → geometry-verify — is the experimental apparatus.

## 5. Headless check (no Rhino)
Everything except the Rhino geometry bridging is tested headlessly:
```
cd <repo> && . .venv/bin/activate
pytest -q                                   # 397 passing
PYTHONPATH=src python -m deixis.demos.demo1_contact_dimension
PYTHONPATH=src python -m deixis.demos.demo7_roundtrip
PYTHONPATH=src python -m deixis.field.scalar     # threshold = grounding (field -> rooms)
```

## 6. Known limits (by design)
- `Solve` domain = axis-aligned boxes (orthogonal). Not arbitrary geometry.
- `Lift` contact-dimension read from Rhino geometry is tolerance-dependent (observation, not
  authority). A 2D EC contact tops out at a *line*; `face` needs 3D.
- Two wire serializers exist (`io.serialize` canonical + `io.adapters` local) — see the
  adapters TODO; unify before any external interchange.
