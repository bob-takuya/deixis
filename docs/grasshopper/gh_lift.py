# r: z3-solver
# r: pyrsistent
# r: networkx
"""Deixis · Lift — observe existing Grasshopper geometry (Kangaroo/Wasp/manual) as a relation spec.
Honest: tolerance-aware OBSERVATION (epistemic=observed_tolerant), never auto-invariant.
Inputs: geometry(list of Breps/Meshes), region_ids(list of str), tolerance(float). Output: spec(str)."""
import sys, json
_SRC = deixis_src if ("deixis_src" in globals() and deixis_src) else "~/deixis/src"
if _SRC not in sys.path: sys.path.insert(0, _SRC)
from deixis.io import adapters

boxes = []
geoms = list(geometry) if geometry else []
for i, g in enumerate(geoms):
    bb = g.GetBoundingBox(True)
    rid = region_ids[i] if (region_ids and i < len(region_ids)) else ("R%d" % i)
    boxes.append({"region_id": rid,
                  "lo": [bb.Min.X, bb.Min.Y, bb.Min.Z],
                  "hi": [bb.Max.X, bb.Max.Y, bb.Max.Z]})
rids = list(region_ids) if region_ids else [b["region_id"] for b in boxes]
spec = adapters.lift(json.dumps(boxes), rids, (tolerance or 0))
