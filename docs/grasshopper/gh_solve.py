# r: z3-solver
# r: pyrsistent
# r: networkx
"""Deixis · Solve — relations -> exact geometry (axis-aligned boxes via Z3), fed back to Grasshopper.
Inputs: spec(str), domain(str "aabb_2d"|"aabb_3d"), bounds(list[float] [x0,y0,(z0,)x1,y1,(z1)]).
Outputs: realization(str), geometry(list of Breps), status(str)."""
import sys, json
_SRC = deixis_src if ("deixis_src" in globals() and deixis_src) else "/path/to/deixis/src"
if _SRC not in sys.path: sys.path.insert(0, _SRC)
from deixis.io import adapters
import Rhino.Geometry as rg

dom = domain or "aabb_2d"
_b = list(bounds) if bounds else ([0, 0, 20, 20] if dom == "aabb_2d" else [0, 0, 0, 20, 20, 20])
half = len(_b) // 2
lo_b, hi_b = _b[:half], _b[half:]
realization = adapters.solve(spec, dom, [lo_b, hi_b], None)

geometry, status = [], ""
try:
    real = adapters.realization_from_json(realization)
    status = getattr(real.status, "value", str(real.status))
    for box in real.boxes:
        lo = [float(c) for c in box.lo]; hi = [float(c) for c in box.hi]
        while len(lo) < 3: lo.append(0.0); hi.append(0.0 if len(hi) < 3 else hi[-1])
        if hi[2] <= lo[2]: hi[2] = lo[2] + 1e-6  # give 2D boxes a sliver of height to preview
        bb = rg.BoundingBox(rg.Point3d(lo[0], lo[1], lo[2]), rg.Point3d(hi[0], hi[1], hi[2]))
        geometry.append(bb.ToBrep())
except Exception as e:
    status = "error: %s" % e
