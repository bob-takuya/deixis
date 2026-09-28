# r: z3-solver
# r: pyrsistent
# r: networkx
"""Deixis · Relate — add/update a qualitative relation. Inputs: spec(str), src(str), dst(str),
rcc8(str e.g. "EC" or "DC,EC"), status(str "delegated"|"grounded"|"invariant"). Output: spec(str)."""
import sys, json
_SRC = deixis_src if ("deixis_src" in globals() and deixis_src) else "/path/to/deixis/src"
if _SRC not in sys.path: sys.path.insert(0, _SRC)
from deixis.io import adapters
names = [s.strip() for s in (rcc8 or "").replace(" ", ",").split(",") if s.strip()]
spec = adapters.relate(spec, src, dst, names, (status or "delegated"))
