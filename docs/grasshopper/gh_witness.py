# r: z3-solver
# r: pyrsistent
# r: networkx
"""Deixis · Witness — assert a contact witness (face/line/point) between two regions BEFORE geometry.
Inputs: spec(str), region_a(str), region_b(str), contact_dim(str "face"|"line"|"point"). Output: spec(str)."""
import sys
_SRC = deixis_src if ("deixis_src" in globals() and deixis_src) else "/path/to/deixis/src"
if _SRC not in sys.path: sys.path.insert(0, _SRC)
from deixis.io import adapters
spec = adapters.witness(spec, region_a, region_b, (contact_dim or "face"))
