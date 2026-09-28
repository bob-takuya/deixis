# r: z3-solver
# r: pyrsistent
# r: networkx
"""Deixis · Ground — fix a delegated freedom NOW (records who/why). This is the decision-rights act.
Inputs: spec(str), target_id(str), fixed(str), actor(str), rationale(str). Output: spec(str)."""
import sys
_SRC = deixis_src if ("deixis_src" in globals() and deixis_src) else "/path/to/deixis/src"
if _SRC not in sys.path: sys.path.insert(0, _SRC)
from deixis.io import adapters
spec = adapters.ground(spec, target_id, fixed, (actor or "designer"), (rationale or ""))
