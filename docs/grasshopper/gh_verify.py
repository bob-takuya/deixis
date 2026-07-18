# r: z3-solver
# r: pyrsistent
# r: networkx
"""Deixis · Verify — reverse-verify geometry against the spec (exact + witness violations).
Inputs: spec(str), realization(str). Output: report(str)."""
import sys
_SRC = deixis_src if ("deixis_src" in globals() and deixis_src) else "~/deixis/src"
if _SRC not in sys.path: sys.path.insert(0, _SRC)
from deixis.io import adapters
report = adapters.verify(spec, realization)
