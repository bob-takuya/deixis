# r: z3-solver
# r: pyrsistent
# r: networkx
"""Deixis · Family — one un-grounded spec + several grounding scenarios -> comparable solutions.
Inputs: spec(str), scenarios(str = JSON list of scenarios, each a list of decision dicts). Output: realizations(list of str)."""
import sys, json
_SRC = deixis_src if ("deixis_src" in globals() and deixis_src) else "/path/to/deixis/src"
if _SRC not in sys.path: sys.path.insert(0, _SRC)
from deixis.io import adapters
scen = json.loads(scenarios) if scenarios else []
_out = adapters.family(spec, scen)             # returns JSON array (string)
realizations = json.loads(_out) if isinstance(_out, str) else _out
