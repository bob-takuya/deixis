# Deixis for Grasshopper — shared bootstrap (paste at the top of every component)
# Rhino 8 "Script" component, language = Python 3 (CPython). The lines beginning with
# "# r:" are Rhino 8 pip requirements: on first run Rhino installs them into its CPython.
# r: z3-solver
# r: pyrsistent
# r: networkx
import sys, os, json
# Point this at your checkout's src/. Optionally expose an input `deixis_src` (str) to override.
_SRC = deixis_src if ("deixis_src" in globals() and deixis_src) else "/path/to/deixis/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from deixis.io import adapters
# All spec/realization values travel between components as JSON *strings*.
