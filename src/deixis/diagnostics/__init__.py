"""Research-only structural diagnostics (auxiliary; NOT part of the verified core).

Everything under :mod:`deixis.diagnostics` is a *measurement instrument*, not a
decision procedure. These tools describe structural properties of a spec; they never
edit it, never assert semantic redundancy or numeric contradiction, and produce no
grounding/lock. See :mod:`deixis.diagnostics.dm` for the honesty caveats.
"""
from . import dm

__all__ = ["dm"]
