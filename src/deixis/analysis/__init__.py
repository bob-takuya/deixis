"""``deixis.analysis`` — *derived readings* over the M-field(b) flow layer.

This package is a **covering-domain extension** on top of :mod:`deixis.field.flow`.
It takes an already-built oriented cell complex plus a flow 1-cochain and maps it
onto a directed graph so that graph-theoretic *space-syntax* readings — most notably
**choice** (媒介中心性 / betweenness centrality) — can be computed from the flow field.

Nothing here changes the M0 contract or the field/flow semantics: it only *reads*
them. Flow magnitudes stay exact :class:`fractions.Fraction`; the only floats that
appear are the ones :mod:`networkx` itself returns from centrality (a ratio of path
counts), which is the library's own output, not a coordinate we compute.
"""

from deixis.analysis.flow_link import (
    flow_betweenness,
    flow_to_graph,
    sources_sinks,
)

__all__ = [
    "flow_to_graph",
    "flow_betweenness",
    "sources_sinks",
]
