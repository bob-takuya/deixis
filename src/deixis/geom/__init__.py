"""M-geom: transformation-group *views* over a realized geometry.

This package holds read-only projections of a :class:`~deixis.core.types.Realization`
onto the invariants of a chosen transformation group (Euclidean … homeomorphism).
It never edits geometry — see :mod:`deixis.geom.group_view`.
"""
from .group_view import (
    TransformGroup,
    LossReport,
    project,
    canonical_key,
    groups_chain,
)

__all__ = [
    "TransformGroup",
    "LossReport",
    "project",
    "canonical_key",
    "groups_chain",
]
