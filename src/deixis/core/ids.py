"""Stable identity, exact-rational helpers, and provenance for the Deixis IR.

Design constraints (from the adversarially-verified spec):
- All geometry coordinates are exact: use ``fractions.Fraction`` (never float in the core).
- IDs are *stable*: they anchor identity across grounding/reground and across the
  region-view / field-view of the same cell complex. The whole IR shares one id space.
- Provenance is first-class and append-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Union
import itertools

Number = Union[int, Fraction]


def F(x: Any) -> Fraction:
    """Coerce to exact Fraction. Reject float silently-lossy inputs by converting via str."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        # Convert through the shortest decimal string to avoid binary-float noise.
        return Fraction(repr(x)).limit_denominator(10**12) if x != int(x) else Fraction(int(x))
    if isinstance(x, str):
        return Fraction(x)
    raise TypeError(f"cannot coerce {type(x)!r} to Fraction")


def frac_to_json(fr: Fraction) -> dict:
    return {"num": fr.numerator, "den": fr.denominator}


def frac_from_json(d: Any) -> Fraction:
    if isinstance(d, dict):
        return Fraction(d["num"], d["den"])
    return Fraction(d)  # tolerate "1/3" or int


class IdGen:
    """Deterministic counter-based id generator (no randomness -> reproducible ids)."""

    def __init__(self, prefix: str = "n"):
        self._prefix = prefix
        self._c = itertools.count()

    def fresh(self, kind: str = "") -> str:
        n = next(self._c)
        return f"{self._prefix}:{kind}:{n}" if kind else f"{self._prefix}:{n}"


@dataclass(frozen=True)
class Provenance:
    """Append-only provenance record (PROV-like Entity/Activity/Agent, minimal)."""
    origin: str          # e.g. "asserted", "lift", "solve", "threshold-ground"
    actor: str = "designer"
    activity: str = ""   # what produced/changed this
    inputs: tuple[str, ...] = ()   # ids of inputs
    note: str = ""

    def to_json(self) -> dict:
        return {
            "origin": self.origin, "actor": self.actor, "activity": self.activity,
            "inputs": list(self.inputs), "note": self.note,
        }
