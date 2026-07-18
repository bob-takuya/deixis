"""Tests for demo1 — the superiority-of-spatial-description core.

Asserts the three central claims:

  (1) The FACE / LINE / POINT specs carry the *identical* RCC-8 mask (EC) yet
      their witnesses are distinguishable (RCC-8 alone cannot separate them).
  (2) Each spec solves to exact-rational AABBs whose realized geometry meets that
      spec's contact-dimension requirement (reverse-verify passes).
  (3) In the RCC-8-only spec (witness stripped) a FACE requirement is lost: an
      edge-contact geometry is accepted at the RCC-8 level but rejected by the
      witnessed verify.

All coordinates are exact Fraction; RCC-8 classification is structural/exact.
"""
from fractions import Fraction

from deixis.core import rcc8
from deixis.core.types import Box, ContactDim, Realizability
from deixis.incidence.witness import (
    collapses_to_same_rcc,
    witnesses_distinguishable,
)
from deixis.demos.demo1_contact_dimension import (
    DOMAIN,
    build_specs,
    realized_contact_dim,
    realized_rcc8_name,
    rcc_only_verify,
    run,
    strip_witnesses,
    verify,
)

_EC = rcc8.mask("EC")


# ------------------------------------------------------------------ build_specs
def test_build_specs_shapes():
    spec_face, spec_edge, spec_point = build_specs()
    assert spec_face.witnesses[0].intended_contact_dimension == int(ContactDim.FACE)
    assert spec_edge.witnesses[0].intended_contact_dimension == int(ContactDim.LINE)
    assert spec_point.witnesses[0].intended_contact_dimension == int(ContactDim.POINT)
    # exactly two regions A,B and one EC constraint each
    for spec in (spec_face, spec_edge, spec_point):
        assert {r.id for r in spec.regions} == {"A", "B"}
        assert len(spec.constraints) == 1
        assert len(spec.witnesses) == 1


# ------------------------------------------------------------------ claim (1)
def test_claim1_same_rcc_mask_but_distinguishable_witnesses():
    specs = build_specs()
    # every spec's every constraint is exactly EC -> identical RCC-8 masks
    masks = {c.rcc8_mask for spec in specs for c in spec.constraints}
    assert masks == {_EC}
    assert all(rcc8.names(m) == ["EC"] for m in masks)

    witnesses = [spec.witnesses[0] for spec in specs]
    # RCC-8 collapses all three onto one relation ...
    assert collapses_to_same_rcc(*witnesses) is True
    # ... yet they remain distinct as witnesses (different contact dimension).
    assert witnesses_distinguishable(*witnesses) is True
    assert len({w.intended_contact_dimension for w in witnesses}) == 3


# ------------------------------------------------------------------ claim (2)
def test_claim2_each_spec_solves_and_verifies():
    spec_face, spec_edge, spec_point = build_specs()
    expected = {
        id(spec_face): int(ContactDim.FACE),
        id(spec_edge): int(ContactDim.LINE),
        id(spec_point): int(ContactDim.POINT),
    }
    for spec in (spec_face, spec_edge, spec_point):
        real = run(spec)
        assert real.status == Realizability.REALIZED_IN_D
        # exact-rational coordinates (no floats leaked from the solver)
        for bx in real.boxes:
            for v in bx.lo + bx.hi:
                assert isinstance(v, Fraction)
        vr = verify(spec, real)
        assert vr.ok is True, vr
        # the single witness check confirms EC + the intended dimension
        (chk,) = vr.checks
        assert chk.realized_rcc8 == "EC"
        assert chk.realized_dim == expected[id(spec)] == chk.intended


# ------------------------------------------------------------------ claim (3)
def test_claim3_rcc_only_cannot_distinguish_face_from_edge():
    spec_face, spec_edge, _ = build_specs()

    edge_real = run(spec_edge)
    assert edge_real.status == Realizability.REALIZED_IN_D
    # the edge geometry really is a line/edge contact (contact dim 1), and EC
    a = next(b for b in edge_real.boxes if b.region_id == "A")
    b = next(b for b in edge_real.boxes if b.region_id == "B")
    assert realized_rcc8_name(a, b) == "EC"
    assert realized_contact_dim(a, b) == int(ContactDim.LINE)

    # RCC-8-only view of the FACE spec: witness stripped, only "A EC B" remains.
    spec_face_rcc = strip_witnesses(spec_face)
    assert spec_face_rcc.witnesses == ()

    # The RCC-8-only check ACCEPTS the edge geometry (it is 'EC') ...
    assert rcc_only_verify(spec_face_rcc, edge_real) is True
    # ... but the witnessed FACE requirement REJECTS it. That gap is the point.
    vr = verify(spec_face, edge_real)
    assert vr.ok is False
    (chk,) = vr.checks
    assert chk.intended == int(ContactDim.FACE)
    assert chk.realized_dim == int(ContactDim.LINE)
    assert chk.passed is False


def test_face_geometry_passes_its_own_witnessed_verify():
    spec_face, *_ = build_specs()
    face_real = run(spec_face)
    a = next(b for b in face_real.boxes if b.region_id == "A")
    b = next(b for b in face_real.boxes if b.region_id == "B")
    # a genuine face contact in 3D: exactly two axes overlap, one axis touches
    assert realized_contact_dim(a, b) == int(ContactDim.FACE)
    assert verify(spec_face, face_real).ok is True


# ------------------------------------------------------------------ classifier
def test_realized_rcc8_classifier_exact():
    F = Fraction
    # face share in 3D: overlap on x,y ; touch on z at z==1
    a = Box("A", (F(0), F(0), F(0)), (F(1), F(1), F(1)))
    b = Box("B", (F(0), F(0), F(1)), (F(1), F(1), F(2)))
    assert realized_rcc8_name(a, b) == "EC"
    assert realized_contact_dim(a, b) == 2  # FACE

    # edge share: touch on y and z, overlap on x
    c = Box("B", (F(0), F(1), F(1)), (F(1), F(2), F(2)))
    assert realized_rcc8_name(a, c) == "EC"
    assert realized_contact_dim(a, c) == 1  # LINE

    # corner touch: touch on all three axes
    d = Box("B", (F(1), F(1), F(1)), (F(2), F(2), F(2)))
    assert realized_rcc8_name(a, d) == "EC"
    assert realized_contact_dim(a, d) == 0  # POINT

    # separated -> DC, no contact dimension
    e = Box("B", (F(5), F(5), F(5)), (F(6), F(6), F(6)))
    assert realized_rcc8_name(a, e) == "DC"
    assert realized_contact_dim(a, e) is None

    # equal -> EQ
    assert realized_rcc8_name(a, Box("B", a.lo, a.hi)) == "EQ"

    # a strictly inside b -> NTPP ; proper tangential subset -> TPP
    big = Box("B", (F(-1), F(-1), F(-1)), (F(2), F(2), F(2)))
    assert realized_rcc8_name(a, big) == "NTPP"
    tang = Box("B", (F(0), F(0), F(0)), (F(2), F(2), F(2)))
    assert realized_rcc8_name(a, tang) == "TPP"
    assert realized_rcc8_name(tang, a) == "TPPi"
