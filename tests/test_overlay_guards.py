"""Guards for the honest non-semilattice structure of the overlay IR.

These tests exercise the four *additive* diagnostics in
:mod:`deixis.incidence.overlay`:

* :func:`intersection_closure_defects` — the realised signature family fails to
  be closed under ``∩``;
* :func:`meet_defects` — a poset pair has no greatest lower bound;
* :func:`covering_relation` — the Hasse diagram (transitive reduction) of the
  signature-inclusion order;
* :func:`check_overlay_axioms` — structural well-formedness of the overlap-atom
  algebra, and the regression guard against re-asserting a meet-semilattice.

Central point (a codex-flagged subtlety): a ``∩``-closure defect and a missing
meet are **not** the same thing. ``S ∩ T`` may be absent from the realised family
while the pair still has a unique greatest lower bound via a deeper realised
signature. The two are checked by two different APIs and can disagree. All exact
set / structural work; no floats, no geometry.
"""
from deixis.core.types import OverlayAtom
from deixis.incidence.overlay import (
    AxiomViolation,
    ClosureDefect,
    MeetDefect,
    _atom_id,
    build_overlay,
    check_overlay_axioms,
    covering_relation,
    intersection_closure_defects,
    meet,
    meet_defects,
    overlay_order,
)


# ================================================================ ∩-closure
def test_intersection_closure_defect_when_shared_sub_signature_missing():
    # {R1,R2} and {R1,R3} realised; their intersection {R1} is NOT realised.
    atoms = build_overlay(["R1", "R2", "R3"], [{"R1", "R2"}, {"R1", "R3"}])
    defects = intersection_closure_defects(atoms)
    assert len(defects) == 1
    d = defects[0]
    assert isinstance(d, ClosureDefect)
    assert {d.left, d.right} == {frozenset({"R1", "R2"}), frozenset({"R1", "R3"})}
    assert d.missing == frozenset({"R1"})


def test_intersection_closure_clean_when_family_is_closed():
    # A chain is closed under ∩ (the meet is the smaller signature, realised).
    atoms = build_overlay(["R1", "R2"], [{"R1"}, {"R1", "R2"}])
    assert intersection_closure_defects(atoms) == []


def test_intersection_closure_ignores_empty_intersection():
    # {R1} and {R2} share nothing; empty intersection is not an overlap atom,
    # so it is NOT reported as a defect.
    atoms = build_overlay(["R1", "R2"], [{"R1"}, {"R2"}])
    assert intersection_closure_defects(atoms) == []


# ================================================================ meet defects
def test_meet_defect_and_closure_defect_appear_separately():
    # The counterexample from the module docstring: {R1,R2} and {R1,R3} realised,
    # would-be meet {R1} absent. BOTH diagnostics fire, each on its own API.
    atoms = build_overlay(["R1", "R2", "R3"], [{"R1", "R2"}, {"R1", "R3"}])

    closure = intersection_closure_defects(atoms)
    meets = meet_defects(atoms)

    assert len(closure) == 1
    assert closure[0].missing == frozenset({"R1"})

    assert len(meets) == 1
    m = meets[0]
    assert isinstance(m, MeetDefect)
    assert {m.left, m.right} == {frozenset({"R1", "R2"}), frozenset({"R1", "R3"})}
    assert m.intersection == frozenset({"R1"})
    # No realised lower bound at all -> zero maximal lower bounds.
    assert m.maximal_lower_bounds == frozenset()


def test_closure_defect_without_meet_defect_shows_non_equivalence():
    # NON-EQUIVALENCE: {a,b,c} and {a,b,d} realised, plus {a}. Their signature
    # intersection {a,b} is NOT realised (a ∩-closure defect), yet the pair DOES
    # have a greatest lower bound: {a} is the unique lower realised signature (a
    # realised proper subset of {a,b}), so the meet exists and meet_defects is
    # empty for that pair.
    atoms = build_overlay(
        ["a", "b", "c", "d"],
        [{"a", "b", "c"}, {"a", "b", "d"}, {"a"}],
    )

    closure = intersection_closure_defects(atoms)
    assert len(closure) == 1
    assert closure[0].missing == frozenset({"a", "b"})

    # Meet exists via the deeper realised signature {a}: no meet defect anywhere.
    assert meet_defects(atoms) == []
    abc = next(x for x in atoms if x.members == frozenset({"a", "b", "c"}))
    abd = next(x for x in atoms if x.members == frozenset({"a", "b", "d"}))
    m = meet(atoms, abc, abd)
    assert m is not None and m.members == frozenset({"a"})


def test_meet_defect_reports_two_maximal_lower_bounds():
    # Two incomparable maximal lower bounds -> genuinely no greatest lower bound.
    # {a,b,x} and {a,b,y} realised; lower bounds {a} and {b} both realised and
    # incomparable; their join {a,b} not realised. Meet is ambiguous.
    atoms = build_overlay(
        ["a", "b", "x", "y"],
        [{"a", "b", "x"}, {"a", "b", "y"}, {"a"}, {"b"}],
    )
    meets = meet_defects(atoms)
    top_pair = [
        d for d in meets
        if {d.left, d.right} == {frozenset({"a", "b", "x"}), frozenset({"a", "b", "y"})}
    ]
    assert len(top_pair) == 1
    d = top_pair[0]
    assert d.intersection == frozenset({"a", "b"})
    assert d.maximal_lower_bounds == frozenset({frozenset({"a"}), frozenset({"b"})})


def test_meet_defects_clean_on_a_chain():
    atoms = build_overlay(["R1", "R2"], [{"R1"}, {"R1", "R2"}])
    assert meet_defects(atoms) == []


# ================================================================ Hasse / covering
def test_covering_relation_on_a_chain_is_the_transitive_reduction():
    atoms = build_overlay(
        ["R1", "R2", "R3"],
        [{"R1"}, {"R1", "R2"}, {"R1", "R2", "R3"}],
    )
    a1 = next(a for a in atoms if a.members == frozenset({"R1"}))
    a12 = next(a for a in atoms if a.members == frozenset({"R1", "R2"}))
    a123 = next(a for a in atoms if a.members == frozenset({"R1", "R2", "R3"}))

    cover = covering_relation(atoms)
    assert cover == {(a1.id, a12.id), (a12.id, a123.id)}
    # the transitive pair is in the full order but NOT in the Hasse diagram
    assert (a1.id, a123.id) in overlay_order(atoms)
    assert (a1.id, a123.id) not in cover
    # covering is a subset of the full order, and its closure recovers it
    assert cover <= overlay_order(atoms)


def test_covering_relation_diamond_skips_no_intermediate():
    # {a} < {a,b} < {a,b,c} and {a} < {a,c} < {a,b,c}: covers are the 4 edges,
    # NOT the two diagonal {a}<{a,b,c} style pairs.
    atoms = build_overlay(
        ["a", "b", "c"],
        [{"a"}, {"a", "b"}, {"a", "c"}, {"a", "b", "c"}],
    )
    ids = {frozenset(a.members): a.id for a in atoms}
    cover = covering_relation(atoms)
    assert cover == {
        (ids[frozenset({"a"})], ids[frozenset({"a", "b"})]),
        (ids[frozenset({"a"})], ids[frozenset({"a", "c"})]),
        (ids[frozenset({"a", "b"})], ids[frozenset({"a", "b", "c"})]),
        (ids[frozenset({"a", "c"})], ids[frozenset({"a", "b", "c"})]),
    }
    assert (ids[frozenset({"a"})], ids[frozenset({"a", "b", "c"})]) not in cover


def test_covering_relation_empty_for_antichain():
    atoms = build_overlay(["R1", "R2", "R3"], [{"R1", "R2"}, {"R1", "R3"}])
    assert covering_relation(atoms) == set()


# ================================================================ axiom guard
def test_check_axioms_clean_on_well_formed_build():
    atoms = build_overlay(
        ["R1", "R2", "R3"],
        [{"R1"}, {"R1", "R2"}, {"R1", "R2", "R3"}],
    )
    assert check_overlay_axioms(atoms) == []


def test_check_axioms_passes_non_semilattice_input_regression_guard():
    # A legitimate NON-meet-semilattice input must be well-formed: being a poset
    # without all meets is NOT an axiom violation. This guards against a
    # regression that silently re-asserts a semilattice.
    atoms = build_overlay(["R1", "R2", "R3"], [{"R1", "R2"}, {"R1", "R3"}])
    assert check_overlay_axioms(atoms) == []
    # ... while the honest diagnostics still report the missing meet.
    assert len(meet_defects(atoms)) == 1


def test_check_axioms_flags_duplicate_signature_exact_vs_cumulative():
    # Two atoms sharing a signature conflate the exact atom A_S with the
    # cumulative domain C_S = ⋂ Ri: exact atoms must be pairwise disjoint.
    sig = frozenset({"R1", "R2"})
    a = OverlayAtom(id=_atom_id(sig), members=sig, nonempty=True)
    b = OverlayAtom(id=_atom_id(sig), members=sig, nonempty=True)
    viols = check_overlay_axioms([a, b])
    codes = [v.code for v in viols]
    assert "duplicate_signature" in codes
    dup = next(v for v in viols if v.code == "duplicate_signature")
    assert dup.signature == sig


def test_check_axioms_flags_empty_signature_nonempty_predicate():
    empty = OverlayAtom(id=_atom_id(frozenset()), members=frozenset(), nonempty=True)
    viols = check_overlay_axioms([empty])
    assert [v.code for v in viols] == ["empty_signature"]


def test_check_axioms_flags_members_not_frozenset():
    bad = OverlayAtom(id="o:x", members={"R1", "R2"}, nonempty=True)  # plain set
    viols = check_overlay_axioms([bad])
    assert [v.code for v in viols] == ["members_not_frozenset"]


def test_check_axioms_flags_nonempty_not_bool():
    bad = OverlayAtom(id=_atom_id(frozenset({"R1"})),
                      members=frozenset({"R1"}), nonempty=1)  # int, not bool
    viols = check_overlay_axioms([bad])
    assert [v.code for v in viols] == ["nonempty_not_bool"]


def test_check_axioms_flags_id_not_canonical():
    sig = frozenset({"R1", "R2"})
    bad = OverlayAtom(id="o:not-canonical", members=sig, nonempty=True)
    viols = check_overlay_axioms([bad])
    assert [v.code for v in viols] == ["id_not_canonical"]


def test_axiom_violation_is_the_reported_type():
    empty = OverlayAtom(id=_atom_id(frozenset()), members=frozenset(), nonempty=True)
    viols = check_overlay_axioms([empty])
    assert all(isinstance(v, AxiomViolation) for v in viols)
