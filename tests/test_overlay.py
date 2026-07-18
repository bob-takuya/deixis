"""Tests for the overlay poset IR (deixis.incidence.overlay).

Core M5 properties exercised here:

* signature-based construction: one non-empty atom per distinct realised
  membership signature sigma(x)={i | x in Ri};
* the signature-inclusion partial order and occupancy comparison agree;
* a program region crossing physical rooms (circulation & daylight & exhibition)
  is a single independently-addressable atom;
* the *honest* structural limit: overlay atoms are a poset + non-empty predicate,
  NOT a meet-semilattice — a concrete counterexample where a realised pair has no
  meet inside the realised set;
* forbidden/empty diagnosis in both directions.

All exact set / structural work; no floats, no geometry."""
from deixis.core.types import OverlayAtom
from deixis.incidence import overlay as O
from deixis.incidence.overlay import (
    OverlayViolation,
    build_overlay,
    diagnose_forbidden_empty,
    is_meet_semilattice,
    meet,
    occupancy_compare,
    overlay_order,
    realized_signatures,
)


# ---------------------------------------------------------------- build_overlay
def test_build_overlay_one_atom_per_distinct_nonempty_signature():
    # Three regions; sample occupancies (some repeated) covering several signatures.
    memberships = [
        {"R1"},
        {"R1", "R2"},
        {"R1", "R2"},        # duplicate signature -> still one atom
        {"R2", "R3"},
        {"R1", "R2", "R3"},
    ]
    atoms = build_overlay(["R1", "R2", "R3"], memberships)
    sigs = {a.members for a in atoms}
    assert sigs == {
        frozenset({"R1"}),
        frozenset({"R1", "R2"}),
        frozenset({"R2", "R3"}),
        frozenset({"R1", "R2", "R3"}),
    }
    # members are frozensets; nonempty True; atoms are frozen OverlayAtom.
    for a in atoms:
        assert isinstance(a, OverlayAtom)
        assert isinstance(a.members, frozenset)
        assert a.nonempty is True


def test_build_overlay_skips_empty_signature():
    atoms = build_overlay(["R1", "R2"], [set(), {"R1"}, frozenset()])
    assert {a.members for a in atoms} == {frozenset({"R1"})}


def test_build_overlay_accepts_mapping_of_points():
    memberships = {"p1": {"R1"}, "p2": {"R1", "R2"}, "p3": {"R1", "R2"}}
    atoms = build_overlay(["R1", "R2"], memberships)
    assert {a.members for a in atoms} == {frozenset({"R1"}), frozenset({"R1", "R2"})}


def test_build_overlay_deterministic_id_and_order():
    a1 = build_overlay(["R1", "R2"], [{"R2", "R1"}, {"R1"}])
    a2 = build_overlay(["R1", "R2"], [{"R1"}, {"R1", "R2"}])
    # Same signatures regardless of input order -> identical ids and ordering.
    assert [x.id for x in a1] == [x.id for x in a2]
    # ordered by (len, sorted members): {R1} before {R1,R2}
    assert [set(x.members) for x in a1] == [{"R1"}, {"R1", "R2"}]
    # id derived from sorted signature (length-prefixed, injective encoding)
    assert a1[0].id == "o:2~R1"
    assert a1[1].id == "o:2~R1;2~R2"


def test_atom_ids_are_injective_across_delimiter_collisions():
    # {"a;b"} and {"a","b"} must NOT collapse to the same id.
    one = build_overlay(["a;b"], [{"a;b"}])
    two = build_overlay(["a", "b"], [{"a", "b"}])
    assert one[0].id != two[0].id


def test_build_overlay_rejects_unknown_region():
    import pytest
    with pytest.raises(ValueError):
        build_overlay(["R1", "R2"], [{"R1", "R9"}])


# ---------------------------------------------------------------- program crossing rooms
def _program_overlay():
    """A program region 'exhibition' that spans two physical rooms while also
    demanding daylight and sitting on a circulation spine. The triple overlap
    circulation & daylight & exhibition is one atom, addressable on its own even
    though no single room bounds it."""
    regions = ["room_a", "room_b", "circulation", "daylight", "exhibition"]
    memberships = [
        {"room_a"},                                   # plain room A interior
        {"room_b"},                                   # plain room B interior
        {"room_a", "exhibition"},                     # exhibition part inside A
        {"room_b", "exhibition"},                     # exhibition part inside B
        {"circulation", "exhibition"},                # exhibition on the spine
        {"circulation", "daylight", "exhibition"},    # the cross-room program atom
        {"daylight"},                                 # a lit spot in no program
    ]
    return regions, build_overlay(regions, memberships)


def test_program_region_crosses_physical_rooms_as_independent_atom():
    _regions, atoms = _program_overlay()
    cross = frozenset({"circulation", "daylight", "exhibition"})
    hit = [a for a in atoms if a.members == cross]
    assert len(hit) == 1
    atom = hit[0]
    # It is independent: it is NOT a sub-part of any single physical room atom.
    room_atoms = [a for a in atoms if a.members & {"room_a", "room_b"}]
    for r in room_atoms:
        assert occupancy_compare(atom, r) == "incomparable"
    assert atom.id == "o:11~circulation;8~daylight;10~exhibition"


# ---------------------------------------------------------------- inclusion order
def test_overlay_order_matches_occupancy_compare():
    atoms = build_overlay(
        ["R1", "R2", "R3"],
        [{"R1"}, {"R1", "R2"}, {"R1", "R2", "R3"}],
    )
    by_sig = {frozenset(k): a for k, a in
              [(("R1",), atoms[0]),
               (("R1", "R2"), atoms[1]),
               (("R1", "R2", "R3"), atoms[2])]}
    a1 = by_sig[frozenset({"R1"})]
    a12 = by_sig[frozenset({"R1", "R2"})]
    a123 = by_sig[frozenset({"R1", "R2", "R3"})]

    order = overlay_order(atoms)
    # chain: {R1} < {R1,R2} < {R1,R2,R3} -> all 3 proper-subset pairs present.
    assert (a1.id, a12.id) in order
    assert (a12.id, a123.id) in order
    assert (a1.id, a123.id) in order       # transitive pair included
    # strict: no reflexive pairs
    assert (a1.id, a1.id) not in order

    # overlay_order agrees exactly with occupancy_compare == 'sub'
    for a in atoms:
        for b in atoms:
            if a.id != b.id:
                assert ((a.id, b.id) in order) == (occupancy_compare(a, b) == "sub")

    assert occupancy_compare(a123, a1) == "sup"
    assert occupancy_compare(a1, a123) == "sub"


def test_occupancy_compare_incomparable_for_crossing_signatures():
    atoms = build_overlay(["R1", "R2", "R3"], [{"R1", "R2"}, {"R1", "R3"}])
    a12, a13 = atoms[0], atoms[1]
    assert occupancy_compare(a12, a13) == "incomparable"
    assert occupancy_compare(a13, a12) == "incomparable"
    assert overlay_order(atoms) == set()


# ---------------------------------------------------------------- NOT a meet-semilattice
def test_overlay_is_not_a_meet_semilattice_counterexample():
    # {R1,R2} and {R1,R3} are both realised, but their would-be meet {R1} is
    # NOT realised (nobody stands in R1 alone). There is thus no greatest lower
    # bound inside the realised set: no meet exists.
    atoms = build_overlay(["R1", "R2", "R3"], [{"R1", "R2"}, {"R1", "R3"}])
    a12, a13 = atoms[0], atoms[1]
    assert frozenset({"R1"}) not in realized_signatures(atoms)
    assert meet(atoms, a12, a13) is None
    assert is_meet_semilattice(atoms) is False


def test_overlay_meet_exists_when_shared_sub_signature_is_realised():
    # Add the {R1} atom: now the meet of {R1,R2} and {R1,R3} exists and is {R1}.
    atoms = build_overlay(["R1", "R2", "R3"], [{"R1"}, {"R1", "R2"}, {"R1", "R3"}])
    a1 = next(a for a in atoms if a.members == frozenset({"R1"}))
    a12 = next(a for a in atoms if a.members == frozenset({"R1", "R2"}))
    a13 = next(a for a in atoms if a.members == frozenset({"R1", "R3"}))
    m = meet(atoms, a12, a13)
    assert m is not None and m.id == a1.id
    # meet with a comparable atom is the smaller one (idempotent-ish along a chain)
    assert meet(atoms, a1, a12).id == a1.id
    # ... but this 3-atom set is still not a full semilattice: meet({R1,R2},{R1,R3})
    # exists here, yet a set can fail elsewhere — checked in the counterexample test.


def test_is_meet_semilattice_ignores_explicitly_empty_atoms():
    # A chain {R1} < {R1,R2} is a meet-semilattice; an extra atom marked empty
    # (nonempty=False) is not a poset element and must not flip the answer.
    real = build_overlay(["R1", "R2"], [{"R1"}, {"R1", "R2"}])
    assert is_meet_semilattice(real) is True
    empty = OverlayAtom(id="o:x", members=frozenset({"R1", "R2", "R3"}), nonempty=False)
    assert is_meet_semilattice(list(real) + [empty]) is True


# ---------------------------------------------------------------- forbidden / empty
def test_diagnose_forbidden_nonempty():
    atoms = build_overlay(["R1", "R2"], [{"R1"}, {"R1", "R2"}])
    forbidden = {frozenset({"R1", "R2"})}   # R1 and R2 must never overlap
    viols = diagnose_forbidden_empty(atoms, forbidden)
    assert len(viols) == 1
    v = viols[0]
    assert isinstance(v, OverlayViolation)
    assert v.code == "forbidden_nonempty"
    assert v.signature == frozenset({"R1", "R2"})


def test_diagnose_required_nonempty_but_empty():
    # An atom explicitly recorded as absent (nonempty=False) whose signature is
    # not forbidden -> flagged as should-be-non-empty.
    absent = OverlayAtom(id="o:R1^R2", members=frozenset({"R1", "R2"}), nonempty=False)
    viols = diagnose_forbidden_empty([absent], forbidden=set())
    assert [v.code for v in viols] == ["required_nonempty_but_empty"]


def test_diagnose_absent_and_forbidden_is_clean():
    # An overlap that is both forbidden AND absent satisfies the policy: no viol.
    absent = OverlayAtom(id="o:R1^R2", members=frozenset({"R1", "R2"}), nonempty=False)
    viols = diagnose_forbidden_empty([absent], forbidden={frozenset({"R1", "R2"})})
    assert viols == []


def test_diagnose_clean_when_nonempty_and_not_forbidden():
    atoms = build_overlay(["R1", "R2"], [{"R1"}, {"R1", "R2"}])
    assert diagnose_forbidden_empty(atoms, forbidden=set()) == []
