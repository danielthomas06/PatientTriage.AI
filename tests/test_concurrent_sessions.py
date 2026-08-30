"""Multiple kiosks, multiple patients, at the same time.

Before this, ENCOUNTERS had exactly one slot ("live") and every route
called current() to get it -- a second kiosk didn't get a second patient,
it got the SAME one, mid-interview. These tests exercise the actual
concurrency property: independent sessions that don't share state, and
locking that serializes a slow request against ITSELF without blocking
anyone else's.
"""

import threading
import time

import pytest

import serve
from triage.core import Category


def setup_function(_):
    serve.USE_MODEL = False
    serve.ENCOUNTERS.clear()
    serve.board.patients.clear()
    serve.BOARD_RECORDS.clear()


def test_two_patients_get_independent_sessions():
    a = serve.new_patient()
    b = serve.new_patient()
    assert a.ref != b.ref

    a.set_identity({"name": "Patient A", "age": "30", "sex": "F", "patient_id": ""})
    a.add_narrative("I've had a headache since this morning.")

    b.set_identity({"name": "Patient B", "age": "50", "sex": "M", "patient_id": ""})

    # Nothing typed for A leaked into B, and vice versa.
    assert b.name == "Patient B"
    assert b.transcript == []
    assert a.transcript == ["I've had a headache since this morning."]


def test_a_ref_survives_lookup_by_the_session_helper():
    e = serve.new_patient()
    with serve.session(e.ref) as looked_up:
        assert looked_up is e


def test_an_unknown_ref_raises_no_such_patient_not_a_crash():
    """The honest failure for an in-memory-only session directory: say so,
    don't silently fall back to someone else's encounter (the old
    current()'s only option) and don't crash the request."""
    with pytest.raises(serve.NoSuchPatient):
        with serve.session("no-such-ref-ever-issued"):
            pass


def test_reset_abandons_the_old_ref_rather_than_reusing_it():
    """'New patient' at a kiosk must not let anything about the last patient
    carry into the next one -- a fresh Encounter, old ref gone entirely."""
    old = serve.new_patient()
    old.set_identity({"name": "Old Patient", "age": "60", "sex": "F", "patient_id": ""})
    old_ref = old.ref

    serve.ENCOUNTERS.pop(old_ref, None)
    new = serve.new_patient()

    assert new.ref != old_ref
    assert new.name == ""
    assert old_ref not in serve.ENCOUNTERS


def test_a_slow_patient_does_not_block_a_different_patients_request():
    """The actual point of per-encounter locking. Patient A holds their own
    lock for a simulated slow model call; Patient B's request, on a
    DIFFERENT encounter, must not wait for it."""
    a = serve.new_patient()
    b = serve.new_patient()

    a_started = threading.Event()
    a_may_finish = threading.Event()

    def slow_request_for_a():
        with a.lock:
            a_started.set()
            a_may_finish.wait(timeout=5)

    t = threading.Thread(target=slow_request_for_a)
    t.start()
    assert a_started.wait(timeout=2), "patient A's slow request never started"

    # While A is still "mid-model-call", B must be immediately answerable.
    t0 = time.time()
    with serve.session(b.ref) as enc:
        enc.set_identity({"name": "B", "age": "20", "sex": "F", "patient_id": ""})
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"patient B waited {elapsed:.2f}s on patient A's lock"

    a_may_finish.set()
    t.join(timeout=5)


def test_two_requests_for_the_same_patient_do_serialize():
    """The other half of the property: locking is per-patient, not absent --
    two concurrent requests for the SAME encounter must still not interleave
    their writes."""
    e = serve.new_patient()
    order = []

    def first():
        with e.lock:
            order.append("first-start")
            time.sleep(0.1)
            order.append("first-end")

    def second():
        time.sleep(0.02)   # ensure `first` gets there first
        with e.lock:
            order.append("second-start")
            order.append("second-end")

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert order == ["first-start", "first-end", "second-start", "second-end"]


def test_accepting_two_different_patients_concurrently_never_collides_on_a_bed():
    """accept() mutates the shared board from inside a per-encounter lock --
    confirms the internal GLOBAL_LOCK actually protects that, not just that
    it compiles. Two patients accepted 'at the same time' must land on two
    different beds, never double-book one."""
    patients = []
    for i in range(5):
        e = serve.new_patient()
        e.set_identity({"name": f"P{i}", "age": "40", "sex": "F", "patient_id": ""})
        e.add_narrative("Worst headache of my life, came on suddenly an hour ago.")
        patients.append(e)

    def accept(enc):
        enc.accept("rn.k.mensah")

    threads = [threading.Thread(target=accept, args=(p,)) for p in patients]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    beds = [serve.board.patients[p.ref].bed for p in patients]
    assert len(beds) == len(set(beds)), f"duplicate bed assignment: {beds}"
    assert sorted(beds) == [1, 2, 3, 4, 5]


def test_queue_rows_lists_every_in_progress_session_critical_first():
    calm = serve.new_patient()
    calm.set_identity({"name": "Calm", "age": "30", "sex": "F", "patient_id": ""})
    calm.add_narrative("I twisted my ankle, mild swelling.")

    critical = serve.new_patient()
    critical.set_identity({"name": "Critical", "age": "40", "sex": "M", "patient_id": ""})
    critical.add_narrative("Worst headache of my life, came on suddenly an hour ago.")

    rows = serve.queue_rows()
    assert {r["ref"] for r in rows} == {calm.ref, critical.ref}

    by_ref = {r["ref"]: r for r in rows}
    assert by_ref[critical.ref]["critical"] is True
    assert by_ref[calm.ref]["critical"] is False
    # Critical sorts first regardless of arrival order.
    assert rows[0]["ref"] == critical.ref


def test_a_nurses_own_walk_in_session_never_appears_in_the_intake_queue():
    """Real UX bug found by actually loading the nurse page: opening the
    dashboard auto-creates a blank walk-in Encounter for that tab, and it
    showed up in its own 'not yet reviewed' queue as 'Unidentified' --
    confusing, since the nurse IS the one reviewing it, by definition. Kiosk
    and nurse-created sessions need to be distinguishable."""
    kiosk_patient = serve.new_patient(origin="kiosk")
    kiosk_patient.set_identity({"name": "Kiosk Patient", "age": "30",
                                 "sex": "F", "patient_id": ""})

    nurse_walk_in = serve.new_patient(origin="nurse")
    # Deliberately left blank, exactly like a freshly-opened dashboard tab.

    refs = {r["ref"] for r in serve.queue_rows()}
    assert kiosk_patient.ref in refs
    assert nurse_walk_in.ref not in refs


def test_new_patient_defaults_to_kiosk_origin():
    """The default matters: every existing test in this file calls
    new_patient() with no argument and expects kiosk-queue visibility."""
    e = serve.new_patient()
    assert e.origin == "kiosk"


def test_reset_always_creates_a_nurse_origin_session():
    """/api/reset is the nurse dashboard's own 'New patient' button --
    kiosk.js never calls it. The session it hands back must not show up in
    the intake queue either."""
    old = serve.new_patient(origin="nurse")
    serve.ENCOUNTERS.pop(old.ref, None)
    replacement = serve.new_patient(origin="nurse")
    assert replacement.origin == "nurse"
    assert replacement.ref not in {r["ref"] for r in serve.queue_rows()}
