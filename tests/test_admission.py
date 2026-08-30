"""Accepting a recommendation now means something: the patient gets a bed.

Ten beds for the demo (triage.monitor.BED_COUNT). Accept is idempotent and
never mutates the audit trail into something a second click can duplicate.
"""

import serve


def fresh(name="Test Patient", age=45):
    serve.USE_MODEL = False
    serve.ENCOUNTERS.clear()
    serve.board.patients.clear()
    serve.BOARD_RECORDS.clear()
    e = serve.new_patient()
    e.set_identity({"name": name, "age": str(age), "sex": "F", "patient_id": ""})
    return e


def test_accept_admits_the_patient_with_a_bed():
    e = fresh()
    e.add_narrative("Worst headache of my life, came on suddenly an hour ago.")
    assert e.state()["admitted"] is False

    e.accept("rn.k.mensah")
    st = e.state()
    assert st["admitted"] is True
    assert st["bed"] == 1
    assert serve.board.patients[e.ref].current == e.decision().category


def test_accept_is_idempotent():
    """A second accept must not hand out a second bed for the same patient."""
    e = fresh()
    e.add_narrative("Worst headache of my life, came on suddenly an hour ago.")
    e.accept("rn.k.mensah")
    first_bed = e.state()["bed"]
    e.accept("rn.k.mensah")
    assert e.state()["bed"] == first_bed
    assert serve.board.free_beds() == 9   # only one bed actually consumed


def test_board_records_a_full_snapshot_for_the_click_through():
    e = fresh(name="Margaret Doyle", age=68)
    e.add_narrative("Worst headache of my life, came on suddenly an hour ago.")
    e.accept("rn.k.mensah")
    rec = serve.BOARD_RECORDS[e.ref]
    assert rec["name"] == "Margaret Doyle"
    assert rec["age"] == 68
    assert len(rec["transcript"]) == 1
    assert rec["checks"]


def test_eleventh_patient_queues_without_a_bed():
    serve.USE_MODEL = False
    serve.ENCOUNTERS.clear()
    serve.board.patients.clear()
    serve.BOARD_RECORDS.clear()
    from triage.monitor import BED_COUNT
    # Each new_patient() call is a genuinely distinct, concurrently-valid
    # session -- no manual ref juggling needed, which is the entire point of
    # this rework: ten of these coexisting is the normal case now, not a
    # workaround.
    for i in range(BED_COUNT):
        e = serve.new_patient()
        e.set_identity({"name": f"Filler {i}", "age": "40", "sex": "F", "patient_id": ""})
        e.add_narrative("Ankle pain after a fall, mild swelling.")
        e.accept("rn.k.mensah")

    assert serve.board.free_beds() == 0
    overflow = serve.new_patient()
    overflow.set_identity({"name": "Overflow Patient", "age": "50", "sex": "M",
                            "patient_id": ""})
    overflow.add_narrative("Worst headache of my life, came on suddenly an hour ago.")
    overflow.accept("rn.k.mensah")
    assert overflow.state()["admitted"] is True
    assert overflow.state()["bed"] is None   # queued, not turned away


def test_board_state_reports_capacity():
    e = fresh()
    e.add_narrative("Worst headache of my life, came on suddenly an hour ago.")
    e.accept("rn.k.mensah")
    bs = serve.board_state()
    assert bs["capacity"] == 10
    assert bs["free_beds"] == 9
    assert bs["patients"][0]["name"] == "Test Patient"
    assert bs["patients"][0]["bed"] == 1
