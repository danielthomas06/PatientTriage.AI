"""FLACC: pain in a patient who cannot self-report.

Recorded case: a 6-month-old with a fever was asked "how bad is the pain,
nought to ten" and "does the pain spread to your jaw" -- a pre-verbal infant
can do neither. CTAS Appendix F has a real answer for this (a behavioural
observation scale, not self-report), and these tests exercise the actual
wiring, not just the scoring arithmetic.
"""

import pytest

from triage.core import Category
from triage.flacc import MINIMUM_SELF_REPORT_PAIN_AGE, SELF_REPORT_PAIN_CHECKS, flacc_score


def test_flacc_score_sums_the_five_categories():
    assert flacc_score(face=1, legs=1, activity=0, cry=2, consolability=1) == 5


def test_flacc_score_rejects_an_out_of_range_value():
    with pytest.raises(ValueError):
        flacc_score(face=3, legs=0, activity=0, cry=0, consolability=0)


def test_a_pre_verbal_infant_is_never_asked_to_self_report_pain():
    """The exact recorded case. Neither the numeric ladder nor the
    radiation question should ever reach the patient."""
    import serve
    serve.USE_MODEL = False
    serve.ENCOUNTERS.clear()
    e = serve.new_patient()
    e.set_identity({"name": "Baby", "age_months": "6", "sex": "F", "patient_id": ""})
    e.add_narrative("Baby is having high fever since yesterday.")
    for _ in range(8):
        s = e.next_step()
        if s.get("stopped") or not s["question"]:
            break
        assert s["check"] not in SELF_REPORT_PAIN_CHECKS
        for p in s["pending"]:
            assert p["check"] not in SELF_REPORT_PAIN_CHECKS or "FLACC" in p["actor"]
        e.answer(s["check"], "no")


def test_a_school_aged_child_still_gets_the_self_report_ladder():
    """The threshold must not swallow children who genuinely can self-report
    -- CTAS's own NRS population is 'school aged children and adolescents'."""
    import serve
    serve.USE_MODEL = False
    serve.ENCOUNTERS.clear()
    e = serve.new_patient()
    e.set_identity({"name": "Child", "age": "10", "sex": "F", "patient_id": ""})
    e.add_narrative("I've had a headache since this morning.")
    found_self_report_pain = False
    for _ in range(8):
        s = e.next_step()
        if s.get("stopped") or not s["question"]:
            break
        if s["check"] in SELF_REPORT_PAIN_CHECKS:
            found_self_report_pain = True
            break
        e.answer(s["check"], "no")
    assert found_self_report_pain


def test_recording_a_flacc_score_resolves_the_same_bands_as_self_report():
    """Same 0-10 scale, same severity bands -- just filled in by observation
    instead of a spoken number."""
    import serve
    serve.USE_MODEL = False
    serve.ENCOUNTERS.clear()
    e = serve.new_patient()
    e.set_identity({"name": "Baby", "age_months": "6", "sex": "F", "patient_id": ""})
    e.record_flacc({"face": 1, "legs": 1, "activity": 0, "cry": 2, "consolability": 1})
    answers = e.belief.answers
    assert answers["moderate_pain_central"].value == "true"    # 5 is 4-7
    assert answers["severe_pain_central"].value == "false"
    assert "severe_pain_peripheral" not in answers   # paediatric: no locality split


def test_paediatric_pain_is_never_split_by_locality():
    """The manual is explicit: 'the Paediatrics guidelines do not
    distinguish between central and peripheral pain' -- even for a leading
    complaint that would normally be peripheral for an adult."""
    import serve
    serve.USE_MODEL = False
    serve.ENCOUNTERS.clear()
    e = serve.new_patient()
    e.set_identity({"name": "Child", "age": "3", "sex": "M", "patient_id": ""})
    e.weights["extremity_injury"] = 0.6   # would be "peripheral" for an adult
    assert e._pain_locality() == "central"


def test_malformed_flacc_input_records_nothing():
    import serve
    serve.USE_MODEL = False
    serve.ENCOUNTERS.clear()
    e = serve.new_patient()
    e.record_flacc({"face": "not a number", "legs": 1, "activity": 0,
                     "cry": 0, "consolability": 0})
    assert not any("pain" in k for k in e.belief.answers)
