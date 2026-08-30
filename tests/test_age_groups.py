"""Age-appropriate scoring, wired into the live app.

Two gaps existed before this: paediatric vitals were never fed into the
decision (only classified and warned about), and NEWS2 was computed and
shown but never actually floored the category for anyone, adult included.
Both are exercised here against a live Encounter, not just the underlying
library functions -- the bug was in the wiring, not the math.
"""

import serve
from triage.core import Category


def fresh():
    serve.USE_MODEL = False
    serve.ENCOUNTERS.clear()
    serve.board.patients.clear()
    serve.BOARD_RECORDS.clear()
    return serve.new_patient()


def test_a_toddlers_normal_pulse_is_not_scored_as_adult_shock():
    """The reproduction that found the bug: pulse 130 is a normal resting
    rate for a 2-year-old and adult-thresholded shock (pulse > 120) turned it
    into a false Immediate."""
    e = fresh()
    e.set_identity({"name": "Toddler", "age": "2", "sex": "M", "patient_id": ""})
    e.record_vitals({"pulse": "130", "systolic_bp": "95", "spo2": "98", "alert": True})
    assert "shock" not in e.belief.answers
    assert e.decision().category is not Category.RED


def test_age_in_months_resolves_the_infant_cohort():
    e = fresh()
    e.set_identity({"age_months": "4", "sex": "F", "patient_id": ""})
    from triage.cohort import Cohort, resolve
    assert resolve(e._patient()).cohort is Cohort.INFANT
    assert abs(e.age - 4 / 12) < 1e-9


def test_paediatric_vitals_floor_the_decision():
    """A child scored 1+ SD outside age-normal must escalate even with
    nothing else recorded -- this is the actual wiring, not just the table
    lookup (which triage.protocols.paediatric_vitals already tested)."""
    e = fresh()
    e.set_identity({"name": "Child", "age": "2.5", "sex": "M", "patient_id": ""})
    e.record_vitals({"respiratory_rate": "24", "pulse": "110", "alert": True})
    d = e.decision()
    assert d.from_vitals is True
    assert d.category < Category.BLUE


def test_normal_paediatric_vitals_plus_looks_unwell_escalates():
    """The manual's own warning, actually enforced: normal-for-age vitals in
    a child who looks ill must not fall back to the least urgent band."""
    e = fresh()
    e.set_identity({"name": "Child", "age": "2.5", "sex": "F", "patient_id": ""})
    e.record_vitals({"respiratory_rate": "25", "pulse": "100", "alert": True,
                      "looks_unwell": True})
    d = e.decision()
    assert d.from_vitals is True
    assert d.category is Category.ORANGE
    assert any("pre-cardiopulmonary arrest" in r for r in e.paediatric_reasons)


def test_normal_paediatric_vitals_without_looks_unwell_do_not_escalate():
    e = fresh()
    e.set_identity({"name": "Child", "age": "2.5", "sex": "F", "patient_id": ""})
    e.record_vitals({"respiratory_rate": "25", "pulse": "100", "alert": True})
    d = e.decision()
    assert d.from_vitals is False


def test_adult_news2_floors_the_category_with_nothing_else_recorded():
    """Below every individual CTAS first-order threshold, so ONLY the
    combined NEWS2 score can be driving this."""
    e = fresh()
    e.set_identity({"name": "Adult", "age": "60", "sex": "M", "patient_id": ""})
    e.record_vitals({"respiratory_rate": "22", "spo2": "93", "systolic_bp": "105",
                      "pulse": "112", "temperature": "38.2", "alert": True})
    d = e.decision()
    assert d.from_vitals is True
    assert d.category is Category.ORANGE
    assert e.state()["news2"]["band"] == "high"


def test_adult_vitals_thresholds_still_apply_to_adults():
    """The age gate must not become a blanket 'vitals never fire' switch --
    a genuinely shocked adult must still be caught directly."""
    e = fresh()
    e.set_identity({"name": "Adult", "age": "40", "sex": "F", "patient_id": ""})
    e.record_vitals({"pulse": "130", "systolic_bp": "85", "alert": True})
    assert e.belief.answers.get("shock") is not None
    assert e.belief.answers["shock"].value == "true"
