"""Paediatric age-banded vital signs.

The table was extracted mechanically from the CTAS manual rather than retyped,
so these tests check the extraction as much as the lookup: if a column ever
shifted, the structural invariants below break.
"""

import pytest

from triage.cohort import Patient, paediatric_vital_category, resolve
from triage.core import Category
from triage.protocols import paediatric_vitals as pv


# ---------------------------------------------------------------- the tables

def test_both_tables_are_complete():
    assert len(pv.RESPIRATORY_RATE) == 25
    assert len(pv.HEART_RATE) == 25


@pytest.mark.parametrize("table", [pv.RESPIRATORY_RATE, pv.HEART_RATE])
def test_boundaries_are_monotonic_within_every_row(table):
    """Six boundaries carving seven bands. Any column shift breaks the ordering."""
    for months, bounds in table:
        assert len(bounds) == 6, months
        assert list(bounds) == sorted(bounds), f"{months}m: {bounds}"


@pytest.mark.parametrize("table", [pv.RESPIRATORY_RATE, pv.HEART_RATE])
def test_ages_ascend_and_span_birth_to_eighteen(table):
    ages = [m for m, _ in table]
    assert ages == sorted(ages)
    assert ages[0] == 0 and ages[-1] == 216


def test_rates_fall_with_age():
    """Physiology: both heart rate and respiratory rate decline through childhood.
    Heart rate peaks around 1-3 months first, so compare from there."""
    hr = dict(pv.HEART_RATE)
    rr = dict(pv.RESPIRATORY_RATE)
    assert hr[3][3] > hr[24][3] > hr[120][3] > hr[216][3]
    assert rr[0][3] > rr[24][3] > rr[120][3] > rr[216][3]


# ------------------------------------------------- independent cross-checks
# Fleming et al. 2011 -- the source CTAS cites -- reports a median heart rate of
# 127 at birth and a median respiratory rate of 44. Both must read as normal, and
# both sit exactly at the centre of the published band. If the extraction had
# slipped a column these would fail.

def test_fleming_median_heart_rate_at_birth_is_normal():
    assert pv.heart_rate(127, 0) is None
    b = dict(pv.HEART_RATE)[0]
    assert (b[2] + b[3]) / 2 == 127


def test_fleming_median_respiratory_rate_at_birth_is_normal():
    assert pv.respiratory_rate(44, 0) is None
    b = dict(pv.RESPIRATORY_RATE)[0]
    assert (b[2] + b[3]) / 2 == 44


# ---------------------------------------------------------------- the lookup

def test_bands_step_out_from_normal():
    b = dict(pv.HEART_RATE)[0]           # (79, 95, 111, 143, 159, 175)
    assert pv.heart_rate(127, 0) is None            # normal
    assert pv.heart_rate(150, 0) is Category.YELLOW  # 1 band out
    assert pv.heart_rate(165, 0) is Category.ORANGE  # 2 bands
    assert pv.heart_rate(200, 0) is Category.RED     # 3+
    assert pv.heart_rate(70, 0) is Category.RED      # and downward too


def test_an_adult_rate_in_a_toddler_is_not_normal():
    """The loud failure the brief describes: a respiratory rate of 16 is fine in
    an adult and abnormally LOW in a two-year-old."""
    assert pv.respiratory_rate(16, 24) is not None


def test_a_toddler_rate_would_alarm_an_adult_chart():
    """And the reverse: 30 breaths a minute is normal at two years old."""
    assert pv.respiratory_rate(30, 24) is None


def test_ages_between_rows_resolve_toward_urgency():
    """A four-month-old sits between published rows whose normal bands differ.
    The appendix says 'when in doubt - triage up', so we take the more urgent
    of the two bracketing rows rather than picking one."""
    at_3m = pv.heart_rate(155, 3)
    at_6m = pv.heart_rate(155, 6)
    at_4m = pv.heart_rate(155, 4)
    assert at_3m is None and at_6m is Category.YELLOW
    assert at_4m is Category.YELLOW


def test_beyond_eighteen_refuses_rather_than_reading_the_last_row():
    with pytest.raises(ValueError, match="beyond the paediatric table"):
        pv.heart_rate(80, 240)


def test_most_urgent_vital_wins():
    cat, why = pv.assess(14, resp_rate=55, pulse=185)
    assert cat is Category.RED
    assert len(why) == 2


# ---------------------------------------------------------------- the cohort

def test_paediatric_patients_are_now_scored_not_withheld():
    cat, why = paediatric_vital_category(Patient(age_months=14), resp_rate=30, pulse=140)
    assert cat is Category.YELLOW
    assert "not yet transcribed" not in " ".join(why).lower()


def test_a_child_is_never_scored_on_the_adult_chart():
    assert resolve(Patient(age_years=6)).may_score_vitals is False


def test_normal_vitals_in_an_ill_looking_child_are_not_reassurance():
    """Normal vitals alone would return (None, ...) -- withheld rather than a
    false reassurance. But 'not reassurance' has to mean something: an
    earlier version of this function said that in its own docstring and then
    returned None anyway, which a caller with no other findings (a bare
    vitals check, nothing else recorded yet) would read as nothing to floor
    the category with -- silently defaulting to the LEAST urgent band despite
    the appendix's own warning. Orange is the floor now, not just the text."""
    cat, why = paediatric_vital_category(
        Patient(age_months=14, looks_unwell=True), resp_rate=30, pulse=125
    )
    assert cat is Category.ORANGE
    assert "pre-cardiopulmonary arrest" in " ".join(why).lower()


def test_normal_vitals_in_a_well_looking_child_are_genuinely_unremarkable():
    """The contrast case: without looks_unwell, normal-for-age vitals must
    NOT be escalated -- otherwise every well child with unremarkable vitals
    would float up to Orange, which defeats the point of scoring at all."""
    cat, why = paediatric_vital_category(
        Patient(age_months=14, looks_unwell=False), resp_rate=30, pulse=125
    )
    assert cat is None


def test_unknown_age_cannot_select_a_range():
    cat, why = paediatric_vital_category(Patient(), resp_rate=30)
    assert cat is None
    assert "age not established" in " ".join(why)
