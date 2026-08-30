"""Age stratification, and the refusal to guess.

The Round 2 brief singles this out:

    "Vital sign thresholds and symptom weights differ significantly across
     pediatric, adult, and geriatric populations... Solutions that apply a
     single adult-calibrated scoring model across all age groups introduce
     silent safety risk."

Two directions of failure, and the second is the nastier one.

A normal respiratory rate for a two-year-old scores as deterioration on an adult
chart. That error is loud -- it over-triages, and someone notices.

The quiet one runs the other way. An eighty-year-old on beta-blockers CANNOT
mount a tachycardia, so their pulse reads normal while they are septic. Nothing
alarms. The score is reassuring precisely where it should not be. Same for
afebrile sepsis in the frail elderly, and for delirium presenting as the only
sign of serious illness.

CTAS handles paediatrics with one general rule rather than a table per age:
score by standard deviations from age-normal.

    >= 3 SD outside normal  ->  CTAS 1
       2 SD outside normal  ->  CTAS 2
       1 SD outside normal  ->  CTAS 3
          within normal     ->  CTAS 4 or 5, decided by history

And the manual's warning, which is the whole reason this module exists:

    "In the child who appears ill, vital signs in the normal range may indicate
     a PRE-CARDIOPULMONARY ARREST state."

A normal score in a patient who looks unwell is not reassurance.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from .core import Category


class Cohort(StrEnum):
    NEONATE = "neonate"        # <= 28 days
    INFANT = "infant"          # < 12 months
    CHILD = "child"            # < 16 years
    OBSTETRIC = "obstetric"    # pregnant, any age
    ADULT = "adult"
    UNRESOLVED = "unresolved"  # age not established -- a fail-safe, not a default


@dataclass(frozen=True, slots=True)
class Assessment:
    cohort: Cohort
    may_score_vitals: bool
    warnings: tuple[str, ...] = ()

    @property
    def is_paediatric(self) -> bool:
        return self.cohort in (Cohort.NEONATE, Cohort.INFANT, Cohort.CHILD)


@dataclass(frozen=True, slots=True)
class Patient:
    """What we know about who this is, as opposed to what is wrong with them."""

    age_years: float | None = None
    age_months: float | None = None
    age_days: float | None = None
    pregnant: bool = False
    frail: bool = False
    rate_limiting_meds: bool = False
    dementia: bool = False
    suspected_infection: bool = False
    temperature: float | None = None
    looks_unwell: bool = False

    @property
    def age_known(self) -> bool:
        return any(v is not None for v in (self.age_years, self.age_months, self.age_days))

    @property
    def years(self) -> float | None:
        if self.age_years is not None:
            return self.age_years
        if self.age_months is not None:
            return self.age_months / 12
        if self.age_days is not None:
            return self.age_days / 365
        return None

    @property
    def months(self) -> float | None:
        if self.age_months is not None:
            return self.age_months
        if self.age_days is not None:
            return self.age_days / 30
        if self.age_years is not None:
            return self.age_years * 12
        return None


# Adult physiological scores are validated in adults. Applying one anywhere else
# is a silent error, so the resolver returns may_score_vitals=False rather than
# producing a number nobody should trust.
def resolve(patient: Patient) -> Assessment:
    if not patient.age_known:
        return Assessment(
            cohort=Cohort.UNRESOLVED,
            may_score_vitals=False,
            warnings=(
                "Age not established -- adult thresholds withheld. Unknown cohort "
                "is a fail-safe trigger, not a licence to apply adult defaults.",
            ),
        )

    if patient.pregnant:
        return Assessment(
            cohort=Cohort.OBSTETRIC,
            may_score_vitals=False,
            warnings=("Adult physiological scores are not validated in pregnancy -- "
                      "use an obstetric chart and clinician judgement.",),
        )

    days = patient.age_days
    months = patient.months
    years = patient.years

    # Paediatric cohorts CAN now be scored -- against the age-banded table,
    # never against the adult one. `may_score_vitals` refers to the adult
    # score specifically, which remains inapplicable here.
    if days is not None and days <= 28:
        return Assessment(Cohort.NEONATE, False, _paediatric_warnings(patient, months))
    if months is not None and months < 12:
        return Assessment(Cohort.INFANT, False, _paediatric_warnings(patient, months))
    if years is not None and years < 16:
        return Assessment(Cohort.CHILD, False, _paediatric_warnings(patient, months))

    return Assessment(Cohort.ADULT, True, _adult_warnings(patient))


def _paediatric_warnings(patient: Patient, months: float | None) -> tuple[str, ...]:
    out = [
        "Paediatric cohort -- adult vital-sign thresholds do not apply. Scored "
        "against age-normal ranges from the CTAS appendix instead.",
    ]
    if months is not None and months < 3:
        out.append(
            "Under three months: immature immune system, raised sepsis risk. Fever "
            "alone escalates."
        )
    elif months is not None and months < 24:
        out.append("Under two years: raised bacteraemia risk.")
    if patient.looks_unwell:
        out.append(
            "Child appears ill: vital signs in the normal range may indicate a "
            "PRE-CARDIOPULMONARY ARREST state. A normal score is not reassurance."
        )
    return tuple(out)


def _adult_warnings(patient: Patient) -> tuple[str, ...]:
    """The quiet failures. Each one makes a normally reassuring reading unreliable
    for this particular patient, and each is invisible unless surfaced."""
    out: list[str] = []
    years = patient.years or 0

    if patient.rate_limiting_meds:
        out.append(
            "Heart rate may be FALSELY REASSURING -- patient is on rate-limiting "
            "medication and cannot mount a tachycardia. Weight respiratory rate and "
            "conscious level instead."
        )
    if years >= 75 and patient.suspected_infection and (patient.temperature or 99) < 38:
        out.append(
            "Older patient with suspected infection but no fever. Absence of fever "
            "does not exclude sepsis at this age."
        )
    if patient.dementia:
        out.append(
            "Conscious-level scoring is unreliable as an absolute here. Establish "
            "the patient's baseline and look for CHANGE from it."
        )
    if years >= 75 or patient.frail:
        out.append(
            "Frail or elderly: presentations are frequently atypical. Non-specific "
            "decline may be the only sign of serious illness."
        )
    return tuple(out)


# The CTAS paediatric rule, kept as data because it is the thing worth citing:
# score by how far outside age-normal a reading falls, not by a fixed threshold.
SD_TO_CATEGORY = {3: 1, 2: 2, 1: 3, 0: None}

PAEDIATRIC_RANGES_SOURCE = (
    "CTAS manual Appendix G, itself sourced from Fleming S, Thompson M, Stevens R, "
    "Heneghan C. 'Normal ranges of heart rate and respiratory rate in children from "
    "birth to 18 years of age: a systematic review of observational studies.' "
    "The Lancet 2011; 377(9770): 1011-1019."
)


def paediatric_vital_category(
    patient: Patient, *, resp_rate: float | None = None, pulse: float | None = None
):
    """Score a child's vitals against age-normal ranges.

    Returns (Category | None, reasons). None means within normal for age -- which,
    per the appendix, is NOT reassurance in a child who looks unwell.
    """
    from .protocols.paediatric_vitals import assess

    months = patient.months
    if months is None:
        return None, ["age not established -- cannot select an age-normal range"]

    category, why = assess(months, resp_rate=resp_rate, pulse=pulse)
    if category is None and patient.looks_unwell:
        why.append(
            "BUT the child appears ill: normal vitals may indicate a "
            "pre-cardiopulmonary arrest state. Do not treat this as reassurance."
        )
        # The docstring above has always said this case is not reassurance;
        # returning None here anyway -- silently falling back to whatever the
        # belief-driven category happened to be, which can be the LEAST
        # urgent band if nothing else was recorded -- said one thing and did
        # another. A gestalt "looks unwell" is exactly the CTAS "Critical
        # Look" category of judgement (Sec 4.1: too "ill or injured" to
        # complete a formal triage process), which the manual treats at
        # Orange or above; Orange is the floor here rather than guessing
        # higher than the observation actually supports.
        return Category.ORANGE, why
    return category, why
