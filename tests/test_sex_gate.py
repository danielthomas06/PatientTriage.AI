"""A male patient must never be asked, or shown as pending, an obstetric or
gynaecological check.

This is the one genuine exclusion in the pack -- everywhere else, a weak
signal only gets deprioritised, never dropped, because the cost of wrongly
excluding a true positive is a missed danger signal. That reasoning doesn't
transfer to sex: it's an explicit recorded fact, not a noisy branch weight,
and the failure mode of asking anyway isn't hypothetical -- see the
docstring on OBSTETRIC_GYNAE_ONLY in protocols/ctas.py for the actual case
that motivated this (a headache patient asked "how many weeks pregnant" as
the FIRST follow-up question, because the branch-weight discount alone
wasn't steep enough to suppress a Red-tier, low-prior obstetric check).
"""

import serve
from triage.protocols.ctas import OBSTETRIC_GYNAE_ONLY


def encounter(sex, text="I've had a headache since this morning and I feel a bit off."):
    serve.USE_MODEL = False
    serve.ENCOUNTERS.clear()
    e = serve.new_patient()
    e.set_identity({"name": "Test", "age": "34", "sex": sex, "patient_id": ""})
    e.add_narrative(text)
    return e


def drive(e, turns=10):
    """Answer 'no' to everything and collect every question and pending item
    the whole interview ever surfaces."""
    questions, pending_checks = [], set()
    for _ in range(turns):
        s = e.next_step()
        if s.get("stopped") or not s["question"]:
            break
        questions.append(s["question"])
        pending_checks.update(p["check"] for p in s["pending"])
        e.answer(s["check"], "no")
    return questions, pending_checks


def test_a_male_patient_is_never_asked_an_obstetric_check():
    e = encounter("M")
    s = e.next_step()
    assert s["check"] not in OBSTETRIC_GYNAE_ONLY
    assert "pregnan" not in (s["question"] or "").lower()
    assert "bleeding" not in (s["question"] or "").lower()


def test_no_obstetric_check_surfaces_anywhere_for_a_male_patient_across_the_interview():
    """Covers the pending panel too, not just the headline question -- an
    obstetric check ranked second or third would otherwise still leak
    through as 'also worth getting'."""
    e = encounter("M")
    questions, pending_checks = drive(e)
    assert not (pending_checks & OBSTETRIC_GYNAE_ONLY)
    for q in questions:
        assert "pregnan" not in q.lower() and "weeks" not in q.lower()


def test_a_female_patient_is_not_affected():
    """The exclusion must not become a general 'never ask this' switch --
    confirms the same branch is still reachable for the cohort it applies to."""
    e = encounter("F")
    s = e.next_step()
    assert s["check"] in OBSTETRIC_GYNAE_ONLY


def test_unrecorded_sex_is_not_treated_as_male():
    """Only an explicit 'M' excludes. Blank/unknown must still ask -- this is
    a fact-based exclusion, not a default, matching the same fail-safe
    reasoning cohort.resolve() already applies to an unknown age."""
    e = encounter("")
    s = e.next_step()
    assert s["check"] in OBSTETRIC_GYNAE_ONLY


def test_the_exclusion_does_not_touch_the_category_decision():
    """Scope check: this only stops the interview loop from proactively
    asking. It must not narrow plausible_set() or decide() -- a record
    lookup or direct observation must still be able to set these checks and
    drive the category, the same as any other check."""
    e = encounter("M")
    e.observe("bleeding_third_trimester", serve.Answer.TRUE, "seen by staff")
    from triage.core import Category
    assert e.decision().category is Category.RED


# --------------------------------------------------------------------------
# age -- the gap sex alone left open
# --------------------------------------------------------------------------

def test_an_infant_is_never_asked_an_obstetric_check():
    """Recorded case: 6 months old, sex F, 'baby is having a high fever' --
    the reasoning step asked about vaginal bleeding, justified as ruling out
    pregnancy complications. Sex=F correctly didn't exclude it; nothing was
    checking age at all."""
    e = encounter("F", text="Baby is having high fever since yesterday.")
    e.age = 0.5   # 6 months
    questions, pending_checks = drive(e)
    assert not (pending_checks & OBSTETRIC_GYNAE_ONLY)
    for q in questions:
        assert "pregnan" not in q.lower() and "vaginal" not in q.lower()


def test_a_nine_year_old_still_is_not_excluded():
    """The age cutoff must not become a blanket young-patient exclusion --
    MINIMUM_PLAUSIBLE_PREGNANCY_AGE is deliberately low so it only fires
    where pregnancy is genuinely not a routine consideration."""
    from triage.protocols.ctas import excludes_obstetric_gynae
    assert not excludes_obstetric_gynae("F", 9)
    assert not excludes_obstetric_gynae("F", 16)


def test_unknown_age_is_not_treated_as_too_young():
    """Same fail-safe direction as everywhere else: an unrecorded age must
    not silently exclude anything."""
    from triage.protocols.ctas import excludes_obstetric_gynae
    assert not excludes_obstetric_gynae("F", None)


def test_age_is_described_in_words_for_the_model_not_a_bare_decimal():
    """'Patient age: 0.5' is exactly the ambiguous phrasing a model reasoned
    past when asking an infant about pregnancy. Confirms the second,
    independent half of the fix actually produces something unambiguous."""
    from triage.extract import _describe_age
    assert "month" in _describe_age(0.5)
    assert "infant" in _describe_age(0.5)
