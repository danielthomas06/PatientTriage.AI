"""The CTAS pack, the cohort guard, and the extraction guards.

Everything here runs offline. Nothing calls a model.
"""

import pytest

from triage import Answer, BeliefState, Category, Evidence, decide, plausible_set
from triage.cohort import Cohort, Patient, resolve
from triage.extract import _denies, _leads, _verify
from triage.ollama import coerce, simplify, vocabulary
from triage.protocols.ctas import PROTOCOL, REASSESSMENT_MINUTES, TARGET_MINUTES


# --------------------------------------------------------------------------
# the pack
# --------------------------------------------------------------------------

def test_pack_is_internally_consistent():
    PROTOCOL.validate()
    assert len(PROTOCOL.branches) == 20
    assert PROTOCOL.vocabulary_size >= 55


def test_first_order_modifiers_are_shared_across_every_branch():
    """CTAS first-order modifiers apply to essentially every complaint. That
    sharing is what lets all branches be evaluated from one vector."""
    shared = set.intersection(
        *(set(b.discriminator_ids) for b in PROTOCOL.branches.values())
    )
    assert "shock" in shared
    assert "resp_distress_severe" in shared
    assert len(shared) >= 15


def test_the_cardiac_check_lives_only_on_chest_pain():
    """The entire thesis. A nurse who commits to abdominal_pain never asks this."""
    carriers = [
        b.id for b in PROTOCOL.branches.values()
        if "pain_radiating" in b.discriminator_ids
    ]
    assert carriers == ["chest_pain"]


def test_targets_are_ctas_not_mts():
    assert TARGET_MINUTES[Category.ORANGE] == 15      # MTS says 10
    assert REASSESSMENT_MINUTES[Category.YELLOW] == 30


# --------------------------------------------------------------------------
# the catch, end to end, deterministically
# --------------------------------------------------------------------------

def test_a_check_on_an_unselected_branch_still_sets_the_category():
    """68F with epigastric pain. abdominal_pain leads; chest_pain is second and
    would not have been worked by a nurse who committed to the leader."""
    belief = BeliefState(PROTOCOL)
    belief = belief.record("moderate_pain_central", Answer.TRUE,
                           Evidence("pain in my upper stomach", "speech"))
    weights = {"abdominal_pain": 0.80, "chest_pain": 0.20}
    plausible = plausible_set(weights)

    before = decide(belief, plausible)
    assert before.category is Category.YELLOW

    belief = belief.record("pain_radiating", Answer.TRUE,
                           Evidence("it goes up into my jaw a bit", "speech"))
    after = decide(belief, plausible)

    assert after.category is Category.ORANGE
    assert after.fired == "pain_radiating"
    assert after.fired_on == ("chest_pain",)


def test_dropping_the_second_branch_loses_the_catch():
    """The counterfactual: this is what committing to one branch costs."""
    belief = BeliefState(PROTOCOL).record(
        "pain_radiating", Answer.TRUE, Evidence("into my jaw", "speech")
    )
    only_abdo = plausible_set({"abdominal_pain": 0.80})
    assert decide(belief, only_abdo).category is not Category.ORANGE


# --------------------------------------------------------------------------
# cohort
# --------------------------------------------------------------------------

def test_unknown_age_withholds_scoring_rather_than_assuming_adult():
    a = resolve(Patient())
    assert a.cohort is Cohort.UNRESOLVED
    assert a.may_score_vitals is False
    assert "fail-safe" in " ".join(a.warnings)


@pytest.mark.parametrize(
    "patient,expected",
    [
        (Patient(age_days=10), Cohort.NEONATE),
        (Patient(age_months=8), Cohort.INFANT),
        (Patient(age_years=14), Cohort.CHILD),
        (Patient(age_years=30, pregnant=True), Cohort.OBSTETRIC),
        (Patient(age_years=40), Cohort.ADULT),
    ],
)
def test_cohort_resolution(patient, expected):
    assert resolve(patient).cohort is expected


def test_only_adults_get_a_vital_sign_score():
    assert resolve(Patient(age_years=40)).may_score_vitals
    for p in (Patient(age_years=6), Patient(age_years=30, pregnant=True), Patient()):
        assert not resolve(p).may_score_vitals


def test_beta_blockers_surface_the_falsely_reassuring_pulse():
    """The quiet failure: this patient cannot mount a tachycardia, so a normal
    heart rate means nothing. Nobody notices unless it is said out loud."""
    warnings = " ".join(resolve(Patient(age_years=81, rate_limiting_meds=True)).warnings)
    assert "FALSELY REASSURING" in warnings


def test_afebrile_sepsis_is_flagged_in_the_elderly():
    p = Patient(age_years=82, suspected_infection=True, temperature=37.2)
    assert "does not exclude sepsis" in " ".join(resolve(p).warnings)


def test_an_ill_looking_child_with_normal_vitals_is_not_reassurance():
    warnings = " ".join(resolve(Patient(age_years=3, looks_unwell=True)).warnings)
    assert "PRE-CARDIOPULMONARY ARREST" in warnings


# --------------------------------------------------------------------------
# extraction guards
# --------------------------------------------------------------------------

TRANSCRIPT = "I've had this pain in my upper stomach since six. I feel sick and I'm sweaty."


def test_quote_must_be_in_the_transcript():
    assert _verify("pain in my upper stomach", TRANSCRIPT)
    assert not _verify("patient reports epigastric discomfort", TRANSCRIPT)


def test_a_negative_needs_an_actual_denial():
    """A fabricated 'denies chest pain' is the worst output this system can
    produce, and a real quote elsewhere in the sentence does not license it."""
    assert not _denies("I feel sick and I'm sweaty")
    assert _denies("no, nothing like that")


def test_denial_check_is_coarse_and_we_know_where():
    """Documented limitation. It checks that a denial word is present, not that
    it scopes over the right thing -- so this passes, wrongly."""
    assert _denies("it's not going away")


def test_a_question_may_not_name_the_symptom_it_tests_for():
    d = PROTOCOL.discriminators["pain_radiating"]
    assert _leads("Where exactly do you feel the pain?", d) is None
    assert _leads("Does the pain spread to your jaw?", d) == "jaw"


# --------------------------------------------------------------------------
# local backend translation
# --------------------------------------------------------------------------

def test_unions_are_flattened_for_the_local_grammar():
    schema = {"properties": {"v": {"anyOf": [
        {"type": "string", "enum": ["severe"]}, {"type": "boolean"}]}}}
    out = simplify(schema)["properties"]["v"]
    assert out["type"] == "string"
    assert set(out["enum"]) == {"severe", "true", "false"}


def test_booleans_survive_the_round_trip():
    assert coerce("true") is True
    assert coerce("False") is False
    assert coerce("severe") == "severe"


def test_sensitive_checks_never_reach_the_local_vocabulary():
    """A kiosk must not ask about self-harm in a public waiting room."""
    assert "self_harm_risk" not in vocabulary(PROTOCOL)


# --------------------------------------------------------------------------
# the 8 branches added to reach 20 (collapse, vertigo, palpitations,
# allergic_reaction, rash, eye_problems, substance_misuse, vaginal_bleeding)
# --------------------------------------------------------------------------

def decide_on(branch_id: str, *positives: str) -> Category:
    belief = BeliefState(PROTOCOL)
    for cid in positives:
        belief = belief.record(cid, Answer.TRUE, Evidence("simulated", "staff"))
    return decide(belief, plausible_set({branch_id: 0.8})).category


def test_the_new_branches_are_reachable_and_validated():
    for b in ("collapse", "vertigo", "palpitations", "allergic_reaction",
              "rash", "eye_problems", "substance_misuse", "vaginal_bleeding"):
        assert b in PROTOCOL.branches


def test_syncope_with_no_warning_is_very_urgent():
    """Sourced: Sec 4.2 Selected Special Complaints, Level 2 --
    'Syncope/presyncope (no prodromal symptoms)'."""
    assert decide_on("collapse", "syncope_no_warning") is Category.ORANGE


def test_isolated_positional_vertigo_is_urgent_not_very_urgent():
    """Sourced: Sec 4.2, Level 3 -- 'Vertigo (positional, no other neuro
    symptoms)'. The contrast in the manual's own wording is the point: add a
    neuro symptom and it must escalate, which the next test checks."""
    assert decide_on("vertigo", "vertigo_positional_only") is Category.YELLOW


def test_vertigo_with_a_neuro_deficit_outranks_the_positional_case():
    assert decide_on("vertigo", "vertigo_positional_only",
                      "new_neuro_deficit") is Category.ORANGE


def test_palpitations_with_a_lethal_arrhythmia_history_is_very_urgent():
    """Sourced: Sec 4.2, Level 2 -- 'Palpitations/irregular heart beat
    (history of documented lethal)'."""
    assert decide_on("palpitations", "palpitations_lethal_history") is Category.ORANGE


def test_throat_swelling_in_an_allergic_reaction_is_very_urgent():
    assert decide_on("allergic_reaction", "throat_or_tongue_swelling") is Category.ORANGE


def test_widespread_hives_alone_is_only_urgent():
    assert decide_on("allergic_reaction", "widespread_hives") is Category.YELLOW


def test_non_blanching_rash_is_very_urgent():
    assert decide_on("rash", "non_blanching_rash") is Category.ORANGE


def test_chemical_eye_exposure_is_very_urgent():
    """Sourced: Sec 4.2, Level 2 -- 'Chemical exposure, eye'."""
    assert decide_on("eye_problems", "chemical_eye_exposure") is Category.ORANGE


def test_high_risk_ingestion_outranks_plain_withdrawal_signs():
    assert decide_on("substance_misuse", "high_risk_ingestion") is Category.ORANGE
    assert decide_on("substance_misuse", "withdrawal_signs") is Category.YELLOW


def test_prolapsed_cord_is_immediate():
    """Sourced: Sec 4.3.2, the manual's own late-pregnancy modifier table --
    'Presenting fetal parts or prolapsed cord' -> Level 1."""
    assert decide_on("vaginal_bleeding",
                      "prolapsed_cord_or_presenting_parts") is Category.RED


def test_third_trimester_bleeding_is_immediate():
    """Sourced: same table -- 'Vaginal bleeding 3rd trimester' -> Level 1."""
    assert decide_on("vaginal_bleeding", "bleeding_third_trimester") is Category.RED


def test_active_labour_frequency_sets_the_right_band():
    """Sourced: same table -- contractions <=2 min apart is Level 2,
    >2 min apart is Level 3. The manual's own escalation with frequency."""
    assert decide_on("vaginal_bleeding", "active_labour_frequent") is Category.ORANGE
    assert decide_on("vaginal_bleeding", "active_labour") is Category.YELLOW


def test_general_vaginal_bleeding_without_obstetric_findings_is_moderate():
    """Not sourced for this sub-case (see the AUTHORED comment in ctas.py) --
    confirms it lands at the intended Yellow tier rather than something
    stronger it was never meant to claim."""
    assert decide_on("vaginal_bleeding", "heavy_bleeding") is Category.YELLOW


def test_shared_first_order_modifiers_still_cover_all_20_branches():
    """The whole-vocabulary sharing property must survive adding 8 branches --
    if a new branch quietly failed to inherit the shared list, its patients
    would miss the RED/ORANGE life-threat catches every other branch gets
    for free."""
    shared = set.intersection(
        *(set(b.discriminator_ids) for b in PROTOCOL.branches.values())
    )
    assert len(PROTOCOL.branches) == 20
    assert "shock" in shared and "unconscious" in shared
