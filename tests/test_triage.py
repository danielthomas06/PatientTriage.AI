"""Tests for the deterministic core.

The one that matters is test_cross_branch_catch: it is the entire pitch,
expressed as an assertion.
"""

import pytest

from triage import (
    Answer, BeliefState, Category, Evidence, NotApplicable, Source, Vitals,
    acuity_distribution, bayes_risk, decide, effective_categories, loss,
    plausible_set, rank, score, should_stop, value_of, walk_in_baseline,
)

# These tests assert on specific discriminator ids, so they name the pack that
# has them rather than taking whatever `triage.PROTOCOL` currently points at.
from triage.charts import PROTOCOL


# --------------------------------------------------------------- vital signs

def test_all_normal_scores_zero():
    v = Vitals(respiratory_rate=16, spo2=98, on_oxygen=False, systolic_bp=120,
               pulse=70, alert=True, temperature=36.8)
    assert score(v).total == 0
    assert score(v).band == "routine"


def test_known_scoring_case():
    # RR 26 (+3), SpO2 91 (+3), on oxygen (+2), SBP 95 (+2), pulse 115 (+2),
    # not alert (+3), temp 39.5 (+2)
    v = Vitals(respiratory_rate=26, spo2=91, on_oxygen=True, systolic_bp=95,
               pulse=115, alert=False, temperature=39.5)
    s = score(v)
    assert s.total == 17
    assert s.band == "high"


def test_single_parameter_three_is_caught():
    """The rule humans skip: a lone 3 on an otherwise well patient."""
    v = Vitals(respiratory_rate=26, spo2=98, on_oxygen=False, systolic_bp=120,
               pulse=70, alert=True, temperature=36.8)
    s = score(v)
    assert s.total == 3
    assert s.any_parameter_is_three
    assert s.band == "low-medium"


def test_refuses_children_and_pregnancy():
    base = dict(respiratory_rate=16, spo2=98, on_oxygen=False, systolic_bp=120,
                pulse=70, alert=True, temperature=36.8)
    with pytest.raises(NotApplicable):
        score(Vitals(**base, age_years=7))
    with pytest.raises(NotApplicable):
        score(Vitals(**base, pregnant=True))


def test_refuses_partial_vitals():
    with pytest.raises(ValueError):
        score(Vitals(respiratory_rate=16, spo2=98))


# ------------------------------------------------------------------- belief

def test_unknown_is_not_false():
    b = BeliefState(PROTOCOL)
    assert b.p_true("cardiac_pain") == PROTOCOL.discriminators["cardiac_pain"].prior
    assert b.p_true("cardiac_pain") > 0.0

    b = b.record("cardiac_pain", Answer.FALSE)
    assert b.p_true("cardiac_pain") == 0.0


def test_belief_is_immutable():
    a = BeliefState(PROTOCOL)
    b = a.record("severe_pain", Answer.TRUE)
    assert a.answers == {}
    assert b.answers == {"severe_pain": Answer.TRUE}


def test_evidence_is_retained():
    b = BeliefState(PROTOCOL).record(
        "cardiac_pain", Answer.TRUE, Evidence("into my jaw", "speech")
    )
    assert "into my jaw" in b.trace()[0]


# ------------------------------------------------------------------- engine

def test_effective_category_takes_most_urgent_across_branches():
    cats = effective_categories(PROTOCOL, frozenset({"chest_pain", "abdominal_pain"}))
    assert cats["cardiac_pain"] is Category.ORANGE
    assert cats["airway_compromise"] is Category.RED


def test_plausible_set_is_deliberately_loose():
    weights = {"chest_pain": 0.25, "abdominal_pain": 0.45, "collapse": 0.01}
    assert plausible_set(weights) == frozenset({"chest_pain", "abdominal_pain"})


def test_decide_uses_only_confirmed_positives():
    plausible = frozenset({"abdominal_pain"})
    b = BeliefState(PROTOCOL)
    assert decide(b, plausible).category is Category.BLUE

    b = b.record("moderate_pain", Answer.TRUE)
    d = decide(b, plausible)
    assert d.category is Category.YELLOW
    assert d.fired == "moderate_pain"


def test_more_urgent_discriminator_wins():
    plausible = frozenset({"chest_pain"})
    b = (BeliefState(PROTOCOL)
         .record("moderate_pain", Answer.TRUE)
         .record("shock", Answer.TRUE))
    assert decide(b, plausible).category is Category.RED


def test_vitals_can_raise_but_are_reported_separately():
    plausible = frozenset({"unwell_adult"})
    b = BeliefState(PROTOCOL).record("mild_pain", Answer.TRUE)
    # RR 22 (+2), SpO2 94 (+1), SBP 105 (+1), pulse 115 (+2) = 6
    v = score(Vitals(respiratory_rate=22, spo2=94, on_oxygen=False,
                     systolic_bp=105, pulse=115, alert=True, temperature=37.0))
    assert v.total == 6
    assert v.band == "medium"
    d = decide(b, plausible, vitals=v)
    assert d.category is Category.ORANGE
    assert d.from_vitals


def test_acuity_distribution_is_a_distribution():
    plausible = frozenset(PROTOCOL.branches)
    dist = acuity_distribution(BeliefState(PROTOCOL), plausible)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in dist.values())


def test_confirmed_red_collapses_the_distribution():
    plausible = frozenset(PROTOCOL.branches)
    b = BeliefState(PROTOCOL).record("shock", Answer.TRUE)
    dist = acuity_distribution(b, plausible)
    assert dist[Category.RED] == pytest.approx(1.0)


# ------------------------------------------------------------- THE PITCH

def test_cross_branch_catch():
    """A danger signal on a branch nobody selected still fires.

    68F with epigastric pain. A nurse commits to Abdominal Pain, so the
    cardiac discriminators are never reachable and she stays Urgent. Keeping
    Chest Pain in the plausible set escalates her on the same evidence.
    """
    committed = frozenset({"abdominal_pain"})
    parallel = frozenset({"abdominal_pain", "chest_pain", "unwell_adult", "vomiting"})

    b = (BeliefState(PROTOCOL)
         .record("moderate_pain", Answer.TRUE, Evidence("pain in my upper stomach", "speech"))
         .record("cardiac_pain", Answer.TRUE, Evidence("a bit, yeah. Into my jaw", "speech")))

    assert decide(b, committed).category is Category.YELLOW
    assert decide(b, parallel).category is Category.ORANGE
    assert decide(b, parallel).fired == "cardiac_pain"
    assert "chest_pain" in decide(b, parallel).fired_on


def test_committed_branch_cannot_even_ask_the_question():
    """Worse than getting it wrong: the question is not on the menu."""
    committed = frozenset({"abdominal_pain"})
    b = BeliefState(PROTOCOL)
    assert "cardiac_pain" not in {c.discriminator_id for c in rank(b, committed)}

    parallel = frozenset({"abdominal_pain", "chest_pain"})
    assert "cardiac_pain" in {c.discriminator_id for c in rank(b, parallel)}


# ---------------------------------------------------------------------- voi

def test_loss_is_asymmetric():
    assert loss(Category.YELLOW, Category.YELLOW) == 0.0
    under = loss(Category.YELLOW, Category.ORANGE)   # ranked one band too low
    over = loss(Category.ORANGE, Category.YELLOW)    # ranked one band too high
    assert under > over * 3

    # and the asymmetry widens with distance
    near = under / over
    far = loss(Category.GREEN, Category.RED) / loss(Category.RED, Category.GREEN)
    assert far > near * 10


def test_prior_case_mix_is_plausible():
    """Guards the calibration the selector depends on.

    If the prior thinks half the department is Very urgent, that becomes the
    best guess regardless of any answer, every question scores zero, and the
    selector silently stops ranking anything.
    """
    dist = acuity_distribution(BeliefState(PROTOCOL), frozenset(PROTOCOL.branches))
    assert dist[Category.RED] < 0.05
    assert dist[Category.ORANGE] < 0.25
    assert dist[Category.YELLOW] + dist[Category.GREEN] + dist[Category.BLUE] > 0.70


def test_bayes_risk_is_zero_when_certain():
    assert bayes_risk({Category.ORANGE: 1.0}) == 0.0


def test_predictable_answers_are_worthless():
    """A question we can already answer carries almost no information."""
    plausible = frozenset({"chest_pain", "abdominal_pain"})
    b = BeliefState(PROTOCOL)
    fresh, _ = value_of(b, plausible, "cardiac_pain")

    answered = b.record("cardiac_pain", Answer.TRUE)
    already, _ = value_of(answered, plausible, "cardiac_pain")
    assert fresh > 0.0
    assert already == pytest.approx(0.0, abs=1e-9)


def test_questions_that_cannot_move_the_band_are_marked():
    plausible = frozenset({"chest_pain", "abdominal_pain"})
    b = BeliefState(PROTOCOL).record("shock", Answer.TRUE)  # already Immediate
    for c in rank(b, plausible):
        assert not c.can_escalate


def test_danger_question_outranks_the_merely_diagnostic():
    """'Does it spread to your jaw' beats 'when did it start'.

    Onset is a fine diagnostic question and a worthless triage one: no answer
    to it can move the band. That distinction is the whole point of weighting
    the objective by the cost of ranking someone too low.
    """
    plausible = frozenset({"abdominal_pain", "chest_pain", "unwell_adult", "vomiting"})
    b = walk_in_baseline().record("moderate_pain", Answer.TRUE)
    order = [c.discriminator_id for c in rank(b, plausible)]
    assert order.index("cardiac_pain") < order.index("recent_onset")


def test_across_the_room_look_clears_those_checks_from_the_queue():
    """Why walk_in_baseline exists.

    Looking at the patient answers four checks that would otherwise sit in the
    question queue costing an observation each.
    """
    plausible = frozenset(PROTOCOL.branches)
    looked_at = {"airway_compromise", "inadequate_breathing", "unresponsive", "currently_fitting"}

    raw = {c.discriminator_id for c in rank(BeliefState(PROTOCOL), plausible)}
    assert looked_at <= raw

    seeded = {c.discriminator_id for c in rank(walk_in_baseline(), plausible)}
    assert not (looked_at & seeded)


def test_vitals_are_the_first_thing_asked_for():
    """Shock and saturation cannot be judged across a room, so they lead."""
    plausible = frozenset(PROTOCOL.branches)
    top = rank(walk_in_baseline(), plausible, limit=3)
    assert any(PROTOCOL.discriminators[c.discriminator_id].source is Source.MEASURE
               for c in top)


def test_free_lookups_outrank_questions_of_similar_value():
    plausible = frozenset(PROTOCOL.branches)
    ranked = rank(walk_in_baseline(), plausible)
    top_ten = [c.discriminator_id for c in ranked[:10]]
    assert any(PROTOCOL.discriminators[d].source is Source.RECORD for d in top_ten)


def test_stops_on_escalation():
    plausible = frozenset({"chest_pain"})
    b = BeliefState(PROTOCOL).record("cardiac_pain", Answer.TRUE)
    reason = should_stop(b, plausible, asked=1)
    assert reason is not None and "stop" in reason


def test_stops_when_budget_spent():
    plausible = frozenset({"unwell_adult"})
    b = BeliefState(PROTOCOL)
    assert should_stop(b, plausible, asked=8) == "question budget exhausted"
