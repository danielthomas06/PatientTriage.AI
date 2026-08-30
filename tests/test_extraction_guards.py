"""Two guard gaps found from live use, both in the model tiers.

Both were found the same way: a real narrative produced a wrong or misleading
belief state even though every existing guard passed it. `_accept()` is the
one place every model-tier finding goes through, so both fixes live there.
"""

from triage.core import Protocol
from triage.charts import PROTOCOL
from triage.extract import _accept


def test_presence_of_pain_is_not_its_severity():
    """A live run had the model quote "I've been having stomach pain" -- true,
    verbatim, and silent on how bad it is -- as evidence for mild_pain=true.
    _verify() passed it because the quote is real; nothing checked that it
    actually rates severity. That falsely closed the pain-severity ladder,
    which then changed which question got asked next.
    """
    findings = [{"check_id": "mild_pain", "present": True,
                 "quote": "I've been having stomach pain."}]
    result = _accept(findings, [], "I've been having stomach pain.", PROTOCOL, tier="local")
    assert "mild_pain" not in result.belief.answers
    assert any("rates nothing" in r for r in result.rejected)


def test_pain_severity_is_accepted_when_the_quote_actually_rates_it():
    """The guard must not overcorrect into rejecting every positive."""
    findings = [{"check_id": "severe_pain", "present": True,
                 "quote": "it's absolutely agonising, an 9 out of 10"}]
    result = _accept(findings, [], "it's absolutely agonising, an 9 out of 10",
                      PROTOCOL, tier="local")
    assert result.belief.answers.get("severe_pain") is not None


def test_a_paraphrase_the_model_misreads_still_surfaces_its_real_branch():
    """Live run: a local model read "pain at the back of my head" as Unwell
    adult (weight 1.0) and never named Headache at all -- a genuine branch-
    classification miss on non-canonical phrasing, not a fabricated example.
    The keyword hints already existed for the no-model tier; this is the same
    table used as a backstop so a model miss doesn't silently drop the branch
    that actually matters.
    """
    branch_guesses = [{"branch_id": "unwell_adult", "weight": 1.0}]
    result = _accept([], branch_guesses, "I have pain at the back of my head.",
                      PROTOCOL, tier="local")
    assert result.branch_weights["headache"] >= 0.5
    assert result.branch_weights["unwell_adult"] == 1.0   # never lowered


def test_the_backstop_never_lowers_what_the_model_named():
    """Additive only. A branch the model raised keeps its weight even where
    no keyword hint applies to it."""
    branch_guesses = [{"branch_id": "collapse", "weight": 0.9}]
    result = _accept([], branch_guesses, "she just went down suddenly",
                      PROTOCOL, tier="local")
    assert result.branch_weights["collapse"] == 0.9


def test_generic_falling_language_reaches_collapse_and_vertigo_on_keyword_tier():
    """Real user report: 'Not able to stand or sit. Feel like falling down.'
    left every branch at the floor on the keyword tier -- no hint pattern
    covered generic falling/unsteady phrasing, only explicit words like
    'fainted' or 'dizzy'. The model tier already generalises this correctly
    (verified separately against a live local model: weight 1.0 on vertigo)
    -- this tier is the crude, no-model fallback, and 'wider is safer' means
    raising both plausible branches rather than picking one.
    """
    from triage.extract import keyword_seed
    from triage.protocols.ctas import PROTOCOL

    result = keyword_seed("Not able to stand or sit. Feel like falling down.", PROTOCOL)
    assert result.branch_weights["collapse"] > 0.06
    assert result.branch_weights["vertigo"] > 0.06
