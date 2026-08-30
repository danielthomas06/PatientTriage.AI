"""Tests for the model layer.

None of these call the API. The interesting behaviour is in the guards --
what happens when the model returns something it should not have — and those
are pure functions.
"""

import pytest

from triage import Answer, Category, keyword_seed, scan
from triage.extract import _verify
from triage.tripwire import ceiling
from triage.charts import PROTOCOL  # these tests assert on the illustrative pack


# ------------------------------------------------------------ evidence guard

def test_quote_must_appear_in_transcript():
    t = "I've had this pain in my upper stomach since about six this morning"
    assert _verify("pain in my upper stomach", t)
    assert not _verify("crushing central chest pain", t)


def test_quote_check_tolerates_punctuation_and_case():
    t = "It's a bit sweaty, and — honestly — I feel sick."
    assert _verify("Its a bit sweaty", t)
    assert _verify("I FEEL SICK", t)


def test_empty_quote_is_never_valid():
    assert not _verify("", "anything at all")
    assert not _verify("   ", "anything at all")


def test_paraphrase_is_rejected():
    """The guard exists to catch exactly this: plausible, unsaid, dangerous."""
    t = "my chest feels tight"
    assert not _verify("patient reports crushing chest pain radiating to the jaw", t)


# ---------------------------------------------------------------- tripwire

def test_tripwire_catches_cardiac_radiation():
    flags = scan("the pain in my chest goes down into my left arm")
    assert any(f.pattern_id == "cardiac_radiation" for f in flags)
    assert ceiling("the pain in my chest goes down into my left arm") is Category.ORANGE


def test_tripwire_catches_radiation_without_the_word_chest():
    """The flagship case: an inferior MI described as stomach pain.

    There is no cardiac word anywhere in this sentence. A chest-anchored
    pattern misses it -- and so does the nurse who picked Abdominal pain.
    This is the exact presentation the whole system exists to catch, so it
    gets its own regression test.
    """
    t = "I have a pain in my upper stomach and it goes into my jaw sometimes"
    flags = scan(t)
    assert any(f.pattern_id == "cardiac_radiation" for f in flags), flags
    assert ceiling(t) is Category.ORANGE


@pytest.mark.parametrize(
    "phrase",
    [
        "the ache spreads to my neck",
        "it radiates into my shoulder",
        "the pain shoots into my jaw",
        "it goes down my left arm",
    ],
)
def test_tripwire_catches_radiation_phrasings(phrase):
    assert ceiling(phrase) is Category.ORANGE, phrase


def test_tripwire_does_not_fire_on_an_ordinary_limb_injury():
    """Radiation language is the signal, not the body part."""
    assert ceiling("I fell and now my arm hurts") is None
    assert ceiling("I've got a sore shoulder from lifting") is None


def test_tripwire_catches_worst_headache():
    assert ceiling("it's the worst headache I've ever had") is Category.ORANGE


def test_tripwire_catches_breathing_as_red():
    assert ceiling("I can't breathe properly") is Category.RED


def test_tripwire_catches_stroke_language():
    for phrase in (
        "my face is drooping on one side",
        "my words won't come out right",
        "my left side has gone weak",
    ):
        assert ceiling(phrase) is Category.ORANGE, phrase


def test_tripwire_catches_self_harm():
    assert ceiling("I don't want to be here any more") is Category.ORANGE


def test_tripwire_is_quiet_on_benign_text():
    assert ceiling("I twisted my ankle playing football yesterday") is None
    assert scan("I need a repeat prescription") == []


def test_tripwire_reports_what_it_matched():
    flags = scan("crushing pain in my chest")
    assert flags
    assert flags[0].matched
    assert flags[0].why


def test_tripwire_orders_most_urgent_first():
    flags = scan("I can't breathe and the pain goes into my jaw from my chest")
    assert len(flags) >= 2
    assert flags[0].category <= flags[1].category
    assert flags[0].category is Category.RED


# --------------------------------------------------------- offline fallback

def test_keyword_fallback_marks_itself_degraded():
    """A dropped rung belongs in `fallbacks`, not `rejected`.

    They used to share one list, so a missing API key rendered under "Rejected
    by the guard" and made routine operation look like the model had done
    something untrustworthy. `rejected` now means only one thing: the guard
    refused to believe a finding.
    """
    out = keyword_seed("chest pain going into my arm", PROTOCOL)
    assert out.degraded
    assert out.tier == "keyword"
    assert out.fallbacks and "keyword" in out.fallbacks[0]
    assert not out.rejected


def test_keyword_fallback_extracts_something_useful():
    out = keyword_seed("the pain goes into my jaw and I'm sweaty", PROTOCOL)
    assert out.belief.answers.get("cardiac_pain") is Answer.TRUE


def test_keyword_fallback_reads_a_pain_score():
    out = keyword_seed("it's about 8 out of 10", PROTOCOL)
    assert out.belief.answers.get("severe_pain") is Answer.TRUE

    out = keyword_seed("maybe 5/10", PROTOCOL)
    assert out.belief.answers.get("moderate_pain") is Answer.TRUE


def test_keyword_fallback_never_asserts_a_negative():
    """Degraded mode may know less. It must not claim to know more."""
    out = keyword_seed("my ankle hurts", PROTOCOL)
    assert Answer.FALSE not in out.belief.answers.values()


def test_keyword_fallback_keeps_every_branch_open():
    out = keyword_seed("I feel unwell", PROTOCOL)
    assert set(out.branch_weights) == set(PROTOCOL.branches)


def test_keyword_fallback_evidence_is_traceable():
    out = keyword_seed("the pain goes into my jaw", PROTOCOL)
    assert "jaw" in out.belief.evidence["cardiac_pain"].quote
    assert out.belief.evidence["cardiac_pain"].origin == "keyword fallback"
