"""choose_next() -- the model picks WHICH shortlisted question to ask and
phrases it, with the conversation in view. It never gets a vote on urgency,
and it can never choose anything outside the shortlist it was handed.

These are guard tests, not model-quality tests: they exercise _accept_choice
directly with fabricated model output, the same way test_extraction_guards.py
tests _accept() without a live model. Whether the model USUALLY makes a good
choice is a separate, ongoing question -- these tests are about what happens
when it doesn't.
"""

from triage.extract import _accept_choice
from triage.protocols.ctas import PROTOCOL


SHORTLIST = ["severe_pain_central", "new_neuro_deficit", "persistent_vomiting"]


def test_a_choice_outside_the_shortlist_is_discarded():
    """The one property that matters most: an id the model invents, or one
    that's real but wasn't offered, must not be trusted -- same discipline
    as a narrative extraction finding that names an unknown check."""
    result = _accept_choice(
        "bleeding_third_trimester", "How many weeks pregnant are you?",
        "just checking", SHORTLIST, PROTOCOL,
    )
    assert result is None

    result = _accept_choice(
        "not_a_real_check_id", "made up", "made up", SHORTLIST, PROTOCOL,
    )
    assert result is None


def test_a_valid_shortlisted_choice_is_accepted():
    result = _accept_choice(
        "new_neuro_deficit", "Have you noticed any weakness or numbness?",
        "worth ruling out given the dizziness", SHORTLIST, PROTOCOL,
    )
    assert result == (
        "new_neuro_deficit", "Have you noticed any weakness or numbness?",
        "worth ruling out given the dizziness",
    )


def test_a_leading_question_still_falls_back_to_reviewed_wording():
    """The anti-leading guard applies here exactly as it does to render() --
    a generated question that names the answer it's testing for is replaced
    with the reviewed canned wording, not trusted as written."""
    d = PROTOCOL.discriminators["pain_radiating"]
    assert d.leading, "test fixture assumption: pain_radiating has leading terms"
    result = _accept_choice(
        "pain_radiating", "Does the pain spread to your jaw?", "why",
        ["pain_radiating"], PROTOCOL,
    )
    assert result[1] == d.question  # reviewed wording, not the leading one


def test_choose_next_returns_none_for_an_empty_shortlist():
    from triage.extract import choose_next
    assert choose_next([], [], {}, PROTOCOL) is None


def test_serve_falls_back_cleanly_when_the_model_is_off():
    """USE_MODEL=False must behave exactly as it did before this feature --
    top VOI pick, canned wording, no round trip attempted at all."""
    import serve
    serve.USE_MODEL = False
    serve.ENCOUNTERS.clear()
    e = serve.new_patient()
    e.set_identity({"name": "Test", "age": "40", "sex": "F", "patient_id": ""})
    e.add_narrative("I've had a headache since this morning and I feel a bit off.")
    s = e.next_step()
    assert s["question"]
    assert s["why"] == ""
