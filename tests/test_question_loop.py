"""The follow-up loop -- the part a nurse actually operates.

Both tests here are regressions for bugs that made the loop look broken on
screen while every unit underneath it passed.
"""

import serve
from triage import Answer


def encounter(text, age=54):
    serve.USE_MODEL = False          # keyword tier: no model, no network
    e = serve.Encounter("test", age=age)
    e.add_narrative(text)
    return e


def run(e, answer_with, turns=10):
    """Drive the loop, replying as a person would to whatever was asked."""
    asked = []
    for _ in range(turns):
        step = e.next_step()
        if step.get("stopped") or not step["question"]:
            return asked, step
        asked.append(step["question"])
        e.answer(step["check"], answer_with(step["question"]))
    return asked, None


def reply(q):
    if "nought to ten" in q:
        return "about a 5"
    if "start" in q:
        return "two hours ago"
    return "no"


def test_a_mild_presentation_actually_gets_asked_something():
    """next_step used to return the top-ranked candidate whatever its source.

    A saturation probe outranks most questions -- it is cheap and settles
    several checks -- so the panel showed a device reading with no question and
    the interview never started. Device work is now listed separately.
    """
    e = encounter("I've had a headache since this morning and I feel a bit off.")
    asked, _ = run(e, reply)
    assert len(asked) >= 3, f"loop asked almost nothing: {asked}"

    step = e.next_step()
    for item in step.get("pending", []):
        assert item["actor"] != "patient"


def test_the_pain_ladder_is_one_question_not_three():
    """severe/moderate/mild share one question, and were ranked separately.

    The patient was asked "how bad is the pain" up to three times in a row. One
    number now settles the whole ladder, in both directions.
    """
    e = encounter("I've had a headache since this morning and I feel a bit off.")
    asked, _ = run(e, reply)
    assert len(asked) == len(set(asked)), f"asked twice: {asked}"

    # CTAS bands central pain 8-10/4-7/1-3 (see triage/pain.py) -- a 5 is
    # moderate, and headache is a central-locality branch (see PAIN_LOCALITY
    # in protocols/ctas.py), so only the central checks should be touched;
    # peripheral is a different branch's business entirely and stays unknown.
    answers = e.belief.answers
    assert answers["moderate_pain_central"] is Answer.TRUE
    assert answers["severe_pain_central"] is Answer.FALSE
    assert answers["mild_pain_central"] is Answer.FALSE
    assert "severe_pain_peripheral" not in answers


def test_a_stop_says_why_rather_than_going_blank():
    """An Orange presentation must not be interviewed further -- but the screen
    used to hide the panel, which is indistinguishable from a hang."""
    e = encounter("Worst headache of my life, came on suddenly an hour ago.")
    step = e.next_step()
    assert step is not None
    assert step["stopped"], "a stop must carry its reason to the page"
    assert "alert and stop" in step["stopped"]


def test_keyword_parse_keeps_the_guards_the_model_tier_has():
    """The bottom rung has to be conservative in the same direction.

    A weaker tier is allowed to know less. It is not allowed to invent a denial
    -- an unanswered check must stay unknown, because unknown widens the
    plausible set and lowers confidence, while a false "no" quietly closes a
    branch nobody ruled out.
    """
    from triage import keyword_parse

    assert keyword_parse("yes it does")[0] is Answer.TRUE
    assert keyword_parse("no, nothing like that")[0] is Answer.FALSE

    for hedge in ["maybe", "I'm not sure", "hard to say", "hmm", ""]:
        assert keyword_parse(hedge)[0] is Answer.UNKNOWN, hedge

    # Both markers at once is ambiguous, not a no.
    assert keyword_parse("no, well, yes actually")[0] is Answer.UNKNOWN
