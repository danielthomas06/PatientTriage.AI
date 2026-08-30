"""Pain in a patient who cannot answer "how bad, nought to ten".

Recorded case that prompted this: a 6-month-old with a fever was asked
"how bad is the pain, nought to ten" and "does the pain spread to your jaw,
neck or arm" -- a pre-verbal infant cannot self-report a number, and cannot
describe where pain radiates. The age/sex gate on obstetric checks fixed one
specific wrong question; this is the same class of problem for pain itself,
and it turns out CTAS has a real answer for it.

CTAS Appendix F names four paediatric pain tools, each for a different
developmental stage, and the population each is validated for is a direct
quote:

    Numerical Rating Scale (NRS) -- "School aged children and adolescents.
    Children must be capable of counting up to 10 and understand the
    concepts of classification and seriation."

    FLACC -- "Infants, toddlers, preschool children; and may also be useful
    for cognitively impaired children & adolescents." A BEHAVIOURAL
    OBSERVATIONAL tool, not self-report: a nurse scores what they see, not
    what the patient says.

This project's existing pain model (triage/pain.py, protocols/ctas.py) is
built entirely on the NRS -- a 0-10 self-report number. That's the right
tool for "school aged children and adolescents", and it's the wrong one for
anyone younger. FLACC is the manual's own answer for that gap, not
something invented here: five behaviours, each scored 0/1/2, summed to a
0-10 total that plugs into the SAME severity bands the NRS already uses
(pain.py's resolve_pain) -- because it's the same 0-10 scale, just filled
in by observation instead of a spoken number.

    Merkel, S.L., Voepel-Lewis, T., Shayeviz, J.R., & Malviya, S. "The
    FLACC: A behavioral scale for scoring postoperative pain in young
    children." Paediatric Nursing 1997; 23: 293-297. (Cited by the CTAS
    manual as FLACC's own source.)

WHAT'S DELIBERATELY SIMPLIFIED. The manual names two more tools for the
band in between -- a 4-point word scale and the Faces Pain Scale-revised,
both for "preschool children" who can self-report but not count to 10. This
module doesn't build a third tier for them; a preschooler currently gets
FLACC too, which is more conservative than asking a self-report question
they may not reliably understand, not less -- it costs a staff observation
instead of a patient answer, never a wrong inference. The manual gives no
exact numeric age boundary for any of these tools either (they're defined
developmentally: "must be capable of counting to 10", not "must be 7"), so
MINIMUM_SELF_REPORT_PAIN_AGE below is this project's own conservative
reading of that guidance, not a quoted number -- same honesty standard as
every other age threshold in this codebase.

ALSO FROM THE MANUAL, and implemented alongside this: "the Paediatrics
guidelines do not distinguish between central and peripheral pain" (Sec
2.4.2) -- so a child's pain, self-reported or FLACC-scored, is never split
by body region the way an adult's is. See PAIN_LOCALITY's paediatric
override in protocols/ctas.py.
"""

from __future__ import annotations

MINIMUM_SELF_REPORT_PAIN_AGE = 6
"""Years. Below this, pain is scored by staff observation (FLACC), not asked
of the patient. Conservative reading of the manual's developmental
criteria for the NRS ("school aged... capable of counting to 10"), not a
quoted number. Deliberately errs toward FLACC rather than self-report --
the failure mode of using it on a child who could have self-reported is a
staff observation instead of a patient answer, never a wrong inference."""


# Every ASK-source pain check a patient answers directly. Below the age
# threshold, none of these are ever put to the patient -- see next_step()'s
# use of this set in serve.py.
SELF_REPORT_PAIN_CHECKS = frozenset({
    "severe_pain_central", "moderate_pain_central", "mild_pain_central",
    "severe_pain_peripheral", "moderate_pain_peripheral", "mild_pain_peripheral",
    "pain_radiating",   # describing WHERE pain spreads is self-report too
})

FLACC_CATEGORIES = ("face", "legs", "activity", "cry", "consolability")

FLACC_DESCRIPTIONS = {
    "face":          ("No expression / smile", "Occasional grimace or frown, withdrawn",
                       "Frequent to constant quivering chin, clenched jaw"),
    "legs":          ("Normal position, relaxed", "Uneasy, restless, tense",
                       "Kicking, or legs drawn up"),
    "activity":      ("Lying quietly, moves easily", "Squirming, shifting back and forth, tense",
                       "Arched, rigid, or jerking"),
    "cry":           ("No cry (awake or asleep)", "Moans or whimpers, occasional complaint",
                       "Crying steadily, screams or sobs, frequent complaints"),
    "consolability": ("Content, relaxed", "Reassured by touch, hugging or talk; distractible",
                       "Difficult to console or comfort"),
}
"""Each category's three levels (0, 1, 2), the manual's own wording verbatim."""


def flacc_score(face: int, legs: int, activity: int, cry: int, consolability: int) -> int:
    """Sum five 0-2 behavioural observations to the same 0-10 scale the NRS
    uses. Each argument must be 0, 1, or 2 -- the manual's own bands."""
    scores = {"face": face, "legs": legs, "activity": activity,
              "cry": cry, "consolability": consolability}
    for name, value in scores.items():
        if value not in (0, 1, 2):
            raise ValueError(f"{name}={value!r} is not a valid FLACC score (0, 1, or 2)")
    return sum(scores.values())
