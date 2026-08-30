"""Suggested first-line actions -- what a UK ED nurse might start, not why.

CTAS and this project's illustrative charts answer one question: how soon does
a clinician need to see this patient. They deliberately answer nothing else --
the manual prioritises time-to-assessment and stops there.

Real departments run a second table alongside that one: nurse-initiated
first-line actions, usually authorised by a Patient Group Direction or local
protocol, keyed on the presenting complaint and how urgent it is. "Chest pain
gets a 12-lead ECG within 10 minutes" is that kind of rule -- a standing order
tied to a category and a branch, not a diagnosis of what is wrong with the
patient.

That distinction is why this module exists as a second table rather than a
field bolted onto Decision. A root-cause or differential-diagnosis suggestion
would need a model to reason about the patient's condition, and that reasoning
would sit upstream of the category -- which is exactly the boundary this
project keeps closed. A first-line action is different in kind: it is a
lookup on (branch, category), the same shape as the CTAS table itself, and no
model touches it.

PROVENANCE, same caveat as charts.py: this table is authored from general,
publicly-documented UK ED practice (PGD-style standing orders that appear
across NHS trust protocols) for demonstration. It is NOT transcribed from a
licensed protocol and NO CLINICIAN HAS REVIEWED IT. Doses are illustrative
where given at all, and every action is gated behind local policy ("per local
PGD") because that gate is real -- a nurse cannot act on this table alone.
"""

from __future__ import annotations

from .core import Category

R, O, Y, G, B = Category.RED, Category.ORANGE, Category.YELLOW, Category.GREEN, Category.BLUE

# branch id -> {
#   "always": actions suggested at every category for this branch,
#   "thresholds": [(category, actions)] -- actions add on once the decision is
#      at least that urgent (category.value <= threshold.value), most urgent
#      first. A RED patient accumulates the ORANGE and YELLOW lines too.
# }
BRANCH_ACTIONS: dict[str, dict] = {
    "chest_pain": {
        "always": (
            "12-lead ECG within 10 minutes of arrival -- the chest-pain time "
            "target applies regardless of category",
        ),
        "thresholds": (
            (O, ("Continuous cardiac monitoring",
                 "IV access",
                 "Oxygen if saturation below 94%",
                 "Aspirin 300mg if not already taken and no contraindication, "
                 "per local PGD")),
            (Y, ("Repeat ECG if pain recurs or changes character",
                 "Troponin bloods per local ACS pathway")),
        ),
    },
    "abdominal_pain": {
        "always": ("Urinalysis (dipstick)",),
        "thresholds": (
            (O, ("IV access",
                 "Analgesia per pain score and local PGD",
                 "Lying and standing BP if occult bleeding is a concern")),
            (Y, ("Pregnancy test where applicable",
                 "Anti-emetic per local PGD if vomiting")),
        ),
    },
    "headache": {
        "always": ("Neurological observations (GCS, pupils) at triage",),
        "thresholds": (
            (O, ("IV access",
                 "Flag urgent CT pathway to the clinician if thunderclap onset "
                 "or meningism",
                 "Neuro observations continued at short, fixed intervals")),
            (Y, ("Analgesia per pain score and local PGD",
                 "Safety-net advice on red-flag symptoms to return for")),
        ),
    },
    "limb_problems": {
        "always": ("Immobilise or splint the affected limb",),
        "thresholds": (
            (O, ("Neurovascular observations (colour, warmth, pulse, "
                 "sensation) every 30 minutes",
                 "Elevate the limb if able",
                 "Flag urgent orthopaedic review to the clinician")),
            (Y, ("Analgesia per pain score and local PGD",
                 "X-ray request per local protocol")),
        ),
    },
}
# The real CTAS pack names the same branch "extremity_injury" rather than
# "limb_problems" -- same content, aliased rather than duplicated so the two
# can't quietly drift apart.
BRANCH_ACTIONS["extremity_injury"] = BRANCH_ACTIONS["limb_problems"]


def first_line_actions(branch_id: str, category: Category) -> list[str]:
    """Suggested actions for this branch at this category, most-urgent-first.

    Returns [] for a branch with no table entry -- silence rather than a
    guess, same discipline as the rest of the engine: unmapped is unmapped,
    never smoothed over.
    """
    entry = BRANCH_ACTIONS.get(branch_id)
    if not entry:
        return []
    out = list(entry.get("always", ()))
    for threshold, actions in entry.get("thresholds", ()):
        if category <= threshold:
            out.extend(actions)
    return out
