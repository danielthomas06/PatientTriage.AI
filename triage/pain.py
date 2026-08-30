"""A 0-10 pain score, resolved to whichever discriminators the active pack uses.

Two packs, two shapes for the same idea.

`charts.py` (illustrative) scores pain on severity alone: severe_pain (7-10),
moderate_pain (4-6), mild_pain (1-3).

The real CTAS pack scores pain on severity AND location, and the manual is
explicit about why:

    "Central pain originates within a body cavity (head, chest, abdomen) or
     organ (eye, testicle, deep soft tissue compartment) and may be associated
     with life- or limb-threatening conditions. Peripheral pain originates in
     the skin, soft tissues, axial skeleton or superficial organs where
     dangerous diagnoses are less likely to be missed."
     (Sec 2.4.2, Pain Severity)

That is not a cosmetic split. In `protocols/ctas.py`, severe peripheral pain is
Yellow where severe central pain is Orange -- the SAME 8-10 score lands a full
category lower if it is scored under the wrong location. Getting the location
right is a safety property, not a display choice, which is why this lives in
its own module rather than as a side-detail of the vitals form or the
follow-up loop -- both call the same function so neither can drift from it.

The manual's own caveat is worth keeping in mind even though nothing here can
enforce it: "If a patient presents with pain in a site traditionally
considered peripheral pain but the nurse suspects a life or limb threatening
condition (e.g. necrotizing fasciitis) then the pain should be scored as
central pain." That is a clinical judgement call, not a rule a locality table
can make -- a nurse can always record the check directly and override the
inferred locality.
"""

from __future__ import annotations

from .core import Protocol

FLAT_LADDER = ("severe_pain", "moderate_pain", "mild_pain")
_FLAT_BANDS = {"severe_pain": (7, 10), "moderate_pain": (4, 6), "mild_pain": (1, 3)}

_LOCALISED_BANDS = {"severe": (8, 10), "moderate": (4, 7), "mild": (1, 3)}


def pain_ladder(protocol: Protocol) -> str:
    """Which shape this pack uses: 'flat', 'localised', or 'none'."""
    if all(c in protocol.discriminators for c in FLAT_LADDER):
        return "flat"
    if all(f"{band}_pain_central" in protocol.discriminators for band in _LOCALISED_BANDS):
        return "localised"
    return "none"


def resolve_pain(
    protocol: Protocol, score: int, locality: str = "central"
) -> list[tuple[str, bool]]:
    """(check_id, is_true) for every pain-severity check this pack defines at
    this score. Only the checks that actually apply are returned -- for a
    localised pack, the OTHER locality's checks are left out entirely rather
    than written as False, because they were never asked about and "false"
    would claim more than was established.
    """
    shape = pain_ladder(protocol)
    if shape == "flat":
        return [(cid, lo <= score <= hi) for cid, (lo, hi) in _FLAT_BANDS.items()]
    if shape == "localised":
        return [
            (f"{band}_pain_{locality}", lo <= score <= hi)
            for band, (lo, hi) in _LOCALISED_BANDS.items()
        ]
    return []
