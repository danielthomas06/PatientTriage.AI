"""Deterministic red-flag detection.

Runs in parallel with the model, never downstream of it. Pure regex over the
transcript: no network, no inference, microseconds.

This exists because the model is allowed to be wrong. If either the tripwire
or the model fires, the patient escalates -- union, not intersection. The
tripwire is the half that still works when everything else is down.

It only ever escalates. There is no pattern here that lowers a priority.
"""

import re
from dataclasses import dataclass

from .core import Category


@dataclass(frozen=True, slots=True)
class Flag:
    pattern_id: str
    category: Category
    why: str
    matched: str
    implies: tuple[str, ...] = ()  # discriminators this supports, if any


# Deliberately broad. A false positive costs someone a few minutes of
# attention; a false negative is the thing this file exists to prevent.
_PATTERNS: tuple[tuple[str, str, Category, str, tuple[str, ...]], ...] = (
    (
        "cardiac_radiation",
        # Two ways in. The first is chest pain sitting near jaw/arm/neck.
        #
        # The second catches the presentation this whole system exists for: an
        # inferior MI felt in the stomach, not the chest. "Pain in my upper
        # stomach that goes into my jaw" contains no cardiac word at all, so a
        # chest-anchored pattern misses it -- and so does the nurse who picked
        # the Abdominal pain branch. Anchor on the *radiation verb* instead of
        # the body part, and the site of the pain stops mattering.
        r"\b(chest|heart)\b.{0,60}\b(jaw|arm|neck|shoulder)\b"
        r"|\b(jaw|arm|neck|shoulder)\b.{0,60}\b(chest|heart)\b"
        r"|\b(goes?|going|spread(s|ing)?|radiat\w+|shoot(s|ing)?|travel(s|ling)?|mov(es|ing)|run(s|ning)?)\b"
        r"\s+(in)?to\b.{0,25}\b(jaw|neck|shoulder|arm)\b"
        r"|\bdown\s+(my\s+)?(left|right)\s+arm\b",
        Category.ORANGE,
        "pain radiating to jaw, neck, shoulder or arm",
        ("cardiac_pain",),
    ),
    (
        "cardiac_sweating",
        r"\b(chest|heart)\b.{0,80}\b(sweat|sweaty|clammy|cold sweat)\b",
        Category.ORANGE,
        "chest pain with sweating",
        (),
    ),
    (
        "crushing_pain",
        r"\b(crushing|vice|vise|elephant|band|tight(ness)?|pressure)\b.{0,40}\bchest\b"
        r"|\bchest\b.{0,40}\b(crushing|vice|vise|elephant|tight(ness)?|pressure)\b",
        Category.ORANGE,
        "crushing or pressure-type chest pain",
        (),
    ),
    (
        "thunderclap",
        r"\bworst\b.{0,30}\bheadache\b"
        r"|\bheadache\b.{0,40}\b(sudden(ly)?|thunderclap|like a bang|hit me)\b"
        r"|\bsudden(ly)?\b.{0,30}\bworst\b.{0,30}\bhead",
        Category.ORANGE,
        "sudden severe headache",
        ("thunderclap_headache",),
    ),
    (
        "stroke_signs",
        r"\b(face|mouth)\b.{0,30}\b(droop|drooping|fell|sagging|numb)\b"
        r"|\b(slurr?(ed|ing)|can'?t get my words|words (won'?t|not) com)"
        r"|\b(one side|left side|right side)\b.{0,40}\b(weak|numb|dead|won'?t move)\b",
        Category.ORANGE,
        "possible stroke signs",
        ("new_neuro_deficit",),
    ),
    (
        "airway_breathing",
        r"\bcan'?t (breathe|catch my breath|get (my )?air)\b"
        r"|\b(struggling|fighting) (to|for) breath"
        r"|\bcan'?t (talk|speak|finish a sentence)\b",
        Category.RED,
        "possible airway or breathing compromise",
        ("cannot_complete_sentence",),
    ),
    (
        "major_bleeding",
        r"\b(bleeding|blood)\b.{0,40}\b(won'?t stop|everywhere|pouring|gushing|soaked)\b"
        r"|\b(pouring|gushing|spurting)\b.{0,20}\bblood\b",
        Category.ORANGE,
        "uncontrolled bleeding",
        ("major_haemorrhage",),
    ),
    (
        "self_harm",
        r"\b(kill myself|end (it|my life)|suicidal|don'?t want to (be here|live|go on))\b"
        r"|\btake(n)? .{0,20}\b(overdose|too many (pills|tablets))\b",
        Category.ORANGE,
        "self-harm risk disclosed",
        (),
    ),
    (
        "pregnancy_bleeding",
        r"\b(pregnan|expecting|weeks gone)\b.{0,60}\b(bleed|bleeding|blood|cramp)\b"
        r"|\b(bleed|bleeding)\b.{0,60}\b(pregnan|expecting)\b",
        Category.ORANGE,
        "bleeding in pregnancy",
        ("pv_bleeding",),
    ),
    (
        "collapse",
        r"\b(passed out|blacked out|collapsed|fainted|went unconscious)\b",
        Category.YELLOW,
        "loss of consciousness reported",
        ("history_of_unconsciousness",),
    ),
    (
        "sepsis_language",
        r"\b(shivering|shaking|rigors?)\b.{0,50}\b(fever|hot|temperature|burning up)\b"
        r"|\bfeel like (i'?m )?(going to |gonna )?di(e|ing)\b",
        Category.ORANGE,
        "possible sepsis / impending doom",
        (),
    ),
    (
        "anticoagulant_head",
        r"\b(warfarin|apixaban|rivaroxaban|blood thinner|anticoagulant)\b.{0,80}"
        r"\b(head|fell|fall|hit)\b"
        r"|\b(hit|banged|knocked)\b.{0,40}\bhead\b.{0,80}\b(warfarin|blood thinner)\b",
        Category.ORANGE,
        "head injury on anticoagulants",
        ("anticoagulated",),
    ),
)

_COMPILED = tuple(
    (pid, re.compile(rx, re.IGNORECASE | re.DOTALL), cat, why, implies)
    for pid, rx, cat, why, implies in _PATTERNS
)


def scan(transcript: str) -> list[Flag]:
    """Every red flag present in the text, most urgent first."""
    found: list[Flag] = []
    for pattern_id, rx, category, why, implies in _COMPILED:
        m = rx.search(transcript)
        if m:
            found.append(
                Flag(
                    pattern_id=pattern_id,
                    category=category,
                    why=why,
                    matched=m.group(0).strip(),
                    implies=implies,
                )
            )
    found.sort(key=lambda f: f.category)
    return found


def ceiling(transcript: str) -> Category | None:
    """The most urgent category the tripwire alone would justify.

    Used as a floor on the engine's answer: the final priority is the more
    urgent of the two. The tripwire can raise a category and can never lower
    one.
    """
    flags = scan(transcript)
    return flags[0].category if flags else None
