"""Paediatric normal ranges, transcribed from the CTAS manual Appendix G.

Extracted mechanically from the source PDF rather than retyped, so there is no
hand-transcription step to get wrong. Both tables parsed with zero malformed rows.

The manual sources these from:

    Fleming S, Thompson M, Stevens R, Heneghan C. "Normal ranges of heart rate
    and respiratory rate in children from birth to 18 years of age: a systematic
    review of observational studies." The Lancet 2011; 377(9770): 1011-1019.

Two independent checks that the extraction is right. At birth the heart-rate
normal band is 111-143, centre 127 -- exactly Fleming's reported median heart
rate at birth. The respiratory band is 35-53, centre 44 -- exactly Fleming's
median respiratory rate at birth.

STRUCTURE. Each row is six boundaries carving seven bands, per the manual's own
column headings:

    < b0            Level 1     3 or more SD outside normal
    b0 .. < b1      Level 2        2 SD outside normal
    b1 .. < b2      Level 3        1 SD outside normal
    b2 .. b3        Level 4/5   within normal
    > b3 .. b4      Level 3
    > b4 .. b5      Level 2
    > b5            Level 1

Boundaries are stored literally rather than as a mean and an SD. For infants the
table is symmetric to within rounding -- heart rate at birth steps by exactly 16
either side -- but respiratory rate in older children is not: the first step
above normal is consistently wider than the one below it, because the
distribution is right-skewed. Deriving mean +/- SD would quietly discard that.

And the warning this module exists to serve, from the same appendix:

    "Children triaged as CTAS 1 or CTAS 2 should never be delayed at triage to
     complete history or measurement of vital signs to confirm their triage
     level."

    "(When in doubt - triage up!)"
"""

from ..core import Category

# (age_months, (b0, b1, b2, b3, b4, b5))
RESPIRATORY_RATE: tuple[tuple[int, tuple[int, ...]], ...] = (
    (  0, ( 17,  26,  35,  53,  62,  71)),  # birth
    (  3, ( 16,  25,  33,  51,  60,  68)),  # 3 mon
    (  6, ( 15,  23,  32,  48,  57,  65)),  # 6 mon
    (  9, ( 14,  22,  30,  46,  54,  62)),  # 9 mon
    ( 12, ( 14,  22,  29,  44,  52,  59)),  # 12 mon
    ( 15, ( 14,  21,  28,  42,  49,  56)),  # 15 mon
    ( 18, ( 14,  20,  27,  39,  46,  52)),  # 18 mon
    ( 21, ( 14,  20,  26,  37,  43,  49)),  # 21 mon
    ( 24, ( 14,  19,  25,  35,  40,  45)),  # 2 yr
    ( 36, ( 14,  18,  22,  30,  34,  38)),  # 3 yr
    ( 48, ( 15,  18,  21,  24,  30,  33)),  # 4 yr
    ( 60, ( 15,  18,  20,  23,  28,  31)),  # 5 yr
    ( 72, ( 15,  17,  19,  22,  27,  29)),  # 6 yr
    ( 84, ( 14,  16,  19,  21,  26,  28)),  # 7 yr
    ( 96, ( 13,  16,  18,  20,  25,  27)),  # 8 yr
    (108, ( 13,  15,  17,  20,  24,  27)),  # 9 yr
    (120, ( 12,  15,  17,  19,  24,  26)),  # 10 yr
    (132, ( 12,  14,  16,  19,  24,  26)),  # 11 yr
    (144, ( 11,  14,  16,  18,  23,  26)),  # 12 yr
    (156, ( 11,  13,  16,  18,  23,  25)),  # 13 yr
    (168, ( 10,  13,  15,  17,  22,  25)),  # 14 yr
    (180, ( 10,  12,  15,  17,  22,  24)),  # 15 yr
    (192, (  9,  12,  14,  16,  21,  24)),  # 16 yr
    (204, (  9,  11,  13,  16,  21,  23)),  # 17 yr
    (216, (  9,  11,  13,  15,  20,  22)),  # 18 yr
)

HEART_RATE: tuple[tuple[int, tuple[int, ...]], ...] = (
    (  0, ( 79,  95, 111, 143, 159, 175)),  # birth
    (  3, ( 95, 111, 127, 158, 173, 189)),  # 3 mon
    (  6, ( 91, 106, 121, 152, 167, 183)),  # 6 mon
    (  9, ( 86, 101, 116, 145, 160, 175)),  # 9 mon
    ( 12, ( 83,  97, 111, 140, 155, 169)),  # 12 mon
    ( 15, ( 79,  94, 108, 137, 152, 166)),  # 15 mon
    ( 18, ( 76,  90, 105, 134, 148, 163)),  # 18 mon
    ( 21, ( 73,  87, 102, 131, 145, 159)),  # 21 mon
    ( 24, ( 71,  85,  99, 128, 142, 156)),  # 2 yr
    ( 36, ( 64,  78,  92, 120, 135, 149)),  # 3 yr
    ( 48, ( 59,  73,  88, 116, 130, 144)),  # 4 yr
    ( 60, ( 56,  70,  84, 112, 126, 140)),  # 5 yr
    ( 72, ( 53,  67,  81, 109, 123, 136)),  # 6 yr
    ( 84, ( 50,  64,  78, 105, 119, 133)),  # 7 yr
    ( 96, ( 47,  61,  75, 102, 116, 129)),  # 8 yr
    (108, ( 45,  59,  72,  99, 113, 126)),  # 9 yr
    (120, ( 43,  57,  70,  97, 110, 124)),  # 10 yr
    (132, ( 42,  55,  68,  95, 108, 122)),  # 11 yr
    (144, ( 40,  53,  67,  93, 106, 120)),  # 12 yr
    (156, ( 39,  52,  65,  92, 105, 118)),  # 13 yr
    (168, ( 37,  51,  64,  90, 103, 116)),  # 14 yr
    (180, ( 36,  49,  62,  89, 102, 115)),  # 15 yr
    (192, ( 35,  48,  61,  87, 100, 113)),  # 16 yr
    (204, ( 34,  47,  60,  86,  99, 112)),  # 17 yr
    (216, ( 33,  45,  58,  85,  97, 110)),  # 18 yr
)


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

_BANDS = (Category.RED, Category.ORANGE, Category.YELLOW)


def _category_for(value: float, bounds: tuple[int, ...]) -> Category | None:
    """Which band this reading falls in for one age row.

    None means within normal -- the vital signs do not constrain the category,
    and history decides between Level 4 and 5.
    """
    b0, b1, b2, b3, b4, b5 = bounds
    if value < b0 or value > b5:
        return Category.RED
    if value < b1 or value > b4:
        return Category.ORANGE
    if value < b2 or value > b3:
        return Category.YELLOW
    return None


def _bracketing(months: float, table) -> list[tuple[int, tuple[int, ...]]]:
    """The rows either side of this age.

    Ages fall between published rows -- a four-month-old sits between the 3 and
    6 month rows, whose normal bands differ by six beats. Rather than pick one,
    take both and let the caller resolve toward urgency, which is what the
    appendix says to do: "When in doubt - triage up!"
    """
    below = [r for r in table if r[0] <= months]
    above = [r for r in table if r[0] >= months]
    out = []
    if below:
        out.append(max(below, key=lambda r: r[0]))
    if above:
        out.append(min(above, key=lambda r: r[0]))
    return out or [table[0] if months < table[0][0] else table[-1]]


def category_for(value: float, months: float, table) -> Category | None:
    """Most urgent category this reading carries across the bracketing age rows.

    Raises ValueError above 18 years: the paediatric table stops there and an
    adult score applies instead. Silently reading off the last row would be the
    same silent-substitution error this module exists to prevent.
    """
    if months > 216:
        raise ValueError(
            f"age {months} months is beyond the paediatric table (18 years); "
            "use the adult vital-sign score"
        )
    results = [
        c for c in (_category_for(value, b) for _, b in _bracketing(months, table))
        if c is not None
    ]
    return min(results) if results else None      # lower value = more urgent


def respiratory_rate(value: float, months: float) -> Category | None:
    return category_for(value, months, RESPIRATORY_RATE)


def heart_rate(value: float, months: float) -> Category | None:
    return category_for(value, months, HEART_RATE)


def assess(months: float, *, resp_rate: float | None = None,
           pulse: float | None = None) -> tuple[Category | None, list[str]]:
    """Both vitals at once. Returns the most urgent category, and what drove it."""
    found: list[Category] = []
    why: list[str] = []

    if resp_rate is not None:
        c = respiratory_rate(resp_rate, months)
        if c is not None:
            found.append(c)
            why.append(f"respiratory rate {resp_rate:g} -> {c.label}")
    if pulse is not None:
        c = heart_rate(pulse, months)
        if c is not None:
            found.append(c)
            why.append(f"heart rate {pulse:g} -> {c.label}")

    if not found:
        why.append("both within normal for age")
        return None, why
    return min(found), why
