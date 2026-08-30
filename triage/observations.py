"""Measured numbers into discriminator answers.

A nurse types 91 into a saturation field. They do not tick a box that says "is
the oxygen saturation low". Everything below exists so the interface can take
what is actually recorded at triage -- numbers off a monitor -- and turn it into
the yes/no checks the engine reads.

Two things this buys beyond convenience.

NEGATIVES BECOME REAL. A measured SpO2 of 97% is genuine evidence that the
low-saturation checks are FALSE, not merely unasked. That is the one place in
this system where a negative can be asserted without a patient denying anything,
because a measurement is not a memory. Everywhere else, absence stays unknown.

AND THE THRESHOLDS LIVE IN ONE PLACE. Written into the UI they would drift from
the engine the first time anyone edited a form; here they sit next to the
discriminators they answer, and a protocol that names its checks differently
simply gets fewer of them rather than silently wrong ones.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import Answer, Protocol
from .pain import resolve_pain


@dataclass(frozen=True, slots=True)
class Measured:
    """What a triage nurse records. Every field optional -- a real encounter is
    partial, and a missing measurement must stay unknown rather than default."""

    respiratory_rate: float | None = None
    spo2: float | None = None
    on_oxygen: bool = False
    systolic_bp: float | None = None
    pulse: float | None = None
    temperature: float | None = None
    alert: bool | None = None          # AVPU: A -> True, anything else -> False
    pain_score: int | None = None      # 0-10
    age_years: float | None = None

    def recorded(self) -> dict[str, float | bool]:
        return {
            k: v for k, v in {
                "respiratory_rate": self.respiratory_rate, "spo2": self.spo2,
                "systolic_bp": self.systolic_bp, "pulse": self.pulse,
                "temperature": self.temperature, "alert": self.alert,
                "pain_score": self.pain_score,
            }.items() if v is not None
        }


def _fmt(value: float) -> str:
    return f"{value:g}"


def derive(
    m: Measured, protocol: Protocol, *,
    pain_locality: str = "central", adult_thresholds: bool = True,
) -> list[tuple[str, Answer, str]]:
    """(check_id, answer, evidence) for everything these numbers settle.

    Only checks the protocol actually defines are returned, so the same function
    serves a pack whose vocabulary differs. `pain_locality` matters only for a
    pack whose pain checks are split by body region (see `pain.py`); a pack
    with a flat severity scale ignores it.

    `adult_thresholds=False` withholds saturation and the pulse/BP-derived
    shock heuristic -- the two places this module turns a raw number directly
    into a threshold judgement. A resting pulse of 130 is shock in an adult
    and normal in a toddler; there is no single cutoff that is right for both,
    so under this flag the numbers are recorded but nothing is inferred from
    them. Alertness and pain are NOT gated -- AVPU and pain severity are
    read the same way across ages, this is specifically about age-VARYING
    numeric thresholds. Call this with `adult_thresholds=False` for any
    cohort that is not a scoreable adult (see `cohort.resolve`); the
    paediatric-appropriate signal comes from `cohort.paediatric_vital_category`
    instead, which is age-banded from birth.
    """
    out: list[tuple[str, Answer, str]] = []

    def add(check_id: str, positive: bool, why: str) -> None:
        if check_id in protocol.discriminators:
            out.append((check_id, Answer.TRUE if positive else Answer.FALSE, why))

    # ---- oxygen saturation ------------------------------------------------
    if m.spo2 is not None and adult_thresholds:
        air = "on oxygen" if m.on_oxygen else "on air"
        why = f"SpO2 {_fmt(m.spo2)}% {air}"
        add("very_low_spo2", m.spo2 < 92, why)
        add("low_spo2", 92 <= m.spo2 <= 94, why)
        add("spo2_under_90", m.spo2 < 90, why)
        add("spo2_under_92", m.spo2 < 92, why)
        add("spo2_92_to_94", 92 <= m.spo2 <= 94, why)

    # ---- respiratory rate -------------------------------------------------
    if m.respiratory_rate is not None and adult_thresholds:
        why = f"respiratory rate {_fmt(m.respiratory_rate)}"
        # Both extremes matter: a rate of 6 is a peri-arrest sign, not a calm one.
        add("inadequate_breathing", m.respiratory_rate < 8 or m.respiratory_rate > 30, why)

    # ---- circulation ------------------------------------------------------
    if m.pulse is not None and adult_thresholds:
        why = f"pulse {_fmt(m.pulse)}"
        add("abnormal_pulse", m.pulse < 50 or m.pulse > 120, why)

    if m.systolic_bp is not None and adult_thresholds:
        why = f"systolic {_fmt(m.systolic_bp)}"
        shocked = m.systolic_bp < 90
        if m.pulse is not None and m.pulse > 120 and m.systolic_bp < 100:
            # Tachycardia with a borderline pressure is compensated shock, which
            # a pressure threshold alone misses until it is late.
            shocked = True
            why = f"systolic {_fmt(m.systolic_bp)} with pulse {_fmt(m.pulse)}"
        add("shock", shocked, why)

    # ---- temperature ------------------------------------------------------
    if m.temperature is not None and adult_thresholds:
        why = f"temperature {_fmt(m.temperature)}C"
        add("very_hot", m.temperature >= 39.1, why)
        add("hot", 38.1 <= m.temperature <= 39.0, why)

    # ---- consciousness ----------------------------------------------------
    if m.alert is not None:
        why = "alert on AVPU" if m.alert else "not alert on AVPU"
        add("altered_conscious_level", not m.alert, why)
        add("altered_loc", not m.alert, why)

    # ---- pain -------------------------------------------------------------
    if m.pain_score is not None:
        why = f"pain {m.pain_score}/10"
        for check_id, positive in resolve_pain(protocol, m.pain_score, pain_locality):
            add(check_id, positive, why)

    return out


def news2_from(m: Measured):
    """The adult vital-sign score, or None when it does not apply.

    Returns None rather than a number whenever the score is not validated for
    this patient -- under 16, in pregnancy, or with too few observations. A score
    nobody should trust is worse than no score, because it looks like one.
    """
    from .news2 import NotApplicable, Vitals, score

    needed = (m.respiratory_rate, m.spo2, m.systolic_bp, m.pulse, m.temperature)
    if any(v is None for v in needed) or m.alert is None:
        return None
    try:
        return score(Vitals(
            respiratory_rate=m.respiratory_rate, spo2=m.spo2, on_oxygen=m.on_oxygen,
            systolic_bp=m.systolic_bp, pulse=m.pulse, alert=m.alert,
            temperature=m.temperature, age_years=m.age_years,
        ))
    except NotApplicable:
        return None
