"""Vital-sign scoring (NEWS2, Royal College of Physicians 2017).

Seven parameters, each 0-3, summed. This is a lookup table and nothing more --
it is here to make the point that no model belongs anywhere near it.

Not valid for under-16s (use a paediatric chart) or pregnancy (use an obstetric
chart). Applying the adult table to those groups is a real, plausible harm from
a naive implementation, so ``score`` refuses rather than guessing.
"""

from dataclasses import dataclass


class NotApplicable(Exception):
    """Raised when the adult scale must not be applied to this patient."""


@dataclass(frozen=True, slots=True)
class Vitals:
    respiratory_rate: int | None = None
    spo2: int | None = None
    on_oxygen: bool | None = None
    systolic_bp: int | None = None
    pulse: int | None = None
    alert: bool | None = None       # False = confusion / voice / pain / unresponsive
    temperature: float | None = None

    age_years: int | None = None
    pregnant: bool | None = None

    @property
    def complete(self) -> bool:
        return all(
            v is not None
            for v in (
                self.respiratory_rate, self.spo2, self.on_oxygen,
                self.systolic_bp, self.pulse, self.alert, self.temperature,
            )
        )


def _band(value: float, bands: list[tuple[float, float, int]]) -> int:
    for low, high, points in bands:
        if low <= value <= high:
            return points
    raise ValueError(f"value {value} outside all bands")


_RESP = [(0, 8, 3), (9, 11, 1), (12, 20, 0), (21, 24, 2), (25, 1e9, 3)]
_SPO2 = [(0, 91, 3), (92, 93, 2), (94, 95, 1), (96, 1e9, 0)]
_SBP = [(0, 90, 3), (91, 100, 2), (101, 110, 1), (111, 219, 0), (220, 1e9, 3)]
_PULSE = [(0, 40, 3), (41, 50, 1), (51, 90, 0), (91, 110, 1), (111, 130, 2), (131, 1e9, 3)]
_TEMP = [(0, 35.0, 3), (35.1, 36.0, 1), (36.1, 38.0, 0), (38.1, 39.0, 1), (39.1, 1e9, 2)]


@dataclass(frozen=True, slots=True)
class Score:
    total: int
    parts: dict[str, int]

    @property
    def any_parameter_is_three(self) -> bool:
        """A single 3 warrants urgent review even when the total looks benign.

        This is the rule humans skip most often -- a lone respiratory rate of
        26 on a patient who is otherwise well.
        """
        return any(p == 3 for p in self.parts.values())

    @property
    def band(self) -> str:
        if self.total >= 7:
            return "high"
        if self.total >= 5:
            return "medium"
        if self.any_parameter_is_three:
            return "low-medium"
        if self.total >= 1:
            return "low"
        return "routine"

    def explain(self) -> str:
        driving = sorted(
            ((v, k) for k, v in self.parts.items() if v > 0), reverse=True
        )
        if not driving:
            return "all parameters within normal range"
        return ", ".join(f"{name} +{pts}" for pts, name in driving)


def score(vitals: Vitals) -> Score:
    """Compute the score. Raises NotApplicable rather than guessing."""
    if vitals.age_years is not None and vitals.age_years < 16:
        raise NotApplicable("adult scale is not validated under 16 -- use a paediatric chart")
    if vitals.pregnant:
        raise NotApplicable("adult scale is not validated in pregnancy -- use an obstetric chart")
    if not vitals.complete:
        raise ValueError("all seven parameters are required; partial sets are not scored")

    parts = {
        "respiratory rate": _band(vitals.respiratory_rate, _RESP),
        "oxygen saturation": _band(vitals.spo2, _SPO2),
        "supplemental oxygen": 2 if vitals.on_oxygen else 0,
        "systolic bp": _band(vitals.systolic_bp, _SBP),
        "pulse": _band(vitals.pulse, _PULSE),
        "consciousness": 0 if vitals.alert else 3,
        "temperature": _band(vitals.temperature, _TEMP),
    }
    return Score(total=sum(parts.values()), parts=parts)
