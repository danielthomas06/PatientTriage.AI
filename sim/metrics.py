"""Scoring a run.

The headline metric is **under-triage**, not accuracy. Acuity is heavily
imbalanced, so a policy that calls everyone Urgent scores well on accuracy and
kills people; a policy that calls everyone Immediate scores well on
under-triage and paralyses the department. Both are reported, so neither can
be gamed without the other showing it.

Under-triage is measured against ground truth **at the moment of clinician
contact**, not at arrival. That is what makes a patient who deteriorated
unnoticed in the waiting room count as under-triaged, which is the entire
argument for re-scoring.
"""

from dataclasses import dataclass
from statistics import mean

from triage import Category

from .hospital import Record


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, int(q * len(s)))
    return s[idx]


@dataclass(frozen=True, slots=True)
class Metrics:
    arm: str
    n: int
    seen: int
    left_unseen: int

    under_triage: float
    critical_under_triage: float
    over_triage: float

    wait_by_truth: dict[Category, float]
    p90_by_truth: dict[Category, float]
    escalations: int
    missed_deteriorations: int

    def report(self) -> str:
        lines = [
            f"{self.arm:>10s}  n={self.n}  seen={self.seen}  left unseen={self.left_unseen}",
            f"{'':>10s}  under-triage {_pct(self.under_triage)}"
            f"   critical {_pct(self.critical_under_triage)}"
            f"   over-triage {_pct(self.over_triage)}",
            f"{'':>10s}  escalations caught {self.escalations}"
            f"   deteriorations missed {self.missed_deteriorations}",
        ]
        for cat in Category:
            w = self.wait_by_truth.get(cat)
            if w is None:
                continue
            lines.append(
                f"{'':>10s}    {cat.label:<12s} mean {w:6.1f} min"
                f"   p90 {self.p90_by_truth.get(cat, float('nan')):6.1f} min"
            )
        return "\n".join(lines)


def summarise(arm: str, records: list[Record]) -> Metrics:
    seen = [r for r in records if r.seen]
    left = [r for r in records if r.left_at is not None]

    under = crit = over = 0
    for r in seen:
        truth = r.truth_at_contact()
        assigned = r.assigned or Category.BLUE
        if assigned > truth:  # numerically larger = less urgent = ranked too low
            under += 1
            if truth <= Category.ORANGE:
                crit += 1
        elif assigned < truth:
            over += 1

    waits: dict[Category, list[float]] = {}
    for r in seen:
        waits.setdefault(r.truth_at_contact(), []).append(r.wait or 0.0)

    missed = 0
    for r in records:
        p = r.patient
        if p.deteriorates_at is None:
            continue
        end = r.seen_at if r.seen_at is not None else r.left_at
        if end is None or end < p.deteriorates_at:
            continue
        if (r.assigned or Category.BLUE) > p.truth_at(end):
            missed += 1

    denom = len(seen) or 1
    return Metrics(
        arm=arm,
        n=len(records),
        seen=len(seen),
        left_unseen=len(left),
        under_triage=under / denom,
        critical_under_triage=crit / denom,
        over_triage=over / denom,
        wait_by_truth={c: mean(v) for c, v in waits.items()},
        p90_by_truth={c: _quantile(v, 0.9) for c, v in waits.items()},
        escalations=sum(r.escalations for r in records),
        missed_deteriorations=missed,
    )


def compare(a: Metrics, b: Metrics) -> str:
    """Baseline vs assistant, as deltas."""

    def delta(x: float, y: float, unit: str = "pp") -> str:
        d = (y - x) * (100 if unit == "pp" else 1)
        return f"{d:+.1f}{unit}"

    lines = [
        f"  under-triage           {_pct(a.under_triage)} -> {_pct(b.under_triage)}"
        f"   ({delta(a.under_triage, b.under_triage)})",
        f"  critical under-triage  {_pct(a.critical_under_triage)} -> {_pct(b.critical_under_triage)}"
        f"   ({delta(a.critical_under_triage, b.critical_under_triage)})",
        f"  over-triage            {_pct(a.over_triage)} -> {_pct(b.over_triage)}"
        f"   ({delta(a.over_triage, b.over_triage)})",
        f"  left unseen            {a.left_unseen} -> {b.left_unseen}",
        f"  deteriorations missed  {a.missed_deteriorations} -> {b.missed_deteriorations}",
    ]
    for cat in (Category.RED, Category.ORANGE, Category.YELLOW, Category.GREEN):
        wa, wb = a.wait_by_truth.get(cat), b.wait_by_truth.get(cat)
        if wa is None or wb is None:
            continue
        lines.append(
            f"  wait, {cat.label:<12s}     {wa:6.1f} -> {wb:6.1f} min"
            f"   ({delta(wa, wb, ' min')})"
        )
    return "\n".join(lines)
