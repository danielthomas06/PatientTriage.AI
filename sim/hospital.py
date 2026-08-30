"""Discrete-event emergency department.

One event queue, two finite resources (triage nurses, clinicians), one waiting
room. Patients arrive, wait to be triaged, wait to be seen, and may leave
before either.

The comparison between arms is only meaningful if everything except the triage
policy is held identical. Two things enforce that:

  * both arms consume the same `list[Patient]` from `patients.generate`
  * service-time randomness is drawn from `Random(seed ^ patient.id)`, so each
    patient's draws are keyed to that patient rather than to call order --
    the assistant's extra re-check calls cannot shift the baseline's timings
"""

import heapq
from dataclasses import dataclass, field

from triage import Category

from .patients import Patient

ARRIVAL, TRIAGE_DONE, CLINICIAN_DONE, RECHECK, PATIENCE, RECHECK_DONE = 0, 1, 2, 3, 4, 5


@dataclass(frozen=True, slots=True)
class Department:
    triage_nurses: int = 2
    recheck_minutes: float = 1.5
    """Nurse time one re-assessment costs.

    Without this, re-checking is free and the optimal policy is to re-check
    everyone constantly -- which no department can staff. Charging for it is
    what makes an interval keyed to severity worth anything.
    """
    clinicians: int = 5
    consult_minutes: float = 22.0
    seed: int = 7

    @property
    def capacity_per_hour(self) -> float:
        return self.clinicians * 60.0 / self.consult_minutes


@dataclass
class Record:
    """One patient's journey, for the metrics layer."""

    patient: Patient
    assigned: Category | None = None
    triaged_at: float | None = None
    seen_at: float | None = None
    left_at: float | None = None
    escalations: int = 0

    @property
    def seen(self) -> bool:
        return self.seen_at is not None

    @property
    def wait(self) -> float | None:
        if self.seen_at is None:
            return None
        return self.seen_at - self.patient.arrival

    def truth_at_contact(self) -> Category:
        when = self.seen_at if self.seen_at is not None else self.left_at
        return self.patient.truth_at(when if when is not None else self.patient.arrival)


def _patience(patient: Patient, rng) -> float:
    """Minutes before a patient gives up and leaves.

    Sicker patients stay longer — they feel too unwell to walk out — so the
    people most likely to leave unseen are the ones triaged least urgent,
    which is also where a mis-ranked patient hides.
    """
    base = {
        Category.RED: 600.0,
        Category.ORANGE: 480.0,
        Category.YELLOW: 300.0,
        Category.GREEN: 210.0,
        Category.BLUE: 180.0,
    }[patient.true_category]
    return rng.expovariate(1.0 / base)


def _interval(policy, assigned):
    """Minutes until this patient's next re-check, or None if the policy does
    not re-check at all.

    Policies that key the interval to severity expose `recheck_interval`; the
    baseline, which never looks again, exposes neither.
    """
    fn = getattr(policy, "recheck_interval", None)
    if fn is not None and assigned is not None:
        return fn(assigned)
    return getattr(policy, "recheck_every", None)


def run(patients: list[Patient], policy, dept: Department = Department()) -> list[Record]:
    records = {p.id: Record(patient=p) for p in patients}
    rngs = {p.id: __import__("random").Random(dept.seed ^ (p.id * 2654435761)) for p in patients}

    events: list[tuple[float, int, int]] = []
    counter = 0

    def push(when: float, kind: int, pid: int) -> None:
        nonlocal counter
        counter += 1
        heapq.heappush(events, (when, kind, pid, counter))  # type: ignore[arg-type]

    for p in patients:
        push(p.arrival, ARRIVAL, p.id)

    triage_queue: list[int] = []
    clinic_queue: list[int] = []
    free_nurses = dept.triage_nurses
    free_clinicians = dept.clinicians

    def start_triage(now: float) -> None:
        nonlocal free_nurses
        while free_nurses > 0 and triage_queue:
            pid = triage_queue.pop(0)
            rec = records[pid]
            if rec.left_at is not None:
                continue
            free_nurses -= 1
            outcome = policy.triage(rec.patient, now, rngs[pid])
            rec.assigned = outcome.category
            push(now + outcome.triage_minutes, TRIAGE_DONE, pid)

    def start_clinician(now: float) -> None:
        nonlocal free_clinicians
        while free_clinicians > 0 and clinic_queue:
            clinic_queue.sort(
                key=lambda i: (records[i].assigned or Category.BLUE, records[i].patient.arrival)
            )
            pid = clinic_queue.pop(0)
            rec = records[pid]
            if rec.left_at is not None:
                continue
            free_clinicians -= 1
            rec.seen_at = now
            push(now + dept.consult_minutes * rngs[pid].uniform(0.5, 1.8), CLINICIAN_DONE, pid)

    while events:
        now, kind, pid, _ = heapq.heappop(events)  # type: ignore[misc]
        rec = records[pid]

        if kind == ARRIVAL:
            triage_queue.append(pid)
            push(now + _patience(rec.patient, rngs[pid]), PATIENCE, pid)
            start_triage(now)

        elif kind == TRIAGE_DONE:
            free_nurses += 1
            rec.triaged_at = now
            clinic_queue.append(pid)
            interval = _interval(policy, rec.assigned)
            if interval:
                push(now + interval, RECHECK, pid)
            start_clinician(now)
            start_triage(now)

        elif kind == RECHECK:
            if rec.seen_at is None and rec.left_at is None and rec.assigned is not None:
                # A re-check occupies a nurse. If none is free the check slips
                # rather than happening for nothing -- which is exactly what
                # happens in a real department, and is the cost a flat interval
                # hides.
                if free_nurses <= 0:
                    push(now + 2.0, RECHECK, pid)
                    continue
                free_nurses -= 1
                push(now + dept.recheck_minutes, RECHECK_DONE, pid)

                raised = policy.recheck(rec.patient, rec.assigned, now)
                if raised is not None:
                    rec.assigned = raised
                    rec.escalations += 1
                    start_clinician(now)
                # Re-read the interval: a patient who has just been escalated is
                # re-checked on their NEW category, which is the point of keying
                # the interval to severity in the first place.
                nxt = _interval(policy, rec.assigned)
                if nxt:
                    push(now + nxt, RECHECK, pid)

        elif kind == RECHECK_DONE:
            free_nurses += 1
            start_triage(now)

        elif kind == CLINICIAN_DONE:
            free_clinicians += 1
            start_clinician(now)
            start_triage(now)

        elif kind == PATIENCE:
            if rec.seen_at is None and rec.left_at is None:
                rec.left_at = now
                if pid in triage_queue:
                    triage_queue.remove(pid)
                if pid in clinic_queue:
                    clinic_queue.remove(pid)

    return list(records.values())
